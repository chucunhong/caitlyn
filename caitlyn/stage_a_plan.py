"""Stage A: role-typed plan induction (paper Sec. 3.1, supplement Table A).

A1  resolving set: pi_role candidates (K samples, majority) + relevance and non-proxy tests
A2  plan: LLM-guided BFS from the protected attributes with association scores (pi_edge, pi_cycle)
A3  evidence gate on edges leaving S; likelihood-ratio test for edges the LLM missed
"""
import json
import warnings
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2

from .assoc import AssocEncoder
from .common import cell_of, cv_perf, folds, holm, is_numeric, perm_p, stable_seed
from .data import rows_text
from .prompts import render


# ============================================================ A1: resolving set
def role_candidates(llm, meta, train, ccfg, seed):
    """pi_role sampled K times at temperature 0.7; keep columns proposed in >= role_min_votes samples."""
    S, Y = meta.protected, meta.label
    cols = [c for c in train.columns if c not in S and c != Y]
    rows = train.sample(ccfg["n_example_rows"], random_state=seed)
    desc = "\n".join(f"- {c}: {meta.describe(c)}" for c in train.columns)
    prompt = render("role", task_description=meta.task, protected=", ".join(S), label=Y,
                    column_descriptions=desc, rows=rows_text(rows, with_id=False))
    votes, pvotes, raw = Counter(), Counter(), []
    for k in range(int(ccfg["K"])):
        out = llm.call(prompt, temperature=ccfg["role_temperature"], seed_tag=f"role|{seed}|{k}",
                       stage="A_role", hint={"kind": "role", "columns": cols})
        raw.append(out)
        res = {d.get("column") for d in out.get("resolving", []) if isinstance(d, dict)} & set(cols)
        prx = {d.get("column") for d in out.get("proxy_candidates", []) if isinstance(d, dict)} & set(cols)
        votes.update(res)
        pvotes.update(prx)
    cand = [c for c in cols if votes[c] >= ccfg["role_min_votes"]]
    llm_proxies = [c for c in cols if pvotes[c] >= ccfg["role_min_votes"] and c not in cand]
    return cand, llm_proxies, {"votes": dict(votes), "proxy_votes": dict(pvotes), "raw": raw}


def proxy_capacity(df, S, r, seed, fi_cache=None):
    """PC_r = max(2 AUC - 1) over each S_i and the intersectional cell; single feature r."""
    targets = {s: df[s].astype(str) for s in S}
    targets["cell"] = cell_of(df, S)
    pcs = {k: 2 * cv_perf(df[[r]], t, seed) - 1 for k, t in targets.items()}
    k = max(pcs, key=pcs.get)
    return float(pcs[k]), k, pcs


def relevance(df, S, y, r, seed, B=200, delta=None, h=10):
    """AUC gain of adding r to S for predicting Y; null = permutations of r within each cell.

    A candidate whose gain is below delta fails the test whatever its p-value, so it is not permuted (p = 1).
    Otherwise the p-value is the sequential Monte Carlo estimate of Besag & Clifford (1991): permutations stop
    once h null statistics reach the observed gain (p = h / L after L permutations), else p = (1 + #) / (B + 1)."""
    rng = np.random.default_rng(stable_seed(seed, "relevance", r))
    fi = folds(df[y], seed)                                   # same folds for observed and null
    auc0 = cv_perf(df[S], df[y], seed, fi)
    gain = cv_perf(df[S + [r]], df[y], seed, fi) - auc0
    if delta is not None and gain < delta:
        return float(gain), 1.0
    groups = list(cell_of(df, S).groupby(cell_of(df, S)).indices.values())
    vals = df[r].values
    exceed = 0
    for L in range(1, B + 1):
        perm = vals.copy()
        for idx in groups:
            perm[idx] = vals[rng.permutation(idx)]
        exceed += cv_perf(df[S].assign(**{r: perm}), df[y], seed, fi) - auc0 >= gain
        if exceed >= h:
            return float(gain), h / L
    return float(gain), (1 + exceed) / (B + 1)


def effective_B(B, m, alpha):
    """Smallest permutation count for which a Holm-corrected family of m tests can reach alpha.
    With p = (1 + #null >= obs) / (B + 1), the smallest adjusted p is m / (B + 1); we require it to be
    at most alpha / 2, i.e. B >= 2 m / alpha (e.g. 39 gate tests on Adult -> B = 1560 instead of 200)."""
    return max(int(B), int(np.ceil(2 * m / alpha)))


def select_resolving(df, S, y, candidates, seed, delta=0.01, tau_p=0.2, alpha=0.05, B=200, n_jobs=1):
    S = list(S)
    B = effective_B(B, len(candidates), alpha)
    res = Parallel(n_jobs=n_jobs)(delayed(relevance)(df, S, y, r, seed, B, delta) for r in candidates)
    rel = dict(zip(candidates, res))
    p_adj = holm({r: p for r, (g, p) in rel.items()}) if rel else {}
    R, rejected, details = [], {}, {}
    for r in candidates:
        gain = rel[r][0]
        pc, by, pcs = proxy_capacity(df, S, r, seed)
        details[r] = dict(gain=gain, p=rel[r][1], p_holm=p_adj[r], pc=pc, pc_by=by)
        if gain < delta or p_adj[r] >= alpha:
            rejected[r] = f"not relevant (gain={gain:.3f}, p={p_adj[r]:.3f})"
        elif pc > tau_p:
            rejected[r] = f"proxy of {by} (PC={pc:.2f})"
        else:
            R.append(r)
    details["_B_effective"] = B
    return R, rejected, details


# ============================================================ A2: plan induction
def _graph_text(G):
    if G.number_of_edges() == 0:
        return "(empty)"
    return "\n".join(f"{u} -> {v}: {d.get('rationale', '')}" for u, v, d in G.edges(data=True))


def roles_of(meta, R, proxy_cands, columns):
    roles = {}
    for c in columns:
        if c in meta.protected:
            roles[c] = "protected"
        elif c == meta.label:
            roles[c] = "label"
        elif c in R:
            roles[c] = "resolving"
        elif c in proxy_cands:
            roles[c] = "proxy candidate"
        else:
            roles[c] = "other"
    return roles


def break_cycles(llm, G, enc, ccfg, seed, log):
    while not nx.is_directed_acyclic_graph(G):
        cyc = nx.find_cycle(G)
        text = "\n".join(f"{u} -> {v}: {G.edges[u, v].get('rationale', '')}" for u, v in cyc)
        out = llm.call(render("cycle", cycle_with_rationales=text), temperature=ccfg["struct_temperature"],
                       seed_tag=f"cycle|{seed}", stage="A_plan", hint={"kind": "cycle", "cycle": cyc})
        rem = out.get("remove") if isinstance(out, dict) else None
        rem = tuple(rem) if isinstance(rem, list) and len(rem) == 2 else None
        if rem not in [tuple(e) for e in cyc]:            # invalid answer: drop the weakest-association edge
            rem = min(cyc, key=lambda e: enc.score(e[0], e[1]))
        G.remove_edge(*rem)
        log.append({"cycle": [list(e) for e in cyc], "removed": list(rem),
                    "reason": out.get("reason", "") if isinstance(out, dict) else ""})


def build_plan(llm, meta, train, roles, ccfg, seed):
    """LLM-guided BFS. Queue starts at the protected roots; when it empties, unvisited columns are
    enqueued so every column is visited once. S are roots; the label is a sink (label_is_sink)."""
    S, Y = list(meta.protected), meta.label
    cols = list(train.columns)
    enc = AssocEncoder(train)
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    rows = rows_text(train.sample(ccfg["n_example_rows"], random_state=seed + 1), with_id=False)
    queue, visited, cyc_log, proposals = list(S), set(), [], {}
    roles_text = json.dumps(roles)
    while True:
        if not queue:
            rest = [c for c in cols if c not in visited and not (c == Y and ccfg.get("label_is_sink", True))]
            if not rest:
                break
            queue.append(rest[0])
        u = queue.pop(0)
        if u in visited:
            continue
        visited.add(u)
        assoc = {v: round(enc.score(u, v), 3) for v in cols if v != u}
        prompt = render("edge", u=u, role_u=roles[u], graph=_graph_text(G), roles=roles_text,
                        assoc=json.dumps(assoc), rows=rows)
        cand = [c for c in cols if c not in S and c != u]
        out = llm.call(prompt, temperature=ccfg["struct_temperature"], seed_tag=f"edge|{seed}", stage="A_plan",
                       hint={"kind": "edge", "candidates": cand})
        kids = out.get("children", []) if isinstance(out, dict) else []
        proposals[u] = kids
        for k in kids:
            if not isinstance(k, dict):
                continue
            v = k.get("column")
            if v not in cand or G.has_edge(u, v):
                continue
            G.add_edge(u, v, rationale=k.get("rationale", ""), source="llm")
            if v not in visited and v not in queue and not (v == Y and ccfg.get("label_is_sink", True)):
                queue.append(v)
        break_cycles(llm, G, enc, ccfg, seed, cyc_log)
    if ccfg.get("label_is_sink", True):
        G.remove_edges_from(list(G.out_edges(Y)))
    G.remove_edges_from([(u, s) for s in S for u in list(G.predecessors(s))])
    return G, {"cycles": cyc_log, "proposals": proposals}


def pc_plan(train, meta, seed, alpha=0.05):
    """Reference plan from the PC algorithm (causal-learn), oriented with S as roots and Y as sink."""
    from causallearn.search.ConstraintBased.PC import pc
    S, Y = list(meta.protected), meta.label
    cols = list(train.columns)
    X = np.column_stack([train[c].values if is_numeric(train[c]) else pd.factorize(train[c].astype(str))[0]
                         for c in cols]).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc(X, alpha, "fisherz", show_progress=False)
    A = cg.G.graph
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    rank = {c: (0 if c in S else 2 if c == Y else 1) for c in cols}
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            a, b = cols[i], cols[j]
            if A[j, i] == 1 and A[i, j] == -1:            # i -> j
                G.add_edge(a, b, source="pc")
            elif A[i, j] == -1 and A[j, i] == -1 and i < j:  # undirected: orient by role rank, then order
                u, v = (a, b) if (rank[a], i) <= (rank[b], j) else (b, a)
                G.add_edge(u, v, source="pc")
    G.remove_edges_from([(u, s) for s in S for u in list(G.predecessors(s))])
    if Y in G:
        G.remove_edges_from(list(G.out_edges(Y)))
    while not nx.is_directed_acyclic_graph(G):
        G.remove_edge(*nx.find_cycle(G)[-1])
    return G


# ============================================================ A3: evidence gate
def lr_test(df, v, parents, u):
    """LRT: v ~ pa(v) versus v ~ pa(v) + u. MNLogit for categorical v, OLS for numerical v."""
    import statsmodels.api as sm

    def design(cols):
        if not cols:
            return np.ones((len(df), 1))
        X = df[cols].copy()
        for c in cols:
            if not is_numeric(X[c]):
                X[c] = X[c].astype(str)
        return sm.add_constant(pd.get_dummies(X, drop_first=True, dtype=float), has_constant="add")

    X0, X1 = design(parents), design(parents + [u])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_numeric(df[v]):
                m0, m1 = sm.OLS(df[v].astype(float), X0).fit(), sm.OLS(df[v].astype(float), X1).fit()
            else:
                yv = pd.Series(pd.factorize(df[v].astype(str))[0])
                m0 = sm.MNLogit(yv, X0).fit(disp=0, method="bfgs", maxiter=300)
                m1 = sm.MNLogit(yv, X1).fit(disp=0, method="bfgs", maxiter=300)
            stat = 2 * (m1.llf - m0.llf)
            dof = max(1, int(round(m1.df_model - m0.df_model)))
        if not np.isfinite(stat):            # separation (v already determined by its parents): no evidence for u
            return 1.0
        return float(chi2.sf(max(stat, 0.0), dof))
    except Exception:
        return 1.0


def _perm_null(enc, s, v, B, seed):
    rng = np.random.default_rng(stable_seed(seed, "gate", s, v))
    x = enc.enc[s]
    return [enc.score(s, v, a_values=rng.permutation(x)) for _ in range(B)]


def gate_tests(df, S, seed, B=200, n_jobs=1, alpha=0.05):
    """Permutation test of a(s, v) for every s in S and every non-protected v; Holm over the family."""
    enc = AssocEncoder(df)
    pairs = [(s, v) for s in S for v in df.columns if v not in S]
    B = effective_B(B, len(pairs), alpha)
    obs = {k: enc.score(*k) for k in pairs}
    nulls = Parallel(n_jobs=n_jobs)(delayed(_perm_null)(enc, s, v, B, seed) for s, v in pairs)
    pv = {k: perm_p(obs[k], nl) for k, nl in zip(pairs, nulls)}
    return obs, pv, holm(pv)


def evidence_gate(G, df, S, seed, alpha=0.05, tau=0.05, B=200, n_jobs=1, lrt_max_rows=20000, tests=None):
    G = G.copy()
    obs, pv, p_adj = tests or gate_tests(df, S, seed, B, n_jobs, alpha)
    supported = {k for k in obs if p_adj[k] < alpha and obs[k] >= tau}
    rejected = []
    for (s, v) in list(G.edges()):
        if s in S and (s, v) not in supported:
            rejected.append({"edge": [s, v], "assoc": obs.get((s, v)), "p_holm": p_adj.get((s, v)),
                             "rationale": G.edges[s, v].get("rationale", "")})
            G.remove_edge(s, v)                               # prior-driven edge
    sub = df.sample(min(len(df), lrt_max_rows), random_state=seed) if len(df) > lrt_max_rows else df
    sub = sub.reset_index(drop=True)
    missing = [k for k in supported if not G.has_edge(*k)]
    lrt = {k: lr_test(sub, k[1], list(G.predecessors(k[1])), k[0]) for k in missing}
    lrt_adj = holm(lrt) if lrt else {}
    added = []
    for (s, v), p in sorted(lrt_adj.items(), key=lambda kv: kv[1]):
        if p < alpha:
            G.add_edge(s, v, rationale="added by likelihood-ratio test", source="lrt")
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(s, v)
            else:
                added.append({"edge": [s, v], "p_holm": p, "assoc": obs[(s, v)]})
    return G, rejected, added, {"assoc": obs, "p": pv, "p_holm": p_adj}


def complete_label_parents(G, df, S, Y, seed, alpha=0.05, lrt_max_rows=20000, exclude=()):
    """Data-driven parents of the label the LLM plan missed: for every non-protected column u outside pa(Y),
    an LRT of Y ~ pa(Y) versus Y ~ pa(Y) + u; edges u -> Y with Holm-adjusted p < alpha are added.
    Protected attributes are left to the evidence gate; added edges on an unresolved path are audited and
    cut by Stages B-C like any other edge."""
    G = G.copy()
    sub = df.sample(min(len(df), lrt_max_rows), random_state=seed) if len(df) > lrt_max_rows else df
    sub = sub.reset_index(drop=True)
    pa = sorted(G.predecessors(Y))
    cand = [u for u in df.columns if u not in S and u != Y and u not in pa and u not in exclude]
    p_adj = holm({u: lr_test(sub, Y, pa, u) for u in cand}) if cand else {}
    added = []
    for u, p in sorted(p_adj.items(), key=lambda kv: kv[1]):
        if p < alpha:
            G.add_edge(u, Y, rationale="label parent added by likelihood-ratio test", source="lrt_label")
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(u, Y)
            else:
                added.append({"edge": [u, Y], "p_holm": p})
    return G, added
