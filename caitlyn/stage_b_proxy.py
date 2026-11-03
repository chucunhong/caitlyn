"""Stage B: generator-side proxy audit (paper Sec. 3.2, Eq. 1).

PC_u  : proxy capacity, single-feature LightGBM predicting each S_i or the cell from u
PU_uv : TV between the call for child v with u visible and with u = UNKNOWN, minus the TV
        between two independent sample sets with u visible (sampling noise), clipped at 0
PRS_u : PC_u * max_v PU_uv;  u is a proxy if PRS_u > median over audited nodes
"""
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np

from .common import is_numeric
from .stage_a_plan import proxy_capacity
from .stage_d_generate import Generator


def unresolved_graph(G, R):
    H = G.copy()
    H.remove_nodes_from([r for r in R if r in H])
    return H


def audited_nodes(G, S, R, Y):
    """Nodes not in S, R or {Y} lying on some path S_i ~> Y that avoids R."""
    H = unresolved_graph(G, R)
    if Y not in H:
        return []
    from_S = set().union(*[nx.descendants(H, s) for s in S if s in H]) if S else set()
    to_Y = nx.ancestors(H, Y)
    return sorted(u for u in (from_S & to_Y) if u not in S and u != Y)


def tv(x, y, bins=None):
    x = [a for a in x if a is not None and a == a]
    y = [a for a in y if a is not None and a == a]
    if not x or not y:
        return 0.0
    if bins is not None:
        x, y = np.digitize(np.asarray(x, float), bins), np.digitize(np.asarray(y, float), bins)
    x, y = np.asarray(x, dtype=object), np.asarray(y, dtype=object)
    keys = set(x.tolist()) | set(y.tolist())
    return 0.5 * sum(abs(np.mean(x == k) - np.mean(y == k)) for k in keys)


def sample_node(gen: Generator, v, ctx, ex, J, tag, unknown=()):
    """J independent samples of node v for each context row -> list (len M) of lists (len J).
    The J draws run concurrently; the LLM client's semaphore still caps the number of open requests."""
    with ThreadPoolExecutor(J) as pool:
        draws = list(pool.map(lambda j: gen.gen_column(v, ctx, seed_tag=f"{tag}|j{j}", ctx_cols=list(ctx.columns),
                                                       fixed_exemplars=ex, unknown_cols=unknown), range(J)))
    return [[d.iloc[i] for d in draws] for i in range(len(ctx))]


def proxy_use(gen, u, v, G, train, seed, M=100, J=5):
    pa = list(G.predecessors(v))
    ctx = train.sample(min(M, len(train)), random_state=seed)[pa].reset_index(drop=True)
    ex = train.sample(10, random_state=seed + 1)[pa + [v]]
    bins = np.quantile(train[v], np.linspace(0.1, 0.9, 9)) if is_numeric(train[v]) else None
    base = f"B|{seed}|{u}->{v}"
    ctx_b = ctx.drop(columns=[u]).assign(**{u: "UNKNOWN"})[pa]
    with ThreadPoolExecutor(3) as pool:                  # conditions (a), (a'), (b) concurrently
        A, A2, Bu = pool.map(lambda job: sample_node(gen, v, job[0], ex, J, base + job[1]),
                             [(ctx, "|a"), (ctx, "|a2"), (ctx_b, "|b")])
    d = [tv(A[i], Bu[i], bins) - tv(A[i], A2[i], bins) for i in range(len(ctx))]
    return max(0.0, float(np.mean(d))), float(np.mean([tv(A[i], Bu[i], bins) for i in range(len(ctx))]))


def proxy_audit(llm, meta, train, G, R, ccfg, seed, extra_candidates=(), skip_pu=False):
    S, Y = list(meta.protected), meta.label
    nodes = audited_nodes(G, S, R, Y)
    H = unresolved_graph(G, R)
    gen = Generator(llm, meta, train, G, ccfg, seed, stage="B_proxy")
    log = {}
    for u in nodes:
        pc, by, pcs = proxy_capacity(train, S, u, seed)
        pu = {}
        if not skip_pu:
            to_Y = nx.ancestors(H, Y) | {Y}
            for v in H.successors(u):
                if v in S or v not in to_Y:          # only edges on an unresolved path to Y can matter
                    continue
                pu[v], raw = proxy_use(gen, u, v, G, train, seed, int(ccfg["M"]), int(ccfg["J"]))
        max_pu = max(pu.values()) if pu else 0.0
        log[u] = dict(pc=pc, pc_by=by, pc_per_target=pcs, pu=pu, max_pu=max_pu, prs=pc * max_pu,
                      llm_candidate=u in extra_candidates)
    P = select_proxies(log, R)
    return P, {"audited": nodes, "nodes": log, "proxies": P}


def select_proxies(log, R, key="prs"):
    vals = {u: d[key] for u, d in log.items() if u not in R}
    if not vals:
        return []
    if len(vals) == 1:                     # median of one value: keep it if it carries any signal
        return [u for u, x in vals.items() if x > 0]
    med = float(np.median(list(vals.values())))
    return sorted(u for u, x in vals.items() if x > med)


def per_attribute_proxies(stage_b_log, train, S_i, R, seed):
    """Proxy set for one attribute: PC recomputed against S_i alone, PU reused (it does not depend on S)."""
    log = {}
    for u, d in stage_b_log["nodes"].items():
        pc, _, _ = proxy_capacity(train, [S_i], u, seed)
        log[u] = dict(prs=pc * d["max_pu"])
    return select_proxies(log, R)
