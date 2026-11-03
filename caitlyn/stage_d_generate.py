"""Stage D: leakage-controlled synthesis (paper Sec. 3.4).

Controls (each can be switched off for the leakage analysis and ablations):
  (i)   context="parent"      : ctx(v) = pa_G'(v)   ("full": every previously generated column)
  (ii)  grouping="layer"      : one call per topological layer and group (A / non-A), see call_groups();
        grouping="column": one call per column; joint_call=True: all non-root columns in one call
  (iii) exemplars="matched", batching="mixed": exemplars showing only pa(v) and v; each row of a batch
        brings one training row drawn among its match_k nearest neighbours on pa(v) (the choice depends on
        pa(v) only); batches mix cells   ("independent": exemplars drawn uniformly from D_train;
        "cell": exemplars and batches conditioned on the cell)
  (iv)  lam                   : root cell quotas pi_lambda(s) = (1-lam) p_hat(s) + lam/|C|
  (v)   novelty_rounds        : rows equal to a training row or to another synthetic row are regenerated
        (every non-root column, same cell), up to novelty_rounds times; rows still not novel are dropped
"""
import json
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

from .common import UNKNOWN, cell_of, stable_seed
from .data import cast_like, column_domains, domain_text, fmt_value, rows_text, validate_value
from .llm import LLMError
from .prompts import render

SUR = "__sur"


def _row_id(k):
    """Row index from an id the model may write as 3, "3", "id_3", "row 3" or "#3"; None otherwise."""
    if isinstance(k, (int, np.integer)):
        return int(k)
    digits = "".join(ch for ch in str(k) if ch.isdigit()) if k is not None else ""
    return int(digits) if digits else None


def answer_rows(o, n, cols):
    """{row index: value or list of values} from the answer formats models produce:
    {"values": {id: ...}}, {"values": [{"id":.., "values"/"value"/<col>: ..}, ...]}, a bare list of such
    objects, or {id: ...} without the "values" wrapper; entries without a usable id are taken by position."""
    body = o.get("values", o.get("rows", o)) if isinstance(o, dict) else o
    out = {}
    if isinstance(body, dict):
        for k, v in body.items():
            j = _row_id(k)
            if j is not None:
                out.setdefault(j, v)
    elif isinstance(body, list):
        for pos, item in enumerate(body):
            if isinstance(item, dict):
                j = _row_id(item.get("id"))
                v = item.get("values", item.get("value"))
                if v is None and any(c in item for c in cols):
                    v = [item.get(c) for c in cols]
            else:
                j, v = None, item
            out.setdefault(pos if j is None else j, v)
    if out and min(out) == 1 and max(out) == n and 0 not in out:     # 1-based ids
        out = {j - 1: v for j, v in out.items()}
    return out


def topo(G: nx.DiGraph):
    return list(nx.lexicographical_topological_sort(G))


def sample_cells(p_hat: dict, m, lam=0.3, rng=None):
    cells = list(p_hat)
    pi = np.array([(1 - lam) * p_hat[c] + lam / len(cells) for c in cells])
    idx = rng.choice(len(cells), size=m, p=pi / pi.sum())
    return [cells[i] for i in idx]


class Generator:
    def __init__(self, llm, meta, train: pd.DataFrame, plan: nx.DiGraph, ccfg: dict, seed: int,
                 context="parent", exemplars="independent", batching="mixed", joint_call=False,
                 lam=None, stage="D_generate", R=None, grouping=None):
        self.llm, self.meta, self.train, self.G, self.c, self.seed = llm, meta, train, plan, ccfg, seed
        self.grouping = grouping or ccfg.get("grouping", "column")
        self.A = None
        if R is not None:                         # nodes reaching Y without passing R (needed for layer groups)
            H = plan.copy()
            H.remove_nodes_from([r for r in R if r in H])
            self.A = (nx.ancestors(H, meta.label) | {meta.label}) if meta.label in H else set()
        self.S, self.Y = list(meta.protected), meta.label
        self.context, self.exemplars, self.batching, self.joint_call = context, exemplars, batching, joint_call
        self.lam = ccfg["lam"] if lam is None else lam
        self.stage = stage
        self.dom = column_domains(train)
        self.bs, self.n_ex = int(ccfg["batch_size"]), int(ccfg["n_exemplars"])
        self.cell_train = cell_of(train, self.S)
        self.order = topo(plan)
        self.real_order = [v for v in self.order if not plan.nodes[v].get("hidden")]
        self.stats = {"invalid": 0, "values": 0, "dropped_rows": 0}
        self.workers = int(llm.cfg.get("max_concurrency", 8))
        self.match_k = int(ccfg.get("match_k", 10))
        self.novelty_rounds = int(ccfg.get("novelty_rounds", 0))
        self._codes = {}                          # categorical column -> {value: code}, for neighbour search

    # ------------------------------------------------------------------ helpers
    def target_of(self, v):
        return self.G.nodes[v].get("surrogate_of", v)

    def ctx_cols(self, v, available):
        """Context column names (as stored in the working frame) for node v."""
        if self.context == "full" and not self.G.nodes[v].get("hidden"):
            pos = self.real_order.index(v)
            return [c for c in self.real_order[:pos] if c in available]
        return [p for p in self.G.predecessors(v)]

    def masked_of(self, v):
        return list(self.G.nodes[v].get("masked", []))

    def _display(self, cols):
        """Working-frame column -> name shown to the LLM (surrogates are shown under the original name)."""
        return [self.G.nodes[c].get("surrogate_of", c) if c in self.G else c for c in cols]

    def _exemplar_pool(self, cell=None):
        if self.exemplars == "cell" and cell is not None:
            pool = self.train[self.cell_train == cell]
            if len(pool) >= self.n_ex:
                return pool
        return self.train

    def max_tokens(self, n_rows, cols):
        """Answer budget: about 2.5 times a complete (possibly indented) answer, from the longest value of each
        target column, so that a looping answer stops early but a valid one is never cut."""
        if not hasattr(self, "_val_tokens"):
            self._val_tokens = {c: 4 + max(len(str(x)) for x in self.train[c].dropna().unique()[:5000]) / 3.0
                                for c in self.train.columns}
        cols = [cols] if isinstance(cols, str) else cols
        per_row = 8 + sum(self._val_tokens.get(self.target_of(c), 8) for c in cols)
        return int(400 + 2.5 * n_rows * per_row)

    def _enc(self, col, values):
        """Encode values of a (shown) column for the Gower distance: scaled floats or category codes."""
        d = self.dom[col]
        if d["type"] == "numeric":
            span = (d["max"] - d["min"]) or 1.0
            return pd.to_numeric(pd.Series(values), errors="coerce").astype(float).values / span
        if col not in self._codes:
            self._codes[col] = {v: i for i, v in enumerate(pd.unique(self.train[col].astype(str)))}
        return pd.Series(values).astype(str).map(self._codes[col]).fillna(-1).values

    def matched_exemplars(self, ctx: pd.DataFrame, targets, rng, pool=None):
        """One exemplar per row of the batch: a training row drawn uniformly among the match_k nearest
        neighbours of the row on the shown context columns (Gower distance, random tie-breaking).
        Exemplars show the context columns and the targets only. Without context: a uniform sample."""
        pool = self.train if pool is None else pool
        ctx = ctx.loc[:, ~ctx.columns.duplicated()]
        cols = [c for c in ctx.columns if c in pool.columns]
        if not cols:
            return pool.sample(min(self.bs, len(pool)), random_state=int(rng.integers(1 << 30)))[list(targets)]
        D = np.zeros((len(ctx), len(pool)))
        for c in cols:
            q, r = self._enc(c, ctx[c].values), self._enc(c, pool[c].values)
            if self.dom[c]["type"] == "numeric":
                D += np.nan_to_num(np.abs(q[:, None] - r[None, :]), nan=1.0)
            else:
                D += q[:, None] != r[None, :]
        D += rng.random(D.shape) * 1e-6                  # random tie-breaking among equal distances
        k = min(self.match_k, len(pool))
        near = np.argpartition(D, k - 1, axis=1)[:, :k]
        pick = near[np.arange(len(ctx)), rng.integers(0, k, len(ctx))]
        return pool.iloc[pd.unique(pick)][cols + [t for t in targets if t not in cols]]

    # ------------------------------------------------------------------ one column
    def _call_batch(self, v, ctx_rows: pd.DataFrame, ex: pd.DataFrame, seed_tag: str):
        """One pi_gen call for node v on <= batch_size rows. Returns {row_index: value}."""
        tgt = self.target_of(v)
        d = self.dom[tgt]
        local = ctx_rows.reset_index(drop=True)
        shown_cols = list(local.columns)
        if shown_cols:
            prompt = render("gen_matched" if self.exemplars == "matched" and self.stage.startswith("D") else "gen",
                            target=tgt, target_description=self.meta.describe(tgt), domain=domain_text(d),
                            parent_columns=", ".join(shown_cols), rows_parent_only=rows_text(local),
                            exemplars=rows_text(ex, with_id=False))
        else:
            prompt = render("gen_root", target=tgt, target_description=self.meta.describe(tgt),
                            domain=domain_text(d), ids=json.dumps(list(range(len(local)))),
                            exemplars=rows_text(ex[[tgt]], with_id=False))
        try:
            out = self.llm.call(prompt, temperature=self.c["gen_temperature"], top_p=self.c["gen_top_p"],
                                seed_tag=seed_tag, stage=self.stage, max_tokens=self.max_tokens(len(local), tgt),
                                hint={"kind": "gen", "domain": d, "ids": list(range(len(local)))})
        except LLMError:                                # unparseable answer: the rows are retried
            self.stats["failed_calls"] = self.stats.get("failed_calls", 0) + 1
            out = {}
        vals = answer_rows(out, len(local), [tgt])
        res = {}
        for j, idx in enumerate(ctx_rows.index):
            r = vals.get(j)
            if isinstance(r, dict):                     # {"<column>": value}
                r = r.get(tgt, next(iter(r.values()), None))
            res[idx] = validate_value(r[0] if isinstance(r, list) and len(r) == 1 else r, d)
        return res

    def gen_column(self, v, frame: pd.DataFrame, seed_tag: str, ctx_cols=None, fixed_exemplars=None,
                   exemplar_pool=None, neighbours: pd.DataFrame = None, probe_per_batch=None,
                   unknown_cols=()):
        """Generate node v for every row of `frame` (index = row ids). Returns a Series (NaN = failed).

        fixed_exemplars : DataFrame shown in every call (Stage B)
        exemplar_pool   : pool exemplars are drawn from (leakage probe); default per self.exemplars
        neighbours      : context rows used to fill each batch next to probe rows (leakage probe)
        unknown_cols    : context columns shown as UNKNOWN (Stage B condition (b), surrogates)
        """
        tgt = self.target_of(v)
        ctx_cols = self.ctx_cols(v, frame.columns) if ctx_cols is None else ctx_cols
        shown = self._display(ctx_cols)
        unknown = list(unknown_cols) + [m for m in self.masked_of(v) if m not in ctx_cols]
        rng = np.random.default_rng(stable_seed(self.seed, v, seed_tag))
        result = pd.Series(np.nan, index=frame.index, dtype=object)
        todo = list(frame.index)
        for attempt in range(int(self.c["max_retries_rows"]) + 1):
            if not todo:
                break
            ids = list(todo)
            if self.batching == "cell" and len(self.S) and all(s in frame.columns for s in self.S):
                key = cell_of(frame.loc[ids], self.S)
                ids = list(key.sort_values(kind="stable").index)
            else:
                rng.shuffle(ids)
            per = probe_per_batch or self.bs
            batches = [ids[i:i + per] for i in range(0, len(ids), per)]
            jobs = []
            for b, bid in enumerate(batches):
                ctx = frame.loc[bid, ctx_cols].copy()
                ctx.columns = shown
                for u in unknown:
                    ctx[u] = UNKNOWN
                if neighbours is not None and per < self.bs:
                    nb = neighbours.sample(self.bs - per, random_state=int(rng.integers(1 << 30)))[ctx_cols].copy()
                    nb.columns = shown
                    for u in unknown:
                        nb[u] = UNKNOWN
                    nb.index = [f"nb{k}" for k in range(len(nb))]
                    ctx = pd.concat([ctx, nb]).sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
                if fixed_exemplars is not None:
                    ex = fixed_exemplars
                elif self.exemplars == "matched" and exemplar_pool is None:
                    known = ctx.loc[[i for i in ctx.index if not str(i).startswith("nb")],
                                    [c for c in ctx.columns if c not in unknown]]
                    ex = self.matched_exemplars(known, [tgt], rng).copy()
                    for u in unknown:
                        ex[u] = UNKNOWN
                else:
                    cell = None
                    if self.exemplars == "cell" and all(s in frame.columns for s in self.S):
                        cell = cell_of(frame.loc[bid], self.S).mode().iloc[0]
                    pool = exemplar_pool if exemplar_pool is not None else self._exemplar_pool(cell)
                    ex = pool.sample(min(self.n_ex, len(pool)), random_state=int(rng.integers(1 << 30)))
                    ex_cols = list(dict.fromkeys([c for c in shown + unknown if c in ex.columns] + [tgt]))
                    ex = ex[ex_cols].copy()
                    for u in unknown:
                        if u in ex.columns:
                            ex[u] = UNKNOWN
                jobs.append((ctx, ex, f"{seed_tag}|{v}|b{b}|r{attempt}"))
            with ThreadPoolExecutor(self.workers) as pool_:
                outs = list(pool_.map(lambda j: self._call_batch(v, j[0], j[1], j[2]), jobs))
            for o in outs:
                for idx, val in o.items():
                    if isinstance(idx, str) and idx.startswith("nb"):
                        continue
                    if val is not None:
                        result.at[idx] = val
            todo = [i for i in todo if pd.isna(result.at[i])]
            self.stats["values"] += len(ids)
            self.stats["invalid"] += len(todo)
        return result

    # ------------------------------------------------------------------ joint call (control (ii) off)
    def gen_rows_joint(self, frame: pd.DataFrame, targets, seed_tag, template="rows_joint", structure=""):
        given = [c for c in frame.columns if c not in targets]
        rng = np.random.default_rng(stable_seed(self.seed, "joint", seed_tag))
        out = frame.copy()
        for t in targets:
            out[t] = np.nan
        out = out.astype({t: object for t in targets})
        todo = list(frame.index)
        tdesc = "\n".join(f"- {t}: {self.meta.describe(t)}; {domain_text(self.dom[t])}" for t in targets)
        for attempt in range(int(self.c["max_retries_rows"]) + 1):
            if not todo:
                break
            ids = list(todo)
            rng.shuffle(ids)
            jobs = []
            for b in range(0, len(ids), self.bs):
                bid = ids[b:b + self.bs]
                ex = self.train.sample(self.n_ex, random_state=int(rng.integers(1 << 30)))
                ex = ex[given + targets]
                local = frame.loc[bid, given].reset_index(drop=True)
                prompt = render(template, targets=tdesc, structure=structure, given_columns=", ".join(given),
                                rows_given=rows_text(local), exemplars=rows_text(ex, with_id=False))
                jobs.append((bid, prompt, f"{seed_tag}|joint|b{b}|r{attempt}"))

            def run(job):
                bid, prompt, tag = job
                o = self.llm.call(prompt, temperature=self.c["gen_temperature"], top_p=self.c["gen_top_p"],
                                  seed_tag=tag, stage=self.stage,
                                  hint={"kind": "rows", "ids": list(range(len(bid))),
                                        "domains": {t: self.dom[t] for t in targets}})
                return bid, o.get("rows", []) if isinstance(o, dict) else []

            with ThreadPoolExecutor(self.workers) as pool_:
                for bid, rows in pool_.map(run, jobs):
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        try:
                            j = int(r.get("id"))
                        except (TypeError, ValueError):
                            continue
                        if not 0 <= j < len(bid):
                            continue
                        vals = {t: validate_value(r.get(t), self.dom[t]) for t in targets}
                        if all(x is not None for x in vals.values()):
                            for t, x in vals.items():
                                out.at[bid[j], t] = x
            todo = [i for i in todo if out.loc[i, targets].isna().any()]
            self.stats["values"] += len(ids)
            self.stats["invalid"] += len(todo)
        return out

    # ------------------------------------------------------------------ full table
    def sample_roots(self, m, rng):
        p_hat = self.cell_train.value_counts(normalize=True).to_dict()
        cells = sample_cells(p_hat, m, self.lam, rng)
        roots = pd.DataFrame([c.split("|") for c in cells], columns=self.S)
        for s in self.S:
            roots[s] = cast_like(roots[s], self.train[s])
        return roots

    # ------------------------------------------------------------------ layer groups (control (ii))
    def call_groups(self, nodes):
        """Partition `nodes` into call groups, in generation order.

        grouping="column": one group per node (one call per column and batch).
        grouping="layer" : nodes of the same topological layer of G' share a call, split into the nodes of A
        (those reaching Y without passing R) and the others; surrogates stay alone. By the closure lemma every
        parent of a node in A lies in A u R, so the shared context of an A-group contains no protected attribute
        and Proposition 1 holds unchanged; A and non-A nodes are never mixed.
        """
        if self.grouping != "layer" or self.A is None:
            return [[v] for v in nodes]
        depth = {}
        for v in self.order:
            depth[v] = 1 + max((depth[p] for p in self.G.predecessors(v)), default=-1)
        groups = {}
        for v in nodes:
            key = (depth[v], "hidden:" + v if self.G.nodes[v].get("hidden") else v in self.A)
            groups.setdefault(key, []).append(v)
        return [g for _, g in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))]

    def gen_group(self, group, frame: pd.DataFrame, seed_tag: str) -> pd.DataFrame:
        """One call per batch generating every column of `group` from the union of their contexts."""
        ctx_cols = list(dict.fromkeys(c for v in group for c in self.ctx_cols(v, frame.columns) if c not in group))
        shown = self._display(ctx_cols)
        tdesc = "\n".join(f"- {t}: {self.meta.describe(t)}; {domain_text(self.dom[t])}" for t in group)
        rng = np.random.default_rng(stable_seed(self.seed, "group", "|".join(group), seed_tag))
        out = pd.DataFrame(np.nan, index=frame.index, columns=group, dtype=object)
        todo = list(frame.index)
        for attempt in range(int(self.c["max_retries_rows"]) + 1):
            if not todo:
                break
            ids = list(todo)
            if self.batching == "cell" and all(s in frame.columns for s in self.S):
                ids = list(cell_of(frame.loc[ids], self.S).sort_values(kind="stable").index)
            else:
                rng.shuffle(ids)
            jobs = []
            for b in range(0, len(ids), self.bs):
                bid = ids[b:b + self.bs]
                local = frame.loc[bid, ctx_cols].reset_index(drop=True)
                local.columns = shown
                cell = None
                if self.exemplars == "cell" and all(s in frame.columns for s in self.S):
                    cell = cell_of(frame.loc[bid], self.S).mode().iloc[0]
                if self.exemplars == "matched":
                    ex = self.matched_exemplars(local, group, rng)
                else:
                    pool = self._exemplar_pool(cell)
                    ex = pool.sample(min(self.n_ex, len(pool)), random_state=int(rng.integers(1 << 30)))
                    ex = ex[[c for c in shown if c in ex.columns] + group]
                if not shown:                           # no context: ask for the row ids explicitly
                    prompt = render("gen_group_root", targets=tdesc, ids=json.dumps([str(k) for k in range(len(bid))]),
                                    target_columns=", ".join(group), exemplars=rows_text(ex, with_id=False))
                else:
                    tpl = "gen_group_matched" if self.exemplars == "matched" else "gen_group"
                    prompt = render(tpl, targets=tdesc, parent_columns=", ".join(shown),
                                    rows_parent_only=rows_text(local), target_columns=", ".join(group),
                                    exemplars=rows_text(ex, with_id=False))
                jobs.append((bid, prompt, f"{seed_tag}|grp:{'+'.join(group)}|b{b}|r{attempt}"))

            def run(job):
                bid, prompt, tag = job
                try:
                    o = self.llm.call(prompt, temperature=self.c["gen_temperature"], top_p=self.c["gen_top_p"],
                                      seed_tag=tag, stage=self.stage, max_tokens=self.max_tokens(len(bid), group),
                                      hint={"kind": "group", "ids": list(range(len(bid))),
                                            "domains": {t: self.dom[t] for t in group}})
                except LLMError:                        # unparseable answer: the rows are retried
                    self.stats["failed_calls"] = self.stats.get("failed_calls", 0) + 1
                    o = {}
                return bid, answer_rows(o, len(bid), group)

            with ThreadPoolExecutor(self.workers) as pool_:
                for bid, answer in pool_.map(run, jobs):
                    for j, r in answer.items():
                        if not 0 <= j < len(bid):
                            continue
                        if isinstance(r, dict):           # tolerate {"col": value} answers
                            r = [r.get(t) for t in group]
                        if not isinstance(r, list) or len(r) != len(group):
                            continue
                        vals = {t: validate_value(x, self.dom[t]) for t, x in zip(group, r)}
                        if all(x is not None for x in vals.values()):
                            for t, x in vals.items():
                                out.at[bid[j], t] = x
            todo = [i for i in todo if out.loc[i].isna().any()]
            self.stats["values"] += len(ids) * len(group)
            self.stats["invalid"] += len(todo) * len(group)
        return out

    def complete(self, frame: pd.DataFrame, nodes, seed_tag: str) -> pd.DataFrame:
        """Generate `nodes` (in topological order) for the rows of `frame`; other columns stay fixed."""
        frame = frame.copy()
        need = set(nodes)
        for v in list(nodes):                 # surrogate parents that are not stored in the frame
            need |= {p for p in self.G.predecessors(v) if self.G.nodes[p].get("hidden") and p not in frame.columns}
        nodes = [v for v in self.order if v in need]
        if self.joint_call:
            hidden = [v for v in nodes if self.G.nodes[v].get("hidden")]
            real = [v for v in nodes if v not in hidden]
            frame = self.gen_rows_joint(frame, real, seed_tag)
            return frame
        alive = frame.index
        for group in self.call_groups(nodes):
            for v in group:
                frame[v] = np.nan
                frame[v] = frame[v].astype(object)
            if len(group) == 1:
                frame.loc[alive, group[0]] = self.gen_column(group[0], frame.loc[alive], seed_tag)
            else:
                vals = self.gen_group(group, frame.loc[alive], seed_tag)
                for v in group:
                    frame.loc[alive, v] = vals[v]
            alive = frame.loc[alive].index[frame.loc[alive, group].notna().all(1)]
        return frame

    def generate(self, m: int, seed_tag="gen") -> pd.DataFrame:
        """Exactly m rows. With novelty on: rows that copy a training row or repeat a synthetic row are edited
        minimally (enforce_novelty); rows that cannot be made novel are replaced by top-up rows generated from
        the same cells and accepted only for the same label value, so the (cell, label) counts of the first pass,
        which follow P(Y | pa(Y)) of the generator, are preserved. Quotas are relaxed to the cell, then dropped,
        only if the top-up rounds run out."""
        rng = np.random.default_rng(self.seed)
        nodes = [v for v in self.order if v not in self.S]
        cols = list(self.train.columns)
        if not self.novelty_rounds:
            frame = self.complete(self.sample_roots(m, rng), nodes, seed_tag)
            return self._finish(frame[cols])
        train_keys = set(self._row_keys(self.train))
        log = {"copies": 0, "repeats": 0, "failed": 0, "regenerated": [], "topups": [], "dropped": 0}
        frame = self.complete(self.sample_roots(m, rng), nodes, seed_tag)
        kept, dropped = self.enforce_novelty(frame, nodes, seed_tag, train_keys, set(), log)
        accepted, taken = [kept[cols]], set(self._row_keys(kept))
        quota = {}
        for c, y in zip(cell_of(dropped, self.S), dropped[self.Y]):
            k = (c, None if pd.isna(y) else str(cast_like(pd.Series([y]), self.train[self.Y]).iloc[0]))
            quota[k] = quota.get(k, 0) + 1
        T = int(self.c.get("max_topups", 5))
        for t in range(1, 3 * T + 1):
            need = sum(quota.values())
            if need == 0:
                break
            level = 0 if t <= 2 * T else 1                     # 0: same cell and label, 1: same cell
            log["topups"].append(need)
            cells = [c for (c, _), k in quota.items() for _ in range(k)] * 2
            roots = pd.DataFrame([c.split("|") for c in cells], columns=self.S)
            for s_ in self.S:
                roots[s_] = cast_like(roots[s_], self.train[s_])
            roots.index = roots.index + t * 10 * m              # row ids stay unique across rounds (cache keys)
            tag = f"{seed_tag}|top{t}"
            new, _ = self.enforce_novelty(self.complete(roots, nodes, tag), nodes, tag, train_keys, taken, log)
            new_keys = self._row_keys(new)
            take = []
            for idx, c, y in zip(new.index, cell_of(new, self.S), new[self.Y]):
                if new_keys[idx] in taken:
                    continue
                y = str(cast_like(pd.Series([y]), self.train[self.Y]).iloc[0])
                for k in ([(c, y), (c, None)] if level == 0 else [kk for kk in quota if kk[0] == c]):
                    if quota.get(k, 0) > 0:
                        quota[k] -= 1
                        take.append(idx)
                        taken.add(new_keys[idx])
                        break
            accepted.append(new.loc[take, cols])
            quota = {k: v for k, v in quota.items() if v > 0}
        out = pd.concat(accepted)
        log["unfilled"] = int(sum(quota.values()))
        log["rows"] = len(out)
        self.stats["novelty"] = log
        return self._finish(out)

    def _finish(self, out: pd.DataFrame) -> pd.DataFrame:
        n0 = len(out)
        out = out.dropna().reset_index(drop=True)
        self.stats["dropped_rows"] = n0 - len(out)
        for c in out.columns:
            out[c] = cast_like(out[c], self.train[c])
        return out

    def _row_keys(self, df: pd.DataFrame) -> pd.Series:
        cols = list(self.train.columns)
        x = df[cols].copy()
        for c in cols:
            x[c] = cast_like(x[c], self.train[c])
            if pd.api.types.is_numeric_dtype(self.train[c]):
                x[c] = x[c].round(6)
        return x.astype(str).agg("|".join, axis=1)

    def not_novel(self, frame: pd.DataFrame, train_keys: set, taken=frozenset()) -> list:
        """Failed (incomplete) rows, and complete rows equal to a training row, to an accepted row or to an
        earlier row of the frame."""
        cols = list(self.train.columns)
        ok = frame[cols].notna().all(1)
        key = self._row_keys(frame.loc[ok])
        bad = key.index[key.isin(train_keys) | key.isin(taken) | key.duplicated()]
        return list(frame.index[~ok]) + list(bad)

    def enforce_novelty(self, frame: pd.DataFrame, nodes, seed_tag: str, train_keys: set, taken: set,
                        log: dict):
        """Minimal-edit novelty: round r regenerates, given their (fixed) parents, the last r call groups of the
        rows that are not novel, restricted to columns that are neither Y nor an ancestor of Y in G', so the
        label is never resampled to escape a copy and P(Y | pa(Y)) is untouched; incomplete rows are regenerated
        fully. Returns (novel rows, rows that stay copies or incomplete)."""
        anc = nx.ancestors(self.G, self.Y) | {self.Y} if self.Y in self.G else set()
        groups = [g for g in ([v for v in grp if v not in anc] for grp in self.call_groups(nodes)) if g]
        cols = list(self.train.columns)
        ok = frame[cols].notna().all(1)
        key = self._row_keys(frame.loc[ok])
        log["copies"] += int(key.isin(train_keys).sum())
        log["repeats"] += int(((key.duplicated() | key.isin(taken)) & ~key.isin(train_keys)).sum())
        log["failed"] += int((~ok).sum())
        for r in range(1, self.novelty_rounds + 1):
            bad = self.not_novel(frame, train_keys, taken)
            if not bad:
                break
            log["regenerated"].append(len(bad))
            failed = [i for i in bad if frame.loc[i, cols].isna().any()]
            copied = [i for i in bad if i not in set(failed)]
            jobs = []
            if copied and groups:
                jobs.append((copied, [v for g in groups[-min(r, len(groups)):] for v in g], False))
            if failed:
                jobs.append((failed, nodes, True))
            if not jobs:
                break
            for k, (ids, redo_nodes, full) in enumerate(jobs):
                tag = f"{seed_tag}|novel{r}" + ("" if k == 0 else f"|f{k}")
                redo = self.complete(frame.loc[ids, self.S] if full else frame.loc[ids], redo_nodes, tag)
                new = [c for c in redo.columns
                       if c not in self.S and (c in redo_nodes or self.G.nodes.get(c, {}).get("hidden"))]
                frame.loc[ids, new] = redo[new]
        left = self.not_novel(frame, train_keys, taken)
        log["dropped"] += len(left)
        return frame.drop(index=left), frame.loc[left]
