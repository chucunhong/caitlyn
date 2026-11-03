"""Estimate LLM calls, tokens and US$ of the experiment plan before running it.

  python scripts/estimate_cost.py                       # full level (5 seeds), Adult at |D_train|
  python scripts/estimate_cost.py --seeds 3 --adult-rows 10000
Prompts are built with the real templates and real rows (tokens counted with tiktoken cl100k_base as a
proxy for the Qwen tokenizer; without tiktoken, 3.6 characters per token). Plans are approximated by giving
each column its 3 most associated earlier columns as parents. The LLM cache is ignored, so this is an
upper bound for variants that share calls.
"""
import argparse
import json
import math

import _common  # noqa: F401
import numpy as np
import pandas as pd

from caitlyn.assoc import AssocEncoder
from caitlyn.config import apply_overrides, load_run_config
from caitlyn.data import column_domains, domain_text, fmt_value, load_split, rows_text
from caitlyn.prompts import render

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    ntok = lambda s: len(_enc.encode(s))
except ImportError:
    ntok = lambda s: int(len(s) / 3.6)

RETRY = 1.05      # ~5% of rows regenerated after a validation failure


def approx_plan(train, S, Y, k=3):
    enc = AssocEncoder(train)
    order = list(S) + [c for c in train.columns if c not in S and c != Y] + [Y]
    pa = {}
    for i, v in enumerate(order):
        if v in S:
            continue
        prev = order[:i]
        pa[v] = sorted(prev, key=lambda u: -enc.score(u, v))[:k]
    return order, pa


def gen_call_tokens(meta, train, v, parents, rng, bs=20, n_ex=10):
    dom = column_domains(train)
    rows = train.sample(bs, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    ex = train.sample(n_ex, random_state=int(rng.integers(1 << 30)))
    p = render("gen", target=v, target_description=meta.describe(v), domain=domain_text(dom[v]),
               parent_columns=", ".join(parents), rows_parent_only=rows_text(rows[parents]),
               exemplars=rows_text(ex[parents + [v]], with_id=False))
    out = json.dumps({"values": {str(i): fmt_value(x) for i, x in enumerate(rows[v])}}, default=str)
    return ntok(p), ntok(out)


def group_call_tokens(meta, train, group, parents, rng, bs=50, n_ex=10):
    dom = column_domains(train)
    rows = train.sample(bs, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    ex = train.sample(n_ex, random_state=int(rng.integers(1 << 30)))
    tdesc = "\n".join(f"- {t}: {meta.describe(t)}; {domain_text(dom[t])}" for t in group)
    p = render("gen_group", targets=tdesc, parent_columns=", ".join(parents) or "none",
               rows_parent_only=rows_text(rows[parents]), target_columns=", ".join(group),
               exemplars=rows_text(ex[parents + group], with_id=False))
    out = json.dumps({"values": {str(i): [fmt_value(r[t]) for t in group] for i, r in rows.iterrows()}}, default=str)
    return ntok(p), ntok(out)


def rows_call_tokens(meta, train, template, rng, n=20, n_ex=10):
    dom = column_domains(train)
    cols = list(train.columns)
    tdesc = "\n".join(f"- {t}: {meta.describe(t)}; {domain_text(dom[t])}" for t in cols)
    ex = train.sample(n_ex, random_state=int(rng.integers(1 << 30)))
    p = render(template, structure="\n".join(f"{a} -> {b}" for a, b in zip(cols, cols[1:])), targets=tdesc,
               n=n, ids=json.dumps(list(range(n))), task_description=meta.task, exemplars=rows_text(ex, with_id=False))
    out_rows = train.sample(n, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    out = json.dumps({"rows": [{"id": i, **{c: fmt_value(r[c]) for c in cols}} for i, r in out_rows.iterrows()]},
                     default=str)
    return ntok(p), ntok(out)


class Profile:
    """Per-dataset token profile: average tokens of each call type."""

    def __init__(self, ds, cfg):
        self.meta, self.train, _ = load_split(ds, 0)
        S, Y = self.meta.protected, self.meta.label
        rng = np.random.default_rng(0)
        self.order, self.pa = approx_plan(self.train, S, Y)
        self.cols = [c for c in self.order if c not in S]
        bs = int(cfg["caitlyn"]["batch_size"])
        g = [gen_call_tokens(self.meta, self.train, v, self.pa[v] or [S[0]], rng, bs) for v in self.cols for _ in range(2)]
        self.gen = tuple(np.mean(g, 0))                   # (in, out) per parent-only call, one column
        # layer groups of the approximate plan: (depth, reaches Y) -> one call per group and batch
        depth, anc = {}, {Y}
        for v in self.order:
            depth[v] = 0 if v in S else 1 + max(depth[q] for q in self.pa[v])
        for v in reversed(self.order):
            if any(v in self.pa.get(w, []) for w in anc):
                anc.add(v)
        groups = {}
        for v in self.cols:
            groups.setdefault((depth[v], v in anc), []).append(v)
        self.groups = list(groups.values())
        gt = [group_call_tokens(self.meta, self.train, gr, sorted({q for v in gr for q in self.pa[v]} - set(gr)), rng, bs)
              for gr in self.groups]
        self.gen_layer = tuple(np.sum(gt, 0))             # (in, out) summed over the groups of one batch
        full = []
        for v in self.cols:                               # full-history context: every earlier column
            prev = self.order[:self.order.index(v)]
            full.append(gen_call_tokens(self.meta, self.train, v, prev, rng, bs))
        self.gen_full = tuple(np.mean(full, 0))
        self.rows_ss = rows_call_tokens(self.meta, self.train, "structsynth_gen", rng)
        self.rows_cllm = rows_call_tokens(self.meta, self.train, "cllm_gen", rng)
        sample = rows_text(self.train.sample(cfg["caitlyn"]["n_example_rows"], random_state=0), with_id=False)
        desc = "\n".join(f"- {c}: {self.meta.describe(c)}" for c in self.train.columns)
        self.role = (ntok(render("role", task_description=self.meta.task, protected=", ".join(S), label=Y,
                                 column_descriptions=desc, rows=sample)), 400)
        graph = "\n".join(f"{u} -> {v}: rationale of about fifteen words grounded in the association score"
                          for v in self.cols[:len(self.cols)] for u in self.pa[v][:2])
        self.edge = (ntok(render("edge", u="x", role_u="other", graph=graph, roles=json.dumps({c: "other" for c in self.train.columns}),
                                 assoc=json.dumps({c: 0.123 for c in self.train.columns}), rows=sample)), 250)
        self.n_train = len(self.train)


def stage_cost(p: Profile, m, cfg, context="parent", audited=4, children=2):
    """{stage: (calls, tok_in, tok_out)} for one full CAITLYN run (Stages A, B, D)."""
    c = cfg["caitlyn"]
    batches = math.ceil(m / c["batch_size"]) * RETRY
    if context == "parent" and c.get("grouping", "column") == "layer":
        calls_d = batches * len(p.groups)
        din, dout = batches * p.gen_layer[0], batches * p.gen_layer[1]
    else:
        gin, gout = p.gen if context == "parent" else p.gen_full
        calls_d = batches * len(p.cols)
        din, dout = calls_d * gin, calls_d * gout
    calls_b = audited * children * 3 * c["M"] * c["J"] / c["batch_size"]
    calls_a_role, calls_a_edge = c["K"], len(p.train.columns) + 2
    return {"A": (calls_a_role + calls_a_edge, calls_a_role * p.role[0] + calls_a_edge * p.edge[0],
                  calls_a_role * p.role[1] + calls_a_edge * p.edge[1]),
            "B": (calls_b, calls_b * p.gen[0], calls_b * p.gen[1]),
            "D": (calls_d, din, dout)}


def add(acc, key, st, mult=1.0, price=None):
    a = acc.setdefault(key, {"calls": 0.0, "in": 0.0, "out": 0.0, "usd": 0.0})
    for _, (n, ti, to) in st.items():
        a["calls"] += n * mult
        a["in"] += ti * mult
        a["out"] += to * mult
        a["usd"] += mult * (ti * price[0] + to * price[1]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="default: len(seeds) in run.yaml")
    ap.add_argument("--synth-rows", type=int, default=None, help="|D_synth| (default caitlyn.synth_rows, else |D_train|)")
    ap.add_argument("--adult-rows", type=int, default=None, help="override |D_synth| on Adult only")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override configs/run.yaml")
    ap.add_argument("--open-price", type=float, nargs=2, default=[1.40, 5.60],
                    help="US$/1M in/out for the open-weight backbone (verify on the pricing page)")
    a = ap.parse_args()
    cfg = apply_overrides(load_run_config(), a.set)
    price = (cfg["llm"]["price_in_per_m"], cfg["llm"]["price_out_per_m"])
    P = {ds: Profile(ds, cfg) for ds in ["adult", "credit_scoring", "german"]}
    rows = a.synth_rows or cfg["caitlyn"].get("synth_rows")
    m = {ds: (a.adult_rows if ds == "adult" and a.adult_rows else rows or P[ds].n_train) for ds in P}
    n = a.seeds or len(cfg["seeds"])
    acc = {}
    print("Average tokens per call (in / out):")
    for ds, p in P.items():
        print(f"  {ds:16s} per-column call {p.gen[0]:6.0f}/{p.gen[1]:4.0f}   {len(p.groups)} layer groups "
              f"vs {len(p.cols)} columns   m={m[ds]}")

    real = ["adult", "credit_scoring", "german"]
    # 1. main CAITLYN runs
    for ds in real:
        add(acc, "1 CAITLYN main (3 datasets)", stage_cost(P[ds], m[ds], cfg), n, price)
    # 2. LLM baselines: StructSynth, CLLM (1.3x oversampling)
    for ds in real:
        p = P[ds]
        ss = math.ceil(m[ds] / 20) * RETRY
        add(acc, "2 StructSynth + CLLM", {"ss": (ss, ss * p.rows_ss[0], ss * p.rows_ss[1]),
                                          "plan": (len(p.cols), len(p.cols) * p.edge[0], len(p.cols) * p.edge[1])}, n, price)
        cl = math.ceil(1.3 * m[ds] / 20) * RETRY
        add(acc, "2 StructSynth + CLLM", {"cl": (cl, cl * p.rows_cllm[0], cl * p.rows_cllm[1])}, n, price)
    # 3. ablations (Table 6): 8 variants on Adult and Credit scoring
    for ds in ["adult", "credit_scoring"]:
        for v in ["nogate", "pcplan", "noproxy", "perattr", "rllm", "rempty", "lam0"]:
            add(acc, "3 Ablations (8 variants x 2 datasets)", stage_cost(P[ds], m[ds], cfg), n, price)
        add(acc, "3 Ablations (8 variants x 2 datasets)", stage_cost(P[ds], m[ds], cfg, context="full"), n, price)
    # 4. seesaw (Table 5): none + 2 orders
    for ds in ["adult", "credit_scoring"]:
        add(acc, "4 Seesaw (3 variants x 2 datasets)", stage_cost(P[ds], m[ds], cfg), 3 * n, price)
    # 5. sensitivity on Adult: n in {100,500,2000}; lambda 0.6, 1; k 1, 2; delta x2; tau_p x2
    for nn in (100, 500, 2000):                          # training size n; |D_synth| stays fixed when set
        add(acc, "5 Sensitivity (Adult)", stage_cost(P["adult"], rows or nn, cfg), n, price)
    add(acc, "5 Sensitivity (Adult)", stage_cost(P["adult"], m["adult"], cfg), 8 * n, price)
    # 6. open-weight backbone: main comparison again
    for ds in real:
        add(acc, "6 Open-weight backbone (3 datasets)", stage_cost(P[ds], m[ds], cfg), n, tuple(a.open_price))
    # 7. leakage (Fig. 2b) on Adult and Credit scoring: 3 extra generator variants + eps-hat probe + twins
    nA = 6                                                # probed nodes of A (approximate)
    for ds in ["adult", "credit_scoring"]:
        p = P[ds]
        gens = stage_cost(p, m[ds], cfg)
        full = stage_cost(p, m[ds], cfg, context="full")
        probe = nA * 3 * 4 * 2 * (cfg["leakage"]["M"] / cfg["leakage"]["probe_per_batch"])
        twins = 4 * 3 * math.ceil(cfg["leakage"]["twins_rows"] / cfg["caitlyn"]["batch_size"]) * len(p.groups)
        add(acc, "7 Leakage (Adult, Credit scoring)", {"D": gens["D"]}, 2 * n, price)
        add(acc, "7 Leakage (Adult, Credit scoring)", {"D": full["D"], "probe": (probe, probe * p.gen_full[0], probe * p.gen[1]),
                                                        "twins": (twins, twins * p.gen[0], twins * p.gen[1])}, n, price)

    print(f"\nExperiment plan: {n} seeds, |D_synth| = {m['adult']}, batch {cfg['caitlyn']['batch_size']}, grouping {cfg['caitlyn'].get('grouping')}, prices {price} US$/1M (open {a.open_price})")
    print(f"{'Block':42s} {'calls':>10s} {'tokens in':>12s} {'tokens out':>11s} {'US$':>9s}")
    tot = {"calls": 0, "in": 0, "out": 0, "usd": 0}
    for k in sorted(acc):
        v = acc[k]
        for kk in tot:
            tot[kk] += v[kk]
        print(f"{k:42s} {v['calls']:10,.0f} {v['in'] / 1e6:10.1f}M {v['out'] / 1e6:9.1f}M {v['usd']:9.0f}")
    print(f"{'TOTAL':42s} {tot['calls']:10,.0f} {tot['in'] / 1e6:10.1f}M {tot['out'] / 1e6:9.1f}M {tot['usd']:9.0f}")


if __name__ == "__main__":
    main()
