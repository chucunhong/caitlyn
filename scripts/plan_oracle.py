"""Plan-level comparison without LLM calls: an oracle generator samples ancestrally along a plan with LightGBM
conditionals fitted on real data (an idealized pi_gen), and TSTR utility / fairness are measured on real test.

  python scripts/plan_oracle.py --dataset adult --method caitlyn --seeds 0 1 2
Variants: G (no surgery), current cut (Stage C file), refined joint cut, and all unresolved edges removed.
"""
import argparse
import json

import _common  # noqa: F401
import networkx as nx
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from caitlyn.common import LGB, is_numeric, prep
from caitlyn.config import RESULTS, load_run_config
from caitlyn.data import load_split
from caitlyn.eval.fairness import fairness_all, setting_name
from caitlyn.eval.utility import fit_predict, utility_metrics
from caitlyn.pipeline import json2g
from caitlyn.stage_c_surgery import (apply_surgery, cut_all_unresolved, cut_loss, edge_costs,
                                     refined_joint_surgery)
from caitlyn.stage_d_generate import topo


def oracle_sample(G, train, S, m, seed, lam=0.3, bins=20):
    rng = np.random.default_rng(seed)
    cell = train[S].astype(str).agg("|".join, axis=1)
    p = cell.value_counts(normalize=True)
    pi = (1 - lam) * p + lam / len(p)
    idx = rng.choice(len(pi), size=m, p=(pi / pi.sum()).values)
    out = pd.DataFrame([pi.index[i].split("|") for i in idx], columns=S)
    for s in S:
        out[s] = out[s].astype(train[s].dtype)
    for v in topo(G):
        if v in S:
            continue
        tgt = G.nodes[v].get("surrogate_of", v)
        pa = list(G.predecessors(v))
        cols = [G.nodes[q].get("surrogate_of", q) for q in pa]      # surrogates fitted on the original column
        y = train[tgt]
        num = is_numeric(y)
        lab = pd.qcut(y, bins, labels=False, duplicates="drop") if num else y.astype(str)
        if not pa:
            draw = rng.choice(lab.values, size=m)
        else:
            X = prep(train[cols]).set_axis(pa, axis=1)
            clf = LGBMClassifier(**{**LGB, "n_estimators": 60}, random_state=seed).fit(X, lab)
            Xs = out[pa].copy()
            for c in pa:
                if isinstance(X[c].dtype, pd.CategoricalDtype):
                    Xs[c] = pd.Categorical(Xs[c].astype(str), categories=X[c].cat.categories)
                else:
                    Xs[c] = pd.to_numeric(Xs[c])
            P = clf.predict_proba(Xs)
            u = rng.random(m)[:, None]
            draw = clf.classes_[(P.cumsum(1) < u).sum(1).clip(0, len(clf.classes_) - 1)]
        if num:
            pool = {b: y[lab == b].values for b in pd.unique(lab)}
            out[v] = [rng.choice(pool[b]) for b in draw]
        else:
            out[v] = pd.Series(draw).astype(train[tgt].dtype) if pd.api.types.is_numeric_dtype(train[tgt]) \
                else pd.Series(draw).values
    return out[[c for c in train.columns]]


def evaluate(synth, train, test, meta, seed):
    Y, S = meta.label, meta.protected
    drop = [c for c in meta.eval_drop if c in train.columns]
    s = fit_predict(synth.drop(columns=drop), test.drop(columns=drop), Y, "xgb", seed, train.drop(columns=drop))
    y = test[Y].astype(int).values
    fa = fairness_all(y, s, test, S)
    return dict(AUC=utility_metrics(y, s)["AUC"], SP=np.nanmean([fa[a]["SP"] for a in S]), dSP=fa["_dSP"],
                epsDF=fa[setting_name(S)]["epsDF"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default="caitlyn")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--root", default="results/_archive_v1")
    a = ap.parse_args()
    c = load_run_config()["caitlyn"]
    rows = []
    for seed in a.seeds:
        meta, train, test = load_split(a.dataset, seed)
        S, Y = list(meta.protected), meta.label
        run = json.loads((RESULTS.parent / a.root / a.dataset / a.method / str(seed) / "run.json")
                         .read_text(encoding="utf-8")) if not a.root.startswith("results/_archive") else \
            json.loads((RESULTS / "_archive_v1" / a.dataset / a.method / str(seed) / "run.json").read_text(encoding="utf-8"))
        st = RESULTS / a.dataset / "_stages" / f"seed{seed}"
        A = json.loads((st / f"A_{run['stage_files']['A']}.json").read_text(encoding="utf-8"))
        B = json.loads((st / f"B_{run['stage_files']['B']}.json").read_text(encoding="utf-8"))
        G, R, P = json2g(A["G"]), A["R"], B["proxies"]
        cost, _ = edge_costs(G, train, S, R, Y, P, seed, float(c["eta"]), float(c["gamma"]))
        cache = {}
        from caitlyn.stage_c_surgery import joint_surgery
        C0 = joint_surgery(G, S, R, Y, cost)
        C1, loss1, w1, trace = refined_joint_surgery(G, train, S, R, Y, cost, seed, float(c["eta"]), cache=cache)
        from caitlyn.stage_a_plan import complete_label_parents
        GL, added = complete_label_parents(G, train, S, Y, seed, float(c["alpha"]), int(c["lrt_max_rows"]))
        costL, _ = edge_costs(GL, train, S, R, Y, P, seed, float(c["eta"]), float(c["gamma"]))
        CL = joint_surgery(GL, S, R, Y, costL)
        CLR, _, wLR, _ = refined_joint_surgery(GL, train, S, R, Y, costL, seed, float(c["eta"]), cache={})
        print("  label parents added:", [e["edge"][0] for e in added], "| cut with them:", sorted(CL))
        variants = {"no surgery": (G, set()), "additive cut": (apply_surgery(G, C0, cost, S, R, Y)[0], C0),
                    "label parents + cut": (apply_surgery(GL, CL, costL, S, R, Y)[0], CL),
                    "label parents + refined": (apply_surgery(GL, CLR, wLR, S, R, Y)[0], CLR),
                    "refined cut": (apply_surgery(G, C1, w1, S, R, Y)[0], C1),
                    "remove all unresolved": (apply_surgery(G, cut_all_unresolved(G, S, R, Y), cost, S, R, Y,
                                                            cut_only=True)[0], None)}
        print(f"\n[{a.dataset} {a.method} seed {seed}] additive cut {sorted(C0)} true loss "
              f"{cut_loss(train, G, C0, seed, cache):.4f}\n  refined cut {sorted(C1)} true loss {loss1:.4f} "
              f"({len(trace)} iterations)")
        for name, (G2, C) in variants.items():
            r = evaluate(oracle_sample(G2, train, S, int(c["synth_rows"]), seed), train, test, meta, seed)
            pa = sorted({G2.nodes[q].get("surrogate_of", q) for q in G2.predecessors(Y)})
            rows.append(dict(seed=seed, variant=name, **r, n_cut=len(C) if C is not None else np.nan))
            print(f"  {name:22s} AUC {r['AUC']:.3f} SP {r['SP']:.3f} dSP {r['dSP']:.3f} epsDF {r['epsDF']:.2f} "
                  f"| pa'(Y)={pa}")
        r = evaluate(train.sample(min(len(train), int(c["synth_rows"])), random_state=seed), train, test, meta, seed)
        rows.append(dict(seed=seed, variant="real@1000", **r))
    d = pd.DataFrame(rows)
    print("\nmean over seeds:\n", d.groupby("variant", sort=False).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
