"""Leakage analysis (paper Sec. 4.4, Fig. 2b): eps_hat_v per Stage D condition and twin flip rates of
the corresponding generator variants.

  # 1) generate the variant tables first (each is a normal CAITLYN run), for adult and credit_scoring:
  python scripts/run_caitlyn.py --dataset adult --variant leak_full
  python scripts/run_caitlyn.py --dataset adult --variant leak_parent
  python scripts/run_caitlyn.py --dataset adult --variant leak_indep
  python scripts/run_caitlyn.py --dataset adult
  # 2) probe and twins:
  python scripts/run_leakage.py --dataset adult
Writes results/_leakage/<dataset>_seed<seed>.json.
"""
import argparse
import json

import _common  # noqa: F401
import pandas as pd

from caitlyn.common import to_jsonable
from caitlyn.config import RESULTS, apply_overrides, load_run_config
from caitlyn.data import load_split
from caitlyn.eval.leakage import CONDITIONS, leakage_report, nodes_A
from caitlyn.eval.twins import twin_flip_rate
from caitlyn.llm import make_llm
from caitlyn.pipeline import Pipeline

# leakage condition -> generator variant whose synthetic table is used for the twins
VARIANT_OF = {"full_history": "leak_full", "parent_only": "leak_parent", "parent_indep": "leak_indep",
              "caitlyn": "full"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--no-twins", action="store_true")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override configs/run.yaml, e.g. caitlyn.B=50")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    cfg = apply_overrides(load_run_config(), a.set)
    lc = cfg["leakage"]
    llm = make_llm(cfg, mock=a.mock)
    out_dir = RESULTS / "_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in (a.seeds if a.seeds is not None else cfg["seeds"]):
        meta, train, test = load_split(a.dataset, seed)
        S, Y = meta.protected, meta.label
        pipe = Pipeline(llm, meta, train, cfg, seed, "full")
        A = pipe.stage_a()
        C = pipe.stage_c(A, pipe.stage_b(A))
        G2 = C["G2"]
        A_nodes = sorted(v for v in nodes_A(G2, A["R"], Y)
                         if v not in S and v not in A["R"] and not G2.nodes[v].get("hidden"))   # surrogates are not output
        gen = pipe.generator(G2, stage="leakage")
        rep, pairs = leakage_report(gen, A_nodes, train, S, Y, seed, int(lc["M"]), int(lc["probe_per_batch"]),
                                    int(lc["n_pairs"]))
        res = {"dataset": a.dataset, "seed": seed, "A": A_nodes, "pairs": pairs, "eps": rep, "twins": {}}
        if not a.no_twins:
            for cond in CONDITIONS:
                v = VARIANT_OF[cond]
                vp = Pipeline(llm, meta, train, cfg, seed, v)
                path = RESULTS / a.dataset / vp.method / str(seed) / "synth.csv"
                if not path.exists():
                    print(f"  missing {path}; run run_caitlyn.py --variant {v} first")
                    continue
                synth = pd.read_csv(path)
                vgen = vp.generator(G2, stage="twins")
                res["twins"][cond] = twin_flip_rate(vgen, synth, train, test, S, A["R"], Y, seed,
                                                    int(lc["twins_rows"]), int(lc["twins_alt_cells"]))
        res["usage"] = llm.usage_dict()
        (out_dir / f"{a.dataset}_seed{seed}.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
        print(json.dumps({c: rep[c]["sum"] for c in rep}), json.dumps(res["twins"]))


if __name__ == "__main__":
    main()
