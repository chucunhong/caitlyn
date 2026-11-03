"""Run CAITLYN (or one of its variants) on a dataset and seed(s).

  python scripts/run_caitlyn.py --dataset adult --seeds 0 1 2
  python scripts/run_caitlyn.py --dataset adult --variant nogate              # ablation
  python scripts/run_caitlyn.py --dataset adult --variant perattr --order sex age race   # seesaw
  python scripts/run_caitlyn.py --dataset adult --lam 0.6                     # sensitivity
  python scripts/run_caitlyn.py --dataset adult --model open                  # open-weight backbone
  python scripts/run_caitlyn.py --dataset german --stages ABC                 # stop before generation
  python scripts/run_caitlyn.py --dataset german --mock --m 200               # offline smoke test

Outputs: results/<dataset>/<method>/<seed>/{synth.csv, run.json, plan_edited.json}
"""
import argparse
import json

import _common  # noqa: F401

from caitlyn.config import RESULTS, apply_overrides, load_run_config
from caitlyn.data import load_split
from caitlyn.llm import make_llm
from caitlyn.pipeline import VARIANTS, Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--order", nargs="*", help="attribute order for --variant perattr")
    ap.add_argument("--stages", default="ABCD")
    ap.add_argument("--m", type=int, help="number of synthetic rows (default |D_train| or caitlyn.synth_rows)")
    ap.add_argument("--n-train", type=int, help="subsample the training split (sensitivity in n)")
    ap.add_argument("--lam", type=float)
    ap.add_argument("--k", type=int, help="protect only the first k protected attributes")
    ap.add_argument("--delta", type=float)
    ap.add_argument("--tau-p", type=float)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--model", help="'open' for the open-weight backbone, or any DashScope model id")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override configs/run.yaml, e.g. caitlyn.B=50")
    ap.add_argument("--mock", action="store_true", help="offline fake LLM (smoke test only)")
    a = ap.parse_args()

    cfg = apply_overrides(load_run_config(), a.set)
    llm = make_llm(cfg, model=a.model, mock=a.mock)
    for seed in (a.seeds if a.seeds is not None else cfg["seeds"]):
        meta, train, test = load_split(a.dataset, seed, a.n_train)
        pipe = Pipeline(llm, meta, train, cfg, seed, a.variant, order=a.order, lam=a.lam, k=a.k, delta=a.delta,
                        tau_p=a.tau_p, gamma=a.gamma, n_train=a.n_train, model=a.model)
        method = pipe.method + ("__mock" if a.mock else "")
        out = RESULTS / a.dataset / method / str(seed)
        print(f"[{a.dataset} | {method} | seed {seed}] -> {out}")
        summary, A, B, C = pipe.run(out, m=a.m, stages=a.stages)
        print(f"  R = {A['R']}  (candidates {A['candidates']})")
        print(f"  prior-driven edges rejected: {[e['edge'] for e in A['prior_driven']]}")
        if B:
            print(f"  proxies = {B['proxies']}")
        if C:
            print(f"  cut = {C['cut']}  ops = {C['ops']}")
        if "rows" in summary:
            print(f"  generated {summary['rows']}/{summary['rows_requested']} rows; stats {summary['gen_stats']}")
        print("  usage:", json.dumps({k: {kk: round(vv) for kk, vv in v.items()} for k, v in summary["usage"].items()}))


if __name__ == "__main__":
    main()
