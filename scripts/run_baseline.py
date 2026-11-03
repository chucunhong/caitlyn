"""Run a baseline, or bridge an external one.

  python scripts/run_baseline.py --dataset adult --method ctgan --seeds 0 1 2
  python scripts/run_baseline.py --dataset adult --method structsynth
  python scripts/run_baseline.py --dataset adult --method export              # files for external repos
  python scripts/run_baseline.py --dataset adult --method import --name tabsyn --seed 0 --file path/to/synth.csv

In-repo methods: real, reweigh, ctgan, decaf (needs CAITLYN's plan for the same seed), great,
tabfairgan (needs the cloned repo), structsynth, cllm.
External (export/import): tabsyn, fairtabddpm, senna, golob, and the official CLLM if preferred.
"""
import argparse

import _common  # noqa: F401

from caitlyn.config import RESULTS, apply_overrides, load_run_config
from caitlyn.data import load_split
from caitlyn.llm import make_llm

INTERNAL = ["real", "reweigh", "ctgan", "decaf", "great", "tabfairgan", "structsynth", "cllm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True, choices=INTERNAL + ["export", "import"])
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--name", help="method name for --method import")
    ap.add_argument("--seed", type=int, help="seed for --method import")
    ap.add_argument("--file", help="synthetic CSV for --method import")
    ap.add_argument("--model", help="LLM backbone override for LLM baselines")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override configs/run.yaml, e.g. caitlyn.B=50")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    cfg = apply_overrides(load_run_config(), a.set)

    if a.method == "import":
        from baselines.external import import_synth
        meta, train, _ = load_split(a.dataset, a.seed)
        print("imported ->", import_synth(meta, train, a.name, a.seed, a.file))
        return

    for seed in (a.seeds if a.seeds is not None else cfg["seeds"]):
        meta, train, test = load_split(a.dataset, seed)
        if a.method == "export":
            from baselines.external import export
            print("exported ->", export(meta, train, test, seed))
            continue
        out = RESULTS / a.dataset / (a.method + ("__mock" if a.mock else "")) / str(seed)
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{a.dataset} | {a.method} | seed {seed}] -> {out}")
        if a.method == "real":
            train.to_csv(out / "synth.csv", index=False)
        elif a.method == "reweigh":
            from baselines import reweighing
            reweighing.run(meta, train, seed, cfg, out)
        elif a.method in ("ctgan", "decaf", "great"):
            from baselines import dgm
            getattr(dgm, f"run_{a.method}")(meta, train, seed, cfg, out)
        elif a.method == "tabfairgan":
            from baselines import tabfairgan
            tabfairgan.run(meta, train, seed, cfg, out)
        else:
            from baselines import llm_baselines
            llm = make_llm(cfg, model=a.model, mock=a.mock)
            getattr(llm_baselines, f"run_{a.method}")(meta, train, seed, cfg, out, llm=llm)
            import json
            (out / "run.json").write_text(json.dumps({"method": a.method, "usage": llm.usage_dict(),
                                                      "rows_requested": int(cfg["caitlyn"].get("synth_rows") or len(train))},
                                                     indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
