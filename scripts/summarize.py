"""Quick TSTR summary (SENNA-style): utility and fairness per dataset, classifier and method.

  python scripts/summarize.py --datasets german credit_scoring
Means over seeds; fairness measured on the real test set. Single-attribute metrics for sex and age,
intersectional SP (dSP_cap) and eps-DF over sex x age, Theil index for individual fairness.
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from caitlyn.config import RESULTS, load_dataset_meta
from caitlyn.eval.fairness import setting_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["german", "credit_scoring", "adult"])
    ap.add_argument("--clf", nargs="*", default=None)
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        meta = load_dataset_meta(ds)
        S = meta.protected
        for p in sorted((RESULTS / ds).glob("*/*/metrics.json")):
            method, seed = p.parent.parent.name, p.parent.name
            m = json.loads(p.read_text(encoding="utf-8"))
            for clf, u in m["utility"]["tstr"].items():
                if a.clf and clf not in a.clf:
                    continue
                fa = m["fairness"][clf]
                row = dict(dataset=ds, method=method, seed=seed, clf=clf, AUC=u["AUC"], BA=u["BA"], F1=u["F1"])
                for s in S:
                    for k in ("SP", "EOd", "PP", "ABROCA"):
                        row[f"{k}_{s}"] = fa[s][k]
                row["dSP_cap"] = fa["_dSP"]
                row["epsDF_cap"] = fa[setting_name(S)]["epsDF"]
                row["Theil"] = fa.get("_Theil")
                rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        print("no metrics.json found")
        return
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for ds, g in d.groupby("dataset"):
        t = g.groupby(["clf", "method"]).mean(numeric_only=True).round(3)
        print(f"\n=== {ds}: TSTR, mean over {g.seed.nunique()} seeds (fairness on the real test set) ===")
        print(t.to_string())


if __name__ == "__main__":
    main()
