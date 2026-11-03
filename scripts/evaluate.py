"""Evaluate every results/<dataset>/<method>/<seed>/synth.csv (utility, fidelity, privacy, fairness).

  python scripts/evaluate.py                         # all datasets / methods / seeds found
  python scripts/evaluate.py --datasets adult --methods caitlyn ctgan --bootstrap
Writes metrics.json and preds.csv next to each synth.csv (skips if up to date unless --force).
"""
import argparse
import json
import traceback

import _common  # noqa: F401
import numpy as np
import pandas as pd

from caitlyn.common import to_jsonable
from caitlyn.config import RESULTS, load_run_config
from caitlyn.data import load_split
from caitlyn.eval.fairness import METRICS, bootstrap_ci, fairness_all, settings, setting_name, summarize_SC
from caitlyn.eval.fidelity import fidelity
from caitlyn.eval.privacy import privacy
from caitlyn.eval.utility import fit_predict, utility_metrics

NO_DISTRIBUTION = {"real", "reweigh"}   # fidelity / privacy / augmentation not meaningful


def caitlyn_R(dataset, seed):
    p = RESULTS / dataset / "caitlyn" / str(seed) / "run.json"
    return json.loads(p.read_text(encoding="utf-8"))["R"] if p.exists() else None


def clean_synth(synth, train):
    synth = synth[[c for c in train.columns]].dropna()
    for c in train.columns:
        if pd.api.types.is_numeric_dtype(train[c]):
            synth[c] = pd.to_numeric(synth[c], errors="coerce")
    synth = synth.dropna()
    return synth.reset_index(drop=True)


def evaluate_run(dataset, method, seed, cfg, bootstrap=False):
    d = RESULTS / dataset / method / str(seed)
    meta, train, test = load_split(dataset, seed)
    S, Y, ev = meta.protected, meta.label, cfg["evaluation"]
    synth = clean_synth(pd.read_csv(d / "synth.csv"), train)
    synth[Y] = synth[Y].round().astype(int)
    base = method.split("__")[0]
    weights = pd.read_csv(d / "weights.csv")["weight"].values if (d / "weights.csv").exists() else None
    R = caitlyn_R(dataset, seed)
    y = test[Y].astype(int).values
    res = {"n_synth": len(synth), "R_for_CSP": R, "utility": {"tstr": {}, "aug": {}}, "fairness": {},
           "fairness_SC": {}}
    preds = test.copy()                                  # all columns (CSP_R needs R when pooling seeds)
    drop = [c for c in meta.eval_drop if c in train.columns]   # leakage columns: not used by classifiers
    tr_f, te_f, sy_f = (x.drop(columns=drop) for x in (train, test, synth))
    res["eval_drop"] = drop
    for clf in ev["classifiers"]:
        score = fit_predict(sy_f, te_f, Y, clf, seed, tr_f, weights=weights)
        preds[f"tstr_{clf}"] = score
        res["utility"]["tstr"][clf] = utility_metrics(y, score)
        fa = fairness_all(y, score, test, S, R, min_n=ev["min_group"], min_pn=ev["min_pos_neg"])
        res["fairness"][clf] = fa
        res["fairness_SC"][clf] = summarize_SC(fa, S)
        if base not in NO_DISTRIBUTION:
            aug = pd.concat([tr_f, sy_f], ignore_index=True)
            s_aug = fit_predict(aug, te_f, Y, clf, seed, tr_f)
            preds[f"aug_{clf}"] = s_aug
            res["utility"]["aug"][clf] = utility_metrics(y, s_aug)
    if base not in NO_DISTRIBUTION:
        res["fidelity"] = fidelity(train, synth, ev["fidelity_max_rows"], seed)
        res["privacy"] = privacy(train, test, synth, ev["privacy_max_rows"], seed)
        res["synthetic_label_SP"] = data_level_sp(synth, S, Y)
    if bootstrap:
        ci = {}
        for attrs in settings(S):
            for m in METRICS:
                ci[f"{setting_name(attrs)}|{m}"] = bootstrap_ci(y, preds["tstr_xgb"].values, test, attrs, m,
                                                                B=ev["bootstrap"], seed=seed, min_n=ev["min_group"],
                                                                min_pn=ev["min_pos_neg"])
        res["bootstrap_ci_xgb"] = ci
    (d / "metrics.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    preds.to_csv(d / "preds.csv", index=False)
    return res


def data_level_sp(df, S, Y):
    """Fairness of the synthetic labels themselves (supplement, item iii)."""
    out = {}
    for attrs in settings(S):
        rate = df.groupby(df[attrs].astype(str).agg("|".join, axis=1))[Y].mean()
        out[setting_name(attrs)] = float(rate.max() - rate.min()) if len(rate) > 1 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--methods", nargs="*", help="method directory names (prefix match)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    a = ap.parse_args()
    cfg = load_run_config()
    for dd in sorted(p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if a.datasets and dd.name not in a.datasets:
            continue
        for md in sorted(p for p in dd.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if a.methods and not any(md.name.startswith(m) for m in a.methods):
                continue
            for sd in sorted(p for p in md.iterdir() if p.is_dir() and (p / "synth.csv").exists()):
                mj = sd / "metrics.json"
                if mj.exists() and not a.force and mj.stat().st_mtime > (sd / "synth.csv").stat().st_mtime:
                    continue
                print(f"evaluating {dd.name}/{md.name}/{sd.name}")
                try:
                    evaluate_run(dd.name, md.name, int(sd.name), cfg, a.bootstrap)
                except Exception:
                    traceback.print_exc()


if __name__ == "__main__":
    main()
