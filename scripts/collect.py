"""Collect metrics.json / preds.csv / run.json files into tidy DataFrames (used by tables and figures)."""
import json
import os

import _common  # noqa: F401
import numpy as np
import pandas as pd

from caitlyn.config import RESULTS, load_dataset_meta, load_run_config
from caitlyn.eval.fairness import METRICS, fairness_all, settings, setting_name


def _runs():
    if not RESULTS.exists():
        return
    for dd in sorted(p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for md in sorted(p for p in dd.iterdir() if p.is_dir() and not p.name.startswith("_")):
            name = md.name
            if name.endswith("__mock"):            # smoke-test outputs: only with CAITLYN_INCLUDE_MOCK=1
                if not os.environ.get("CAITLYN_INCLUDE_MOCK"):
                    continue
                name = name[: -len("__mock")]
            for sd in sorted(p for p in md.iterdir() if p.is_dir() and (p / "metrics.json").exists()):
                yield dd.name, name, sd.name, sd


def collect():
    """Returns dict of DataFrames: utility, distribution (fidelity+privacy), fairness, fairness_extra."""
    cfg = load_run_config()
    pool = set(cfg["evaluation"].get("pool_seeds_for", []))
    U, D, F, X = [], [], [], []
    preds_by = {}
    for ds, method, seed, sd in _runs():
        m = json.loads((sd / "metrics.json").read_text(encoding="utf-8"))
        for mode in ("tstr", "aug"):
            for clf, v in m["utility"].get(mode, {}).items():
                U.append(dict(dataset=ds, method=method, seed=seed, mode=mode, clf=clf, **v))
        if "fidelity" in m:
            D.append(dict(dataset=ds, method=method, seed=seed, **m["fidelity"], **m["privacy"]))
        if ds in pool and (sd / "preds.csv").exists():
            preds_by.setdefault((ds, method), []).append(pd.read_csv(sd / "preds.csv"))
            continue
        for clf, fa in m["fairness"].items():
            _add_fair(F, X, ds, method, seed, clf, fa)
    # pooled seeds (German): intersectional metrics on pooled test predictions
    for (ds, method), frames in preds_by.items():
        meta = load_dataset_meta(ds)
        P = pd.concat(frames, ignore_index=True)
        R = _first_R(ds)
        for clf in cfg["evaluation"]["classifiers"]:
            if f"tstr_{clf}" not in P:
                continue
            fa = fairness_all(P[meta.label].values, P[f"tstr_{clf}"].values, P, meta.protected,
                              R if R and set(R) <= set(P.columns) else None,
                              min_n=cfg["evaluation"]["min_group"], min_pn=cfg["evaluation"]["min_pos_neg"])
            _add_fair(F, X, ds, method, "pooled", clf, fa)
    return dict(utility=pd.DataFrame(U), distribution=pd.DataFrame(D), fairness=pd.DataFrame(F),
                fairness_extra=pd.DataFrame(X))


def _first_R(ds):
    for sd in sorted((RESULTS / ds / "caitlyn").glob("*/run.json")):
        return json.loads(sd.read_text(encoding="utf-8"))["R"]
    return None


def _add_fair(F, X, ds, method, seed, clf, fa):
    meta = load_dataset_meta(ds)
    S = meta.protected
    for attrs in settings(S):
        name = setting_name(attrs)
        for k in METRICS:
            F.append(dict(dataset=ds, method=method, seed=seed, clf=clf, setting=name, kind="S" if len(attrs) == 1 else "C",
                          metric=k, value=fa[name].get(k, np.nan) if fa[name].get(k) is not None else np.nan))
    X.append(dict(dataset=ds, method=method, seed=seed, clf=clf, CSP_R=fa.get("_CSP_R"), dSP=fa.get("_dSP")))


def runs_json():
    rows = []
    if not RESULTS.exists():
        return pd.DataFrame()
    for dd in sorted(p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for p in dd.glob("*/*/run.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            name = p.parent.parent.name
            if name.endswith("__mock"):
                if not os.environ.get("CAITLYN_INCLUDE_MOCK"):
                    continue
                name = name[: -len("__mock")]
            rows.append(dict(dataset=dd.name, method=name, seed=p.parent.name, **{
                k: r.get(k) for k in ("R", "candidates", "proxies", "cut", "ops", "usage", "rows", "rows_requested",
                                      "gen_seconds", "stage_seconds", "gen_stats")}))
    return pd.DataFrame(rows)
