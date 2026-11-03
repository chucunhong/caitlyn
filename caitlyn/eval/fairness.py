"""Group fairness over every setting (single attributes and combinations), following
Le Quy et al. / SENNA: SP, EO, EOd, PP, PE, TE, ABROCA, eps-DF; plus CSP_R (supplement Sec. D)."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

_trapz = getattr(np, "trapezoid", None) or np.trapz
METRICS = ["SP", "EO", "EOd", "PP", "PE", "TE", "ABROCA", "epsDF"]


def settings(S):
    return [list(c) for r in range(1, len(S) + 1) for c in combinations(S, r)]


def setting_name(attrs):
    return "*".join(attrs)


def group_rates(y, yhat, mask):
    yy, pp = y[mask], yhat[mask]
    tp = int(((pp == 1) & (yy == 1)).sum()); fp = int(((pp == 1) & (yy == 0)).sum())
    fn = int(((pp == 0) & (yy == 1)).sum()); tn = int(((pp == 0) & (yy == 0)).sum())
    return dict(PR=pp.mean(), TPR=tp / max(tp + fn, 1), FPR=fp / max(fp + tn, 1),
                PPV=tp / max(tp + fp, 1) if tp + fp > 0 else np.nan, TE=(fn / fp if fp > 0 else np.nan),
                npos=tp + fn, nneg=fp + tn)


def abroca(y, score, m1, m2, grid=np.linspace(0, 1, 1001)):
    f1, t1, _ = roc_curve(y[m1], score[m1])
    f2, t2, _ = roc_curve(y[m2], score[m2])
    return float(_trapz(np.abs(np.interp(grid, f1, t1) - np.interp(grid, f2, t2)), grid))


def eps_df(yhat, groups, alpha=1.0):
    eps = 0.0
    for v in (0, 1):
        p = [((yhat[m] == v).sum() + alpha) / (m.sum() + 2 * alpha) for m in groups]
        eps = max(eps, float(np.log(max(p)) - np.log(min(p))))
    return eps


def _gap(key, rs):
    vals = [r[key] for r in rs if r[key] == r[key]]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else np.nan


def fairness_setting(y, yhat, score, df, attrs, min_n=30, min_pn=5):
    y, yhat, score = np.asarray(y), np.asarray(yhat), np.asarray(score)
    key = df[attrs].astype(str).agg("|".join, axis=1).values
    uniq = pd.unique(key)
    groups = [key == k for k in uniq if (key == k).sum() >= min_n]
    excluded = [k for k in uniq if (key == k).sum() < min_n]
    out = dict(n_groups=len(groups), excluded=list(map(str, excluded)))
    if len(groups) < 2:
        return {**out, **{m: np.nan for m in METRICS}}
    R = [group_rates(y, yhat, m) for m in groups]
    ok = [r for r in R if r["npos"] >= min_pn and r["nneg"] >= min_pn]
    te_ok = [r for r in ok if r["TE"] == r["TE"]]
    out.update(SP=_gap("PR", R), EO=_gap("TPR", ok), PE=_gap("FPR", ok), PP=_gap("PPV", R), TE=_gap("TE", te_ok),
               n_te_dropped=len(ok) - len(te_ok))
    out["EOd"] = out["EO"] + out["PE"] if out["EO"] == out["EO"] else np.nan
    aucg = [(roc_auc_score(y[m], score[m]), m) for m in groups if len(np.unique(y[m])) == 2]
    if len(aucg) >= 2:
        hi = max(aucg, key=lambda t: t[0])[1]
        lo = min(aucg, key=lambda t: t[0])[1]
        out["ABROCA"] = abroca(y, score, hi, lo)
    else:
        out["ABROCA"] = np.nan
    out["epsDF"] = eps_df(yhat, groups)
    return out


def csp_r(y, yhat, df, S, R, min_stratum=30, min_group=10):
    """P(r)-weighted SP over the full intersection within strata of R (quintiles for numerical R)."""
    yhat = np.asarray(yhat)
    cell = df[S].astype(str).agg("|".join, axis=1).values
    if not R:
        strata = np.zeros(len(df), dtype=object)
    else:
        parts = []
        for r in R:
            col = df[r]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > 15:
                parts.append(pd.qcut(col, 5, labels=False, duplicates="drop").astype(str))
            else:
                parts.append(col.astype(str))
        strata = pd.concat(parts, axis=1).agg("|".join, axis=1).values
    tot, acc = 0, 0.0
    for st in pd.unique(strata):
        m = strata == st
        if m.sum() < min_stratum:
            continue
        rates = [yhat[m & (cell == c)].mean() for c in pd.unique(cell[m]) if (m & (cell == c)).sum() >= min_group]
        if len(rates) < 2:
            continue
        acc += m.sum() * (max(rates) - min(rates))
        tot += m.sum()
    return float(acc / tot) if tot else np.nan


def fairness_all(y, score, df, S, R=None, thr=0.5, min_n=30, min_pn=5):
    """All settings -> {setting_name: metrics}; plus 'CSP_R' and the full-intersection SP 'dSP'."""
    yhat = (np.asarray(score) >= thr).astype(int)
    out = {setting_name(a): fairness_setting(y, yhat, score, df, a, min_n, min_pn) for a in settings(S)}
    out["_CSP_R"] = csp_r(y, yhat, df, S, R or []) if R is not None else np.nan
    out["_dSP"] = out[setting_name(list(S))]["SP"]
    out["_Theil"] = theil_index(y, yhat)
    return out


def theil_index(y, yhat):
    """Individual fairness (Speicher et al. 2018): benefit b_i = yhat_i - y_i + 1, Theil index of b."""
    b = np.asarray(yhat, float) - np.asarray(y, float) + 1.0
    mu = b.mean()
    if mu <= 0:
        return np.nan
    r = b / mu
    return float(np.mean(np.where(r > 0, r * np.log(np.where(r > 0, r, 1.0)), 0.0)))


def summarize_SC(fa, S):
    """Mean of each metric over single settings (S) and over combinations (C)."""
    res = {}
    for m in METRICS:
        single = [fa[setting_name([s])][m] for s in S]
        combo = [fa[setting_name(a)][m] for a in settings(S) if len(a) >= 2]
        res[m] = (float(np.nanmean(single)) if np.any(np.isfinite(single)) else np.nan,
                  float(np.nanmean(combo)) if np.any(np.isfinite(combo)) else np.nan)
    return res


def bootstrap_ci(y, score, df, attrs, metric, B=1000, seed=0, min_n=30, min_pn=5):
    """95% CI of one metric for one setting, resampling the test set stratified by cell."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    yhat = (score >= 0.5).astype(int)
    cell = df[attrs].astype(str).agg("|".join, axis=1).values
    idx_by = [np.where(cell == c)[0] for c in pd.unique(cell)]
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by])
        r = fairness_setting(y[idx], yhat[idx], score[idx], df.iloc[idx].reset_index(drop=True), attrs, min_n, min_pn)
        vals.append(r[metric])
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) else (np.nan, np.nan)
