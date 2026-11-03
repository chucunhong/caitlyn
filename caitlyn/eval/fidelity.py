"""Fidelity: SDMetrics column shapes / pair trends, alpha-precision and beta-recall, density/coverage."""
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..common import is_numeric


def _metadata(real):
    return {"columns": {c: {"sdtype": "numerical" if is_numeric(real[c]) else "categorical"} for c in real.columns}}


def _align(real, synth):
    synth = synth[real.columns].copy()
    real = real.copy()
    for c in real.columns:
        if is_numeric(real[c]):
            real[c], synth[c] = real[c].astype(float), pd.to_numeric(synth[c], errors="coerce").astype(float)
        else:
            real[c], synth[c] = real[c].astype(str), synth[c].astype(str)
    return real, synth.dropna()


def shape_trend(real, synth):
    real, synth = _align(real, synth)
    try:
        from sdmetrics.reports.single_table import QualityReport
        rep = QualityReport()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep.generate(real, synth, _metadata(real), verbose=False)
        props = rep.get_properties().set_index("Property")["Score"]
        return float(props["Column Shapes"]), float(props["Column Pair Trends"])
    except ImportError:
        return _shape_trend_local(real, synth)


def _shape_trend_local(real, synth):
    """Same definitions as SDMetrics (KS/TV complement; correlation/contingency similarity)."""
    from scipy.stats import ks_2samp
    cols = list(real.columns)
    num = {c: is_numeric(real[c]) for c in cols}
    shapes = []
    for c in cols:
        if num[c]:
            shapes.append(1 - ks_2samp(real[c], synth[c]).statistic)
        else:
            p, q = real[c].value_counts(normalize=True), synth[c].value_counts(normalize=True)
            shapes.append(1 - 0.5 * p.subtract(q, fill_value=0).abs().sum())
    trends = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if num[a] and num[b]:
                r1, r2 = real[a].corr(real[b]), synth[a].corr(synth[b])
                trends.append(1 - abs(np.nan_to_num(r1) - np.nan_to_num(r2)) / 2)
            else:
                ra = pd.qcut(real[a], 10, duplicates="drop").astype(str) if num[a] else real[a]
                rb = pd.qcut(real[b], 10, duplicates="drop").astype(str) if num[b] else real[b]
                sa = pd.cut(synth[a], pd.qcut(real[a], 10, duplicates="drop", retbins=True)[1], include_lowest=True).astype(str) if num[a] else synth[a]
                sb = pd.cut(synth[b], pd.qcut(real[b], 10, duplicates="drop", retbins=True)[1], include_lowest=True).astype(str) if num[b] else synth[b]
                p = pd.crosstab(ra, rb, normalize=True).stack()
                q = pd.crosstab(sa, sb, normalize=True).stack()
                trends.append(1 - 0.5 * p.subtract(q, fill_value=0).abs().sum())
    return float(np.mean(shapes)), float(np.mean(trends))


def embed(real, synth):
    num = [c for c in real.columns if is_numeric(real[c])]
    cat = [c for c in real.columns if c not in num]
    parts_r, parts_s = [], []
    if num:
        sc = StandardScaler().fit(real[num].astype(float))
        parts_r.append(sc.transform(real[num].astype(float)))
        parts_s.append(sc.transform(synth[num].astype(float)))
    if cat:
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(real[cat].astype(str))
        parts_r.append(oh.transform(real[cat].astype(str)))
        parts_s.append(oh.transform(synth[cat].astype(str)))
    return np.hstack(parts_r), np.hstack(parts_s)


def alpha_beta(R, Sy, n_steps=30):
    """Integrated alpha-precision and beta-recall (Alaa et al. 2022), naive estimator as in synthcity."""
    alphas = np.linspace(0, 1, n_steps)
    center = R.mean(0)
    radii = np.quantile(np.linalg.norm(R - center, axis=1), alphas)
    synth_center = Sy.mean(0)
    s2c = np.linalg.norm(Sy - center, axis=1)
    r2r = NearestNeighbors(n_neighbors=2).fit(R).kneighbors(R)[0][:, 1]
    d, idx = NearestNeighbors(n_neighbors=1).fit(Sy).kneighbors(R)
    r2s = d[:, 0]
    closest_d = np.linalg.norm(Sy[idx[:, 0]] - synth_center, axis=1)
    closest_radii = np.quantile(closest_d, alphas)
    ap = np.array([np.mean(s2c <= radii[k]) for k in range(n_steps)])
    bc = np.array([np.mean((r2s <= r2r) * (closest_d <= closest_radii[k])) for k in range(n_steps)])
    a_prec = 1 - np.sum(np.abs(alphas - ap)) / np.sum(alphas)          # synthcity's Delta_precision_alpha
    b_rec = 1 - np.sum(np.abs(alphas - bc)) / np.sum(alphas)           # and Delta_coverage_beta
    return float(a_prec), float(b_rec)


def density_coverage(R, Sy, k=5):
    """Naeem et al. 2020."""
    radii = NearestNeighbors(n_neighbors=k + 1).fit(R).kneighbors(R)[0][:, -1]
    d, idx = NearestNeighbors(n_neighbors=k).fit(R).kneighbors(Sy)
    density = float(np.mean(np.sum(d <= radii[idx], axis=1)) / k)
    d1 = NearestNeighbors(n_neighbors=1).fit(Sy).kneighbors(R)[0][:, 0]
    coverage = float(np.mean(d1 <= radii))
    return density, coverage


def fidelity(real, synth, max_rows=10000, seed=0):
    real, synth = _align(real, synth)
    rs = real.sample(min(len(real), max_rows), random_state=seed)
    ss = synth.sample(min(len(synth), max_rows), random_state=seed)
    shape, trend = shape_trend(rs, ss)
    # alpha/beta and density/coverage compare nearest-neighbour radii, so both samples have the same size
    R, Sy = embed(rs.sample(min(len(rs), len(ss)), random_state=seed), ss)
    a, b = alpha_beta(R, Sy)
    dens, cov = density_coverage(R, Sy)
    return dict(Shape=shape, Trend=trend, alpha_precision=a, beta_recall=b, density=dens, coverage=cov)
