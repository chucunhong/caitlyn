"""Average ranks, Friedman / Nemenyi (with critical-difference diagram) and Wilcoxon-Holm."""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from ..common import holm

# Studentized range q_alpha / sqrt(2) for alpha = 0.05 (Demsar 2006), k = 2..20
_Q05 = [1.960, 2.344, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313, 3.354, 3.391,
        3.426, 3.458, 3.489, 3.517, 3.544]


def block_ranks(df, block_cols, method_col, value_col, higher_is_better=True):
    """Rank methods within each block (dataset x classifier x seed); returns wide table of ranks."""
    wide = df.pivot_table(index=block_cols, columns=method_col, values=value_col, aggfunc="mean")
    wide = wide.dropna(axis=0, how="any")
    ranks = wide.rank(axis=1, ascending=not higher_is_better, method="average")
    return wide, ranks


def friedman_nemenyi(wide, ranks):
    k, n = wide.shape[1], wide.shape[0]
    stat, p = friedmanchisquare(*[wide[c].values for c in wide.columns]) if k >= 3 and n >= 2 else (np.nan, np.nan)
    cd = _Q05[k - 2] * np.sqrt(k * (k + 1) / (6.0 * n)) if 2 <= k <= 20 else np.nan
    return dict(friedman_stat=float(stat), friedman_p=float(p), cd=float(cd), n_blocks=n,
                avg_rank=ranks.mean().sort_values().to_dict())


def wilcoxon_holm(wide, ours, higher_is_better=True):
    """One-sided Wilcoxon signed-rank test of `ours` against each other method, Holm-corrected."""
    pv = {}
    for m in wide.columns:
        if m == ours:
            continue
        d = wide[ours] - wide[m]
        d = d if higher_is_better else -d
        if np.allclose(d, 0):
            pv[m] = 1.0
            continue
        try:
            pv[m] = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            pv[m] = 1.0
    return holm(pv) if pv else {}


def plot_cd(avg_rank: dict, cd: float, ax, title="", highlight=None):
    """Demsar-style critical-difference diagram on a matplotlib axis."""
    names = sorted(avg_rank, key=avg_rank.get)
    r = np.array([avg_rank[n] for n in names])
    k = len(names)
    lo, hi = 1, max(k, int(np.ceil(r.max())))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(-(k // 2 + 2) * 0.35, 1.45)
    ax.axis("off")
    ax.hlines(0.6, lo, hi, color="#52514e", lw=0.8)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0.6, 0.72, color="#52514e", lw=0.8)
        ax.text(t, 0.8, str(t), ha="center", va="bottom", fontsize=7, color="#52514e")
    half = (k + 1) // 2
    for i, (n, x) in enumerate(zip(names, r)):
        left = i < half
        level = (i if left else k - 1 - i)
        y = -0.25 - 0.3 * level
        xe = lo - 0.2 if left else hi + 0.2
        bold = "bold" if n == highlight else "normal"
        ax.plot([x, x, xe], [0.6, y, y], color="#0b0b0b", lw=0.7)
        ax.text(xe + (-0.05 if left else 0.05), y, f"{n} ({x:.2f})", ha="right" if left else "left",
                va="center", fontsize=7, fontweight=bold)
    # cliques of methods whose ranks differ by < CD
    cliques, i = [], 0
    for i in range(k):
        j = max([j for j in range(i, k) if r[j] - r[i] < cd], default=i)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (a, b) in enumerate(cliques):
        y = 0.45 - 0.08 * c
        ax.hlines(y, r[a] - 0.03, r[b] + 0.03, color="#0b0b0b", lw=2)
    ax.hlines(1.2, lo, lo + cd, color="#0b0b0b", lw=1)
    ax.text(lo + cd / 2, 1.23, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    if title:
        ax.set_title(title, fontsize=8)
