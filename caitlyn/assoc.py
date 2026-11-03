"""Association scores on a 0-1 scale: bias-corrected Cramer's V (Bergsma 2013) for
categorical pairs, |Pearson r| for numerical pairs, correlation ratio eta for mixed pairs.
Fast versions on integer codes are used inside permutation tests."""
import numpy as np
import pandas as pd

from .common import is_numeric


def codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s.astype(str))[0]


def cramers_v_codes(a: np.ndarray, b: np.ndarray) -> float:
    ka, kb = a.max() + 1, b.max() + 1
    n = len(a)
    if ka < 2 or kb < 2:
        return 0.0
    tab = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((tab - exp) ** 2 / np.where(exp > 0, exp, np.nan))
    phi2 = chi2 / n
    r, k = ka, kb
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else 0.0


def corr_ratio_codes(cat: np.ndarray, num: np.ndarray) -> float:
    k = cat.max() + 1
    cnt = np.bincount(cat, minlength=k)
    means = np.bincount(cat, weights=num, minlength=k) / np.maximum(cnt, 1)
    mu = num.mean()
    ss_b = float(np.sum(cnt * (means - mu) ** 2))
    ss_t = float(np.sum((num - mu) ** 2))
    return float(np.sqrt(ss_b / ss_t)) if ss_t > 0 else 0.0


def abs_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


class AssocEncoder:
    """Pre-encodes columns once so that repeated (permuted) association scores are cheap."""

    def __init__(self, df: pd.DataFrame):
        self.num = {c: is_numeric(df[c]) for c in df.columns}
        self.enc = {c: (df[c].astype(float).values if self.num[c] else codes(df[c])) for c in df.columns}

    def score(self, a, b, a_values=None):
        x = self.enc[a] if a_values is None else a_values
        y = self.enc[b]
        na, nb = self.num[a], self.num[b]
        if not na and not nb:
            return cramers_v_codes(x, y)
        if na and nb:
            return abs_pearson(x, y)
        return corr_ratio_codes(x, y) if not na else corr_ratio_codes(y, x)


def assoc(a: pd.Series, b: pd.Series) -> float:
    enc = AssocEncoder(pd.DataFrame({"a": a.values, "b": b.values}))
    return enc.score("a", "b")


def assoc_table(df: pd.DataFrame, u: str) -> dict:
    enc = AssocEncoder(df)
    return {v: round(enc.score(u, v), 3) for v in df.columns if v != u}
