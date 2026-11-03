"""Fixed settings shared by every auxiliary model (Sec. 3b of the guide, Tables A/B of the supplement)."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

LGB = dict(n_estimators=100, learning_rate=0.1, num_leaves=31, min_child_samples=20,
           subsample=1.0, colsample_bytree=1.0, class_weight=None, verbose=-1)
UNKNOWN = "UNKNOWN"


def stable_seed(*parts) -> int:
    """Process-independent seed (Python's hash() is salted per process and would break the cache)."""
    import hashlib
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.nunique() > 15


def prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if not is_numeric(X[c]):
            X[c] = X[c].astype(str).astype("category")
    return X


def folds(y: pd.Series, seed: int):
    if is_numeric(y):
        return list(KFold(5, shuffle=True, random_state=seed).split(y))
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y.astype(str)))


def _auc(y_true, P, classes):
    """Binary AUC, or macro one-vs-rest AUC over the classes present in y_true."""
    y_true = np.asarray(y_true)
    if P.shape[1] == 2:
        return roc_auc_score(y_true == classes[1], P[:, 1])
    present = [i for i, c in enumerate(classes) if (y_true == c).any() and (y_true != c).any()]
    if not present:
        return 0.5
    return float(np.mean([roc_auc_score(y_true == classes[i], P[:, i]) for i in present]))


def cv_perf(X: pd.DataFrame, y: pd.Series, seed: int, fold_idx=None) -> float:
    """AUC (binary), macro OvR AUC (multiclass) or R^2 (numerical), averaged over 5 folds."""
    numeric = is_numeric(y)
    if X.shape[1] == 0:
        return 0.0 if numeric else 0.5
    X = prep(X).reset_index(drop=True)
    y = y.reset_index(drop=True)
    if not numeric:
        y = y.astype(str)
    fold_idx = fold_idx or folds(y, seed)
    scores = []
    for tr, te in fold_idx:
        if numeric:
            m = LGBMRegressor(**LGB, random_state=seed).fit(X.iloc[tr], y.iloc[tr])
            scores.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
        else:
            if y.iloc[tr].nunique() < 2:
                scores.append(0.5)
                continue
            m = LGBMClassifier(**LGB, random_state=seed).fit(X.iloc[tr], y.iloc[tr])
            scores.append(_auc(y.iloc[te].values, m.predict_proba(X.iloc[te]), list(m.classes_)))
    return float(np.mean(scores))


def cv_accuracy(X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    """Stratified 5-fold accuracy of the fixed LightGBM classifier (leakage probe)."""
    X, y = prep(X).reset_index(drop=True), y.reset_index(drop=True)
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        m = LGBMClassifier(**LGB, random_state=seed).fit(X.iloc[tr], y.iloc[tr])
        accs.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))
    return float(np.mean(accs))


def perm_p(obs, null) -> float:
    return (1 + sum(n >= obs for n in null)) / (len(null) + 1)


def holm(pvals: dict) -> dict:
    # a non-finite p-value (e.g. a failed fit) counts as 1; NaN would otherwise break the sort order
    pvals = {k: (float(p) if p is not None and np.isfinite(p) else 1.0) for k, p in pvals.items()}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for k, (key, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        out[key] = run
    return out


def cell_of(df: pd.DataFrame, S) -> pd.Series:
    return df[list(S)].astype(str).agg("|".join, axis=1)


def to_jsonable(o):
    """Recursively convert numpy / tuple keys for json.dump."""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "->".join(map(str, k)) if isinstance(k, tuple) else str(k)):
                to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o
