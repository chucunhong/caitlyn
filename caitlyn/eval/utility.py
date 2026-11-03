"""Utility: TSTR and augmentation with XGBoost, RF, LR and MLP (supplement Sec. D)."""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..common import is_numeric


def make_clf(name, seed):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed,
                             n_jobs=4, eval_metric="logloss", verbosity=0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=4)
    if name == "lr":
        return LogisticRegression(C=1.0, penalty="l2", max_iter=2000)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=seed, max_iter=300)
    raise ValueError(name)


def column_types(ref: pd.DataFrame, label):
    feats = [c for c in ref.columns if c != label]
    num = [c for c in feats if is_numeric(ref[c])]
    return feats, num, [c for c in feats if c not in num]


def to_features(df, feats, num):
    X = df[feats].copy()
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float) if c in num else X[c].astype(str)
    return X


def fit_predict(train, test, label, clf_name, seed, ref, weights=None):
    """Fit on `train` (encoders fitted on it) and return scores P(y=1) on `test`.
    Column types come from the real training split `ref`."""
    feats, num, cat = column_types(ref, label)
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pipe = Pipeline([("pre", pre), ("clf", make_clf(clf_name, seed))])
    y = train[label].astype(int).values
    if len(np.unique(y)) < 2:                       # degenerate synthetic label
        return np.full(len(test), float(y.mean()))
    kw = {}
    if weights is not None:
        if clf_name == "mlp":                       # MLP has no sample weights: resample
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train), size=len(train), p=weights / weights.sum())
            train, y = train.iloc[idx], y[idx]
        else:
            kw["clf__sample_weight"] = weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(to_features(train, feats, num), y, **kw)
    return pipe.predict_proba(to_features(test, feats, num))[:, 1]


def utility_metrics(y, score, thr=0.5):
    yhat = (score >= thr).astype(int)
    return dict(AUC=float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
                BA=float(balanced_accuracy_score(y, yhat)),
                F1=float(f1_score(y, yhat, average="macro")))
