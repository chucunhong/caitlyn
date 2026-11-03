"""Loading, splitting, column domains and row serialization for prompts."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import cell_of, is_numeric
from .config import DATA, load_dataset_meta


def processed_path(name):
    return DATA / "processed" / name / "data.csv"


def load_processed(name) -> pd.DataFrame:
    return pd.read_csv(processed_path(name))


def make_split(df, S, y, seed, test_size=0.3):
    strata = df[list(S) + [y]].astype(str).agg("|".join, axis=1)
    vc = strata.value_counts()
    strata = strata.where(strata.map(vc) >= 2, "rare")   # very rare strata: merge so stratify works
    tr, te = train_test_split(df, test_size=test_size, stratify=strata, random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def load_split(name, seed, n_train=None):
    """Train/test split of a processed dataset; optional subsample of the train split (sensitivity in n)."""
    meta = load_dataset_meta(name)
    df = load_processed(name)
    tr, te = make_split(df, meta.protected, meta.label, seed)
    if n_train and n_train < len(tr):
        tr, _ = train_test_split(tr, train_size=n_train, stratify=cell_of(tr, [meta.label]), random_state=seed)
        tr = tr.reset_index(drop=True)
    return meta, tr, te


def column_domains(df: pd.DataFrame) -> dict:
    """Domain of every column: categorical values, or numeric range widened by 5%."""
    dom = {}
    for c in df.columns:
        s = df[c].dropna()
        if is_numeric(s):
            lo, hi = float(s.min()), float(s.max())
            pad = 0.05 * (hi - lo)
            integer = bool(np.all(np.isclose(s, np.round(s))))
            lo2 = lo - pad if lo - pad >= 0 or lo < 0 else 0.0
            hi2 = hi + pad
            if integer:                                   # integer columns: round the widened range outwards
                lo2, hi2 = float(np.floor(lo2)), float(np.ceil(hi2))
            dom[c] = dict(type="numeric", min=lo2, max=hi2, integer=integer)
        else:
            vals = sorted(s.astype(str).unique().tolist())
            dom[c] = dict(type="categorical", values=vals, numeric_codes=bool(pd.api.types.is_numeric_dtype(s)),
                          norm={v.strip().lower(): v for v in vals})
    return dom


def domain_text(d: dict) -> str:
    if d["type"] == "numeric":
        kind = "integer" if d["integer"] else "number"
        return f"{kind} between {d['min']:.4g} and {d['max']:.4g}"
    return "one of " + json.dumps(d["values"])


def validate_value(v, d: dict):
    """Return the cleaned value, or None if it is outside the column's domain."""
    if v is None:
        return None
    if d["type"] == "numeric":
        try:
            x = float(str(v).replace(",", "").strip())
        except ValueError:
            return None
        if not (d["min"] <= x <= d["max"]) or np.isnan(x):
            return None
        return int(round(x)) if d["integer"] else x
    s = str(v)
    if s in d["values"]:
        return s
    norm = d.get("norm") or {x.strip().lower(): x for x in d["values"]}
    s = s.strip()
    if d.get("numeric_codes"):                # e.g. 1.0 vs "1"
        try:
            f = float(s)
            s = str(int(f)) if f.is_integer() else str(f)
        except ValueError:
            pass
    return norm.get(s.lower())                # tolerant to case and surrounding spaces


def cast_like(col: pd.Series, ref: pd.Series) -> pd.Series:
    """Cast generated string values back to the dtype of the real column."""
    if pd.api.types.is_numeric_dtype(ref):
        return pd.to_numeric(col, errors="coerce")
    return col.astype(object)


def fmt_value(v):
    if isinstance(v, (float, np.floating)):
        return float(f"{v:.4g}")
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def rows_text(df: pd.DataFrame, with_id=True) -> str:
    lines = []
    # to_dict(records) keeps each column's dtype (iterrows would turn integers into floats, e.g. 1 -> 1.0)
    for i, r in zip(df.index, df.to_dict(orient="records")):
        rec = {"id": i} if with_id else {}
        rec.update({c: fmt_value(r[c]) for c in df.columns})
        lines.append(json.dumps(rec, ensure_ascii=False, default=str))
    return "\n".join(lines)
