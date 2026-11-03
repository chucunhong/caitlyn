"""Preprocess Adult, German credit and Credit scoring.

  python scripts/prepare_data.py --dataset all
Writes data/processed/<name>/data.csv and prints cell counts per split (supplement Sec. E).
"""
import argparse
import json
import urllib.request

import _common  # noqa: F401
import numpy as np
import pandas as pd

from caitlyn.common import cell_of
from caitlyn.config import DATA, load_dataset_meta, load_run_config
from caitlyn.data import make_split

RAW = DATA / "raw"
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise SystemExit(f"Download failed ({e}). Place the file manually at {path}.")
    return path


def save(name, df):
    for c in df.columns:                          # strip stray spaces in category labels (e.g. '>= 200 DM ')
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    out = DATA / "processed" / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "data.csv", index=False)
    meta = load_dataset_meta(name)
    missing = [c for c in df.columns if c not in meta.columns]
    if missing:
        print(f"  WARNING: no description in configs for columns {missing}")
    report = {"rows": len(df), "label_rate": float(df[meta.label].mean()), "cells": {}}
    for seed in range(5):
        tr, te = make_split(df, meta.protected, meta.label, seed)
        report["cells"][seed] = {"train": cell_of(tr, meta.protected).value_counts().to_dict(),
                                 "test": cell_of(te, meta.protected).value_counts().to_dict()}
    (out / "cells.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"  {name}: {len(df)} rows, positive rate {report['label_rate']:.3f}, "
          f"test cells (seed 0): {report['cells'][0]['test']}")


def adult():
    if (RAW / "zeri_adult.csv").exists():          # cleaned copy shared with the group's earlier work (45,222 rows)
        df = pd.read_csv(RAW / "zeri_adult.csv").rename(columns={
            "educational-num": "education_num", "gender": "sex", "class-label": "income"})
        df.columns = [c.replace("-", "_") for c in df.columns]
        df["race"] = np.where(df["race"] == "White", "White", "non-White")
        df["age"] = pd.cut(df["age"], [0, 30, 50, 200], labels=["<=30", "31-50", ">50"]).astype(str)
        df = df.drop(columns=["fnlwgt"]).reset_index(drop=True)
        save("adult", df)
        return
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
            "income"]
    a = pd.read_csv(fetch(f"{UCI}/adult/adult.data", RAW / "adult.data"), names=cols, skipinitialspace=True)
    b = pd.read_csv(fetch(f"{UCI}/adult/adult.test", RAW / "adult.test"), names=cols, skipinitialspace=True,
                    skiprows=1)
    df = pd.concat([a, b], ignore_index=True).replace("?", np.nan).dropna()
    df["income"] = df["income"].str.rstrip(".").eq(">50K").astype(int)
    df["race"] = np.where(df["race"] == "White", "White", "non-White")
    df["age"] = pd.cut(df["age"], [0, 30, 50, 200], labels=["<=30", "31-50", ">50"]).astype(str)
    df = df.drop(columns=["fnlwgt"]).reset_index(drop=True)
    save("adult", df)


GERMAN_CODES = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking account"},
    "credit_history": {"A30": "no credits taken / all paid back duly", "A31": "all credits at this bank paid back duly",
                       "A32": "existing credits paid back duly till now", "A33": "delay in paying off in the past",
                       "A34": "critical account / other credits existing"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/television",
                "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "others"},
    "savings": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
                "A65": "unknown / no savings account"},
    "employment": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "building society savings / life insurance",
                 "A123": "car or other", "A124": "unknown / no property"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee / official", "A174": "management / self-employed / highly qualified"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


ZERI_GERMAN = {"checking-account": "checking_status", "credit-history": "credit_history", "credit-amount": "credit_amount",
               "savings-account": "savings", "employment-since": "employment", "installment-rate": "installment_rate",
               "other-debtors": "other_debtors", "residence-since": "residence_since",
               "other-installment": "other_installment_plans", "existing-credits": "existing_credits",
               "number-people-provide-maintenance-for": "num_dependents", "foreign-worker": "foreign_worker",
               "class-label": "credit"}


def german():
    if (RAW / "zeri_german.csv").exists():         # decoded copy (sex already derived from personal_status)
        df = pd.read_csv(RAW / "zeri_german.csv").rename(columns=ZERI_GERMAN)
        df = df.drop(columns=["marital-status"])    # not used: A92 merges several statuses for women
        df["age"] = np.where(df["age"] <= 25, "<=25", ">25")
        df["credit"] = df["credit"].astype(int)
        save("german", df)
        return
    cols = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings", "employment",
            "installment_rate", "personal_status", "other_debtors", "residence_since", "property", "age",
            "other_installment_plans", "housing", "existing_credits", "job", "num_dependents", "telephone",
            "foreign_worker", "credit"]
    df = pd.read_csv(fetch(f"{UCI}/statlog/german/german.data", RAW / "german.data"), sep=" ", names=cols)
    for c, m in GERMAN_CODES.items():
        df[c] = df[c].map(m)
    # sex from personal_status: A91, A93, A94 male; A92, A95 female. Marital status is not used.
    df["sex"] = np.where(df["personal_status"].isin(["A92", "A95"]), "female", "male")
    df["age"] = np.where(df["age"] <= 25, "<=25", ">25")
    df["credit"] = (df["credit"] == 1).astype(int)
    df = df.drop(columns=["personal_status"])
    df = df[[c if c != "personal_status" else "sex" for c in cols]]
    save("german", df)


def credit_scoring():
    meta = load_dataset_meta("credit_scoring")
    r = meta.raw
    path = DATA.parent / r["path"]
    if not path.exists():
        raise SystemExit(f"Place the Kaggle file at {path} (see configs/datasets/credit_scoring.yaml)")
    raw = pd.read_csv(path)
    print("raw columns:", list(raw.columns))
    need = [r["label_col"], r["sex_col"], r["age_col"]] + ([r["marital_col"]] if r.get("marital_col") else [])
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise SystemExit(f"Edit configs/datasets/credit_scoring.yaml: columns {miss} not found")
    df = raw.drop(columns=r.get("drop", [])).copy()
    df["sex"] = df.pop(r["sex_col"]).map(lambda v: r["sex_map"].get(v, r["sex_map"].get(str(v), str(v))))
    age = df.pop(r["age_col"])
    thr = int(age.median()) if r["age_threshold"] == "median" else int(r["age_threshold"])
    df["age"] = np.where(age <= thr, f"<={thr}", f">{thr}")
    print(f"  age threshold {thr}: {int((age <= thr).sum())} rows <= {thr}")
    if r.get("marital_col"):                       # optional protected marital status (needs a codebook)
        married = {str(v) for v in r["married_values"]}
        df["marital"] = np.where(df.pop(r["marital_col"]).astype(str).isin(married), "married", "not married")
    df = df.rename(columns={r["label_col"]: meta.label})
    df[meta.label] = df[meta.label].astype(int)
    df = df.dropna().reset_index(drop=True)
    save("credit_scoring", df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "adult", "german", "credit_scoring"])
    a = ap.parse_args()
    cfg = load_run_config()
    jobs = {"adult": adult, "german": german, "credit_scoring": credit_scoring}
    for name, fn in jobs.items():
        if a.dataset in ("all", name):
            print(f"[{name}]")
            try:
                fn()
            except SystemExit as e:
                if a.dataset != "all":
                    raise
                print(f"  skipped: {e}")
