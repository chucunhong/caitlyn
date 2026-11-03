"""Privacy: DCR ratio with Gower distance, and nearest-neighbour privacy risk (van Breugel et al. 2023)."""
import numpy as np
import pandas as pd

from ..common import is_numeric


class Gower:
    def __init__(self, ref: pd.DataFrame):
        self.cols = list(ref.columns)
        self.num = [c for c in self.cols if is_numeric(ref[c])]
        self.cat = [c for c in self.cols if c not in self.num]
        self.rng = {c: float(ref[c].max() - ref[c].min()) or 1.0 for c in self.num}
        self.cats = {c: {v: i for i, v in enumerate(pd.unique(ref[c].astype(str)))} for c in self.cat}

    def enc(self, df):
        N = np.column_stack([df[c].astype(float).values / self.rng[c] for c in self.num]) if self.num else None
        C = np.column_stack([df[c].astype(str).map(self.cats[c]).fillna(-1).values for c in self.cat]) if self.cat else None
        return N, C

    def min_dist(self, Q: pd.DataFrame, Ref: pd.DataFrame, chunk=256, return_arg=False):
        qn, qc = self.enc(Q)
        rn, rc = self.enc(Ref)
        p = len(self.cols)
        mins, args = np.empty(len(Q)), np.empty(len(Q), int)
        for i in range(0, len(Q), chunk):
            D = np.zeros((min(chunk, len(Q) - i), len(Ref)))
            if qn is not None:
                for j in range(qn.shape[1]):
                    D += np.abs(qn[i:i + chunk, j][:, None] - rn[:, j][None, :])
            if qc is not None:
                for j in range(qc.shape[1]):
                    D += qc[i:i + chunk, j][:, None] != rc[:, j][None, :]
            D /= p
            mins[i:i + chunk], args[i:i + chunk] = D.min(1), D.argmin(1)
        return (mins, args) if return_arg else mins


def privacy(train, hold, synth, max_rows=3000, seed=0):
    cols = list(train.columns)
    synth = synth[cols].dropna()
    g = Gower(train)
    tr = train.sample(min(len(train), 20000), random_state=seed)
    s = synth.sample(min(len(synth), max_rows), random_state=seed)
    h = hold.sample(min(len(hold), max_rows), random_state=seed)
    d_s, d_h = g.min_dist(s, tr), g.min_dist(h, tr)
    dcr = float(np.median(d_s) / max(np.median(d_h), 1e-12))
    # exact copies of a training row and repeated synthetic rows; the holdout gives the rate of real data
    num = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]

    def key(df):
        x = df[cols].copy()
        x[num] = x[num].astype(float).round(6)
        return x.astype(str).agg("|".join, axis=1)

    train_keys = set(key(train))
    copies = dict(copy_rate=float(key(synth).isin(train_keys).mean()), repeat_rate=float(key(synth).duplicated().mean()),
                  copy_rate_holdout=float(key(h).isin(train_keys).mean()),
                  repeat_rate_holdout=float(key(h.head(len(synth))).duplicated().mean()),
                  DCR_mean=float(d_s.mean()), DCR_mean_holdout=float(d_h.mean()),
                  DCR_p5=float(np.quantile(d_s, 0.05)), DCR_p5_holdout=float(np.quantile(d_h, 0.05)))
    # NN risk: reference = train subsample of the same size as the holdout, union holdout
    n = min(len(hold), len(train))
    ref = pd.concat([train.sample(n, random_state=seed + 1), hold.sample(n, random_state=seed + 2)],
                    ignore_index=True)
    _, arg = g.min_dist(s, ref, return_arg=True)
    nn_risk = float(np.mean(arg < n))
    return dict(DCR=dcr, NN_risk=nn_risk, **copies)
