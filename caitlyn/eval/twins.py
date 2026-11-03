"""Twin flip rate: regenerate the descendants of S (outside R) of synthetic rows under another cell,
keep R and non-descendants fixed, and count changed predictions of the TSTR XGBoost classifier,
net of null twins (regenerated under the same cell)."""
import networkx as nx
import numpy as np
import pandas as pd

from ..common import cell_of
from .utility import fit_predict


def descendants_to_regenerate(G2, S, R):
    """Descendants of S reachable without passing through R (R is held fixed in the twins)."""
    H = G2.copy()
    H.remove_nodes_from([r for r in R if r in H])
    d = set().union(*[nx.descendants(H, s) for s in S if s in H])
    return [v for v in d if v not in S and v not in R]


def make_twins(gen, rows, S, R, alt_cells, seed_tag):
    nodes = descendants_to_regenerate(gen.G, S, R)
    if gen.context == "full":                 # full history: every later column sees S
        nodes = [v for v in gen.order if v not in S and v not in R]
    twins = []
    for k, c in enumerate(alt_cells):
        t = rows.copy()
        vals = c.split("|")
        for s, v in zip(S, vals):
            t[s] = pd.Series([v] * len(t), index=t.index).astype(rows[s].dtype) \
                if pd.api.types.is_numeric_dtype(rows[s]) else v
        twins.append(gen.complete(t, nodes, seed_tag=f"{seed_tag}|twin{k}"))
    return twins


def twin_flip_rate(gen, synth, train, test, S, R, Y, seed, n_rows=1000, n_alt=2):
    rng = np.random.default_rng(seed)
    rows = synth.sample(min(n_rows, len(synth)), random_state=seed).reset_index(drop=True)
    cells = sorted(cell_of(train, S).unique())
    own = cell_of(rows, S).values
    alts = [[c for c in rng.permutation(cells) if c != own[i]][:n_alt] for i in range(len(rows))]

    def regenerate(assign, tag):
        """assign[i] = cell for row i; rows are grouped by target cell for batching."""
        out = rows.copy().astype(object)
        groups = pd.Series(assign)
        for c in groups.unique():
            idx = groups.index[groups == c]
            out.loc[idx] = make_twins(gen, rows.loc[idx], S, R, [c], tag)[0][rows.columns].values
        return out

    twins = [regenerate([a[k] if k < len(a) else own[i] for i, a in enumerate(alts)], f"twins|{seed}|alt{k}")
             for k in range(n_alt)]
    null = regenerate(list(own), f"twins|{seed}|null")
    frames = [rows] + twins + [null]
    keep = np.ones(len(rows), bool)
    for f in frames:
        keep &= f.notna().all(1).values
    frames = [f[keep].reset_index(drop=True) for f in frames]
    for f in frames:
        for c in f.columns:
            f[c] = pd.to_numeric(f[c]) if pd.api.types.is_numeric_dtype(train[c]) else f[c].astype(str)
    allx = pd.concat(frames, ignore_index=True)
    pred = fit_predict(synth, allx, Y, "xgb", seed, train) >= 0.5    # one TSTR classifier for all frames
    n = keep.sum()
    P = [pred[i * n:(i + 1) * n] for i in range(len(frames))]
    Yv = [pd.to_numeric(f[Y]).values for f in frames]
    flip = float(np.mean([np.mean(P[k] != P[0]) for k in range(1, n_alt + 1)]))
    null_flip = float(np.mean(P[-1] != P[0]))
    lab = float(np.mean([np.mean(Yv[k] != Yv[0]) for k in range(1, n_alt + 1)]))
    lab0 = float(np.mean(Yv[-1] != Yv[0]))
    return dict(flip=flip, null_flip=null_flip, net_flip=flip - null_flip, label_flip=lab, label_null=lab0,
                net_label_flip=lab - lab0, n_rows=int(n))
