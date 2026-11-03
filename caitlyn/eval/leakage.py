"""Leakage probe eps_hat_v (paper Sec. 4.4, supplement Table B).

For a node v in A, the same M parent contexts are generated twice: once with exemplars and batch
neighbours from cell s, once from cell s'. A LightGBM classifier on (parents, output) predicts the
condition; eps_hat_v = max(0, 2 acc - 1) is a lower bound on the TV leakage of the call.

Conditions (the Stage D controls switched on cumulatively):
  full_history   : ctx = all preceding columns incl. S (taken from rows of the cell), exemplars and
                   neighbours from the cell
  parent_only    : ctx = pa(v); exemplars and neighbours from the cell
  parent_indep   : ctx = pa(v); exemplars independent of S; neighbours from the cell
  caitlyn        : ctx = pa(v); exemplars independent of S; neighbours mixed (null calibration)
"""
import itertools

import numpy as np
import pandas as pd

import networkx as nx

from ..common import cell_of, cv_accuracy
from ..stage_b_proxy import unresolved_graph

CONDITIONS = ["full_history", "parent_only", "parent_indep", "caitlyn"]


def nodes_A(G2, R, Y):
    """Nodes from which Y is reachable in G' without passing through R (Y included)."""
    H = unresolved_graph(G2, R)
    return (nx.ancestors(H, Y) | {Y}) if Y in H else {Y}


def top_sp_pairs(train, S, Y, n_pairs=3):
    cell = cell_of(train, S)
    rate = train.groupby(cell)[Y].mean()
    rate = rate[cell.value_counts().reindex(rate.index) >= 30]
    pairs = sorted(itertools.combinations(rate.index, 2), key=lambda p: -abs(rate[p[0]] - rate[p[1]]))
    return pairs[:n_pairs]


def _contexts(gen, v, train, cell, s, ctx_pa, condition, seed):
    """Context frame for condition and cell s: parents fixed, other visible columns from rows of s."""
    pa = list(gen.G.predecessors(v))
    if condition == "full_history":
        pos = gen.real_order.index(v)
        hist = [c for c in gen.real_order[:pos]]
        rows = train[cell == s].sample(len(ctx_pa), replace=True, random_state=seed).reset_index(drop=True)
        frame = rows[hist].copy()
        for p in pa:
            if p in frame.columns:
                frame[p] = ctx_pa[p].values
        return frame, hist
    return ctx_pa.copy(), pa


def eps_hat(gen, v, train, S, s, s2, seed, M=100, probe_per_batch=5, condition="caitlyn"):
    real_pa = [p for p in gen.G.predecessors(v) if not gen.G.nodes[p].get("hidden")]
    hidden_pa = [p for p in gen.G.predecessors(v) if gen.G.nodes[p].get("hidden")]
    cell = cell_of(train, S)
    base = train.sample(M, random_state=seed).reset_index(drop=True)
    ctx_pa = base[real_pa].copy()
    for h in hidden_pa:                       # surrogate parents: use the real value of the original column
        ctx_pa[h] = base[gen.G.nodes[h]["surrogate_of"]].values
    outs = []
    for k, c in enumerate((s, s2)):
        frame, cols = _contexts(gen, v, train, cell, c, ctx_pa, condition, seed + 7)
        pool_c = train[cell == c]
        ex_pool = pool_c if condition in ("full_history", "parent_only") else train
        nb = pool_c if condition != "caitlyn" else train
        nb = nb.copy()
        for h in hidden_pa:
            nb[h] = nb[gen.G.nodes[h]["surrogate_of"]]
        o = gen.gen_column(v, frame, seed_tag=f"leak|{condition}|{seed}|{v}|{k}", ctx_cols=cols,
                           exemplar_pool=ex_pool, neighbours=nb, probe_per_batch=probe_per_batch)
        outs.append(o.reset_index(drop=True))
    X = pd.concat([ctx_pa.assign(out=outs[0].values), ctx_pa.assign(out=outs[1].values)], ignore_index=True)
    y = pd.Series([0] * M + [1] * M)
    keep = X["out"].notna().values
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    if y.nunique() < 2 or y.value_counts().min() < 5:
        return np.nan
    if pd.api.types.is_numeric_dtype(train[gen.target_of(v)]):
        X["out"] = pd.to_numeric(X["out"], errors="coerce")
    return max(0.0, 2 * cv_accuracy(X, y, seed) - 1)


def leakage_report(gen, A_nodes, train, S, Y, seed, M=100, probe_per_batch=5, n_pairs=3, conditions=CONDITIONS):
    pairs = top_sp_pairs(train, S, Y, n_pairs)
    rep = {}
    for cond in conditions:
        per_v = {}
        for v in A_nodes:
            vals = [eps_hat(gen, v, train, S, a, b, seed, M, probe_per_batch, cond) for a, b in pairs]
            vals = [x for x in vals if x == x]
            per_v[v] = max(vals) if vals else np.nan
        rep[cond] = dict(per_node=per_v, sum=float(np.nansum(list(per_v.values()))))
    return rep, [list(p) for p in pairs]
