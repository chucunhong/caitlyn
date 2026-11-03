"""Stage C: joint path surgery (paper Sec. 3.3, Eq. 2, Proposition 3).

H = G without R, plus omega -> S_i (infinite capacity); capacities w_uv = max(eta, Delta_uv - gamma 1[u in P]).
The minimum omega-Y cut blocks every unresolved path from any protected attribute to Y.
"""
import networkx as nx
import numpy as np

from .common import cv_perf, folds, is_numeric
from .stage_b_proxy import unresolved_graph
from .stage_d_generate import SUR

OMEGA = "__omega__"


def edge_cost(df, u, v, parents, proxies, seed, eta=1e-3, gamma=0.01):
    fi = folds(df[v], seed)
    full = cv_perf(df[parents], df[v], seed, fi)
    rest = [p for p in parents if p != u]
    drop = cv_perf(df[rest], df[v], seed, fi) if rest else (0.0 if is_numeric(df[v]) else 0.5)
    delta = full - drop
    return max(eta, delta - gamma * (u in proxies)), float(delta)


def unresolved_edges(G, S, R, Y):
    """Edges of H lying on some omega ~> Y path (only these matter for the cut)."""
    H = unresolved_graph(G, R)
    if Y not in H:
        return []
    from_S = set(S) | set().union(*[nx.descendants(H, s) for s in S if s in H])
    to_Y = nx.ancestors(H, Y) | {Y}
    return [(u, v) for u, v in H.edges() if u in from_S and v in to_Y]


def edge_costs(G, train, S, R, Y, proxies, seed, eta, gamma, cache=None):
    cache = {} if cache is None else cache
    cost, delta = {}, {}
    for (u, v) in unresolved_edges(G, S, R, Y):
        parents = sorted(G.predecessors(v))
        key = (u, v, tuple(parents), u in proxies)
        if key not in cache:
            cache[key] = edge_cost(train, u, v, parents, proxies, seed, eta, gamma)
        cost[(u, v)], delta[(u, v)] = cache[key]
    return cost, delta


def joint_surgery(G, S, R, Y, cost):
    H = nx.DiGraph()
    for (u, v), w in cost.items():
        if u in R or v in R:
            continue                                   # paths through R are resolved
        H.add_edge(u, v, capacity=w)
    for s in S:
        H.add_edge(OMEGA, s, capacity=float("inf"))
    if Y not in H or not nx.has_path(H, OMEGA, Y):
        return set()
    _, (reach, nonreach) = nx.minimum_cut(H, OMEGA, Y, capacity="capacity")
    return {(u, v) for u in reach for v in H.successors(u) if v in nonreach}


def joint_loss(df, G, v, removed, seed, cache):
    """Loss in cross-validated fit of v when the parents in `removed` are dropped together."""
    parents = sorted(G.predecessors(v))
    key = (v, tuple(parents), tuple(sorted(removed)))
    if key not in cache:
        fi = folds(df[v], seed)
        full = cv_perf(df[parents], df[v], seed, fi)
        rest = [p for p in parents if p not in removed]
        drop = cv_perf(df[rest], df[v], seed, fi) if rest else (0.0 if is_numeric(df[v]) else 0.5)
        cache[key] = max(0.0, full - drop)
    return cache[key]


def cut_loss(df, G, C, seed, cache, eta=1e-3):
    """True cost of a cut: per-node loss of dropping all of its cut in-edges together (non-additive)."""
    by_v = {}
    for u, v in C:
        by_v.setdefault(v, []).append(u)
    return sum(max(eta * len(us), joint_loss(df, G, v, us, seed, cache)) for v, us in by_v.items())


def refined_joint_surgery(G, train, S, R, Y, cost, seed, eta=1e-3, max_iter=10, cache=None):
    """Minimum cut under non-additive costs by iterative reweighting: after each cut, the edges cut into the
    same node share that node's joint loss (when it exceeds the sum of their capacities) and the cut is
    recomputed; the cut with the smallest true cost is returned. Redundant parents of a node each have a
    small marginal loss, so additive capacities would cut them all at once."""
    cache = {} if cache is None else cache
    w = dict(cost)
    best, best_loss, trace = None, float("inf"), []
    for _ in range(max_iter):
        C = joint_surgery(G, S, R, Y, w)
        loss = cut_loss(train, G, C, seed, cache, eta)
        trace.append({"cut": sorted(map(list, C)), "loss": loss})
        if loss < best_loss - 1e-12:
            best, best_loss = C, loss
        changed = False
        by_v = {}
        for u, v in C:
            by_v.setdefault(v, []).append((u, v))
        for v, es in by_v.items():
            L = joint_loss(train, G, v, [u for u, _ in es], seed, cache)
            s = sum(w[e] for e in es)
            if L > s * (1 + 1e-6):
                for e in es:
                    w[e] *= L / s
                changed = True
        if not changed:
            break
    return best if best is not None else set(), best_loss, w, trace


def per_attribute_surgery(G, S_order, R, Y, cost_fn):
    """Sequential min-cuts, one attribute at a time, each on the graph cut by the previous ones.
    cost_fn(G_current, S_i) -> capacities (recomputed, since parent sets change)."""
    Gc, cuts, steps = G.copy(), set(), []
    for s in S_order:
        cost = cost_fn(Gc, s)
        C = joint_surgery(Gc, [s], R, Y, cost)
        steps.append({"attr": s, "cut": sorted(map(list, C)), "weight": float(sum(cost[e] for e in C))})
        Gc.remove_edges_from(C)
        cuts |= C
    return cuts, steps


def cut_all_unresolved(G, S, R, Y):
    """Reference (DECAF-style): remove every edge on an unresolved path."""
    return set(unresolved_edges(G, S, R, Y))


def check_blocked(G2, S, R, Y):
    H = unresolved_graph(G2, R)
    return not any(s in H and Y in H and nx.has_path(H, s, Y) for s in S)


def apply_surgery(G, cuts, cost, S, R, Y, cut_only=False):
    """Build G'. Edges into Y: cut. Other cut edges: surrogate if w_uv > median weight of the cut."""
    G2 = G.copy()
    ops = {}
    w_cut = [cost[e] for e in cuts] or [0.0]
    med = float(np.median(w_cut))
    for (u, v) in cuts:
        ops[(u, v)] = "cut" if (v == Y or cut_only or cost[(u, v)] <= med) else "surrogate"
    G2.remove_edges_from(cuts)
    desc_S = set().union(*[nx.descendants(G2, s) for s in S]) if S else set()
    for (u, v), op in ops.items():
        if op != "surrogate":
            continue
        sur = u + SUR
        masked = [p for p in G.predecessors(u) if p in S or (p in desc_S and p not in R)]
        if sur not in G2:
            G2.add_node(sur, hidden=True, surrogate_of=u, masked=masked)
            for p in G.predecessors(u):
                if p not in masked:
                    G2.add_edge(p, sur)
        G2.add_edge(sur, v)
    assert nx.is_directed_acyclic_graph(G2)
    return G2, ops, med


def surgery(G, train, S, R, Y, proxies, ccfg, seed, mode="joint", S_order=None, per_attr_proxies=None,
            cut_only=False):
    eta, gamma = float(ccfg["eta"]), float(ccfg["gamma"])
    cache = {}
    cost, delta = edge_costs(G, train, S, R, Y, proxies, seed, eta, gamma, cache)
    log = {"cost": cost, "delta": delta}
    op_cost = cost
    if mode == "joint" and ccfg.get("refine_cut", False):
        C, true_loss, op_cost, trace = refined_joint_surgery(G, train, S, R, Y, cost, seed, eta)
        log.update(refined_loss=true_loss, refine_trace=trace, additive_cut=sorted(map(list, joint_surgery(G, S, R, Y, cost))))
    elif mode == "joint":
        C = joint_surgery(G, S, R, Y, cost)
    else:
        order = S_order or list(S)

        def cost_fn(Gc, s):
            P_s = per_attr_proxies[s] if per_attr_proxies else proxies
            c, _ = edge_costs(Gc, train, [s], R, Y, P_s, seed, eta, gamma, cache)
            cost.update({k: v for k, v in c.items() if k not in cost})
            return c

        C, steps = per_attribute_surgery(G, order, R, Y, cost_fn)
        log["steps"] = steps
    # the joint cut and per-attribute cuts are always computed for Proposition 3 (logged in run.json)
    log["joint_cut"] = sorted(map(list, joint_surgery(G, S, R, Y, cost)))
    log["joint_weight"] = float(sum(cost[tuple(e)] for e in log["joint_cut"]))
    per = {s: joint_surgery(G, [s], R, Y, cost) for s in S}
    log["single_cuts"] = {s: sorted(map(list, c)) for s, c in per.items()}
    log["sum_single_weight"] = float(sum(cost[e] for c in per.values() for e in c))
    log["cut_all_n"] = len(cut_all_unresolved(G, S, R, Y))
    G2, ops, med = apply_surgery(G, C, op_cost, S, R, Y, cut_only=cut_only)
    log.update(cut=sorted(map(list, C)), ops={f"{u}->{v}": o for (u, v), o in ops.items()},
               weight=float(sum(cost[e] for e in C)), median_weight=med, blocked=check_blocked(G2, S, R, Y))
    return G2, log
