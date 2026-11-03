"""Case study figure and decision table: the plan of one run through Stages A-D.

  python scripts/make_case_study.py --dataset german --method caitlyn__model-deepseek-v4.1-flash --seed 0
Reads run.json (stage signatures), the Stage A/B/C files in results/<dataset>/_stages/seed<k>/ and the
edited plan; writes paper/figures/case_<dataset>.png and the AUTO blocks case_table and case_numbers of paper/main.tex.
"""
import argparse
import json

import _common  # noqa: F401
from _common import ROOT, inject
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

from caitlyn.config import RESULTS, load_dataset_meta
from caitlyn.pipeline import json2g
from caitlyn.stage_b_proxy import unresolved_graph

FIG = ROOT / "paper" / "figures"
ROLE_COLOR = {"protected": "#d62728", "resolving": "#2ca02c", "proxy": "#ff7f0e", "label": "#222222", "other": "#9e9e9e"}


def load_run(ds, method, seed, root=RESULTS):
    run = json.loads((root / ds / method / str(seed) / "run.json").read_text(encoding="utf-8"))
    st = RESULTS / ds / "_stages" / f"seed{seed}"
    sig = run["stage_files"]
    A = json.loads((st / f"A_{sig['A']}.json").read_text(encoding="utf-8"))
    B = json.loads((st / f"B_{sig['B']}.json").read_text(encoding="utf-8"))
    C = json.loads((st / f"C_{sig['C']}.json").read_text(encoding="utf-8"))
    return run, A, B, C


def short(c):
    return c.replace("_", "\n", 1).replace("-", "-\n", 1) if len(c) > 9 else c


def salient_nodes(G0, G, G2, S, Y, R):
    """Protected attributes, label, resolving set, and every node on an S ~> Y path of G0 or G."""
    keep = set(S) | {Y} | set(R)
    for H in (G0, G):
        for s in S:
            if s in H and Y in H:
                down = nx.descendants(H, s) | {s}
                up = nx.ancestors(H, Y) | {Y}
                keep |= down & up
    keep |= set(G.predecessors(Y)) | {G2.nodes[p].get("surrogate_of", p) for p in G2.predecessors(Y)}
    return keep


def layout(G, nodes, S, Y):
    """Layered layout: x = depth in G restricted to the shown nodes (S at 0, Y last), y spread per layer."""
    H = G.subgraph(nodes).copy()
    depth = {}
    for v in nx.topological_sort(H):
        depth[v] = 0 if v in S else 1 + max((depth[p] for p in H.predecessors(v)), default=0)
    ymax = max(depth.values())
    depth[Y] = ymax + 1 if any(depth[v] == depth[Y] and v != Y for v in depth) else depth[Y]
    layers = {}
    for v in sorted(nodes, key=lambda v: (v not in S, v)):
        layers.setdefault(depth[v], []).append(v)
    pos = {}
    for x, vs in layers.items():
        for i, v in enumerate(vs):
            pos[v] = (x, (len(vs) - 1) / 2 - i)
    return pos


def role_of(v, S, Y, R, P):
    return "protected" if v in S else "label" if v == Y else "resolving" if v in R else "proxy" if v in P else "other"


def draw_nodes(ax, pos, nodes, S, Y, R, P, sizes=None, edgecolors=None):
    for v in nodes:
        r = role_of(v, S, Y, R, P)
        ax.scatter(*pos[v], s=(sizes or {}).get(v, 520), c=ROLE_COLOR[r], zorder=3, alpha=0.9,
                   edgecolors=(edgecolors or {}).get(v, "white"), linewidths=2.2)
        ax.text(pos[v][0], pos[v][1] - 0.36, short(v), ha="center", va="top", fontsize=6.3, zorder=4)


def draw_edges(ax, pos, edges, color, style="solid", width=1.0, alpha=0.8, rad=0.08, label=None):
    for i, (u, v) in enumerate(edges):
        if u not in pos or v not in pos:
            continue
        ax.annotate("", xy=pos[v], xytext=pos[u], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=width, ls=style, alpha=alpha,
                                    shrinkA=9, shrinkB=9, mutation_scale=8, connectionstyle=f"arc3,rad={rad}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--method", default="caitlyn__model-deepseek-v4.1-flash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None, help="results root holding run.json (default results/)")
    a = ap.parse_args()
    meta = load_dataset_meta(a.dataset)
    S, Y = list(meta.protected), meta.label
    run, A, B, C = load_run(a.dataset, a.method, a.seed, RESULTS if a.root is None else ROOT / a.root)
    G0, G, G2 = json2g(A["G0"]), json2g(A["G"]), json2g(C["G2"])
    R, P = A["R"], B["proxies"]
    prior = [tuple(e["edge"]) for e in A["prior_driven"]]
    gate_added = [tuple(e["edge"]) for e in A["lrt_added"]]
    label_added = [tuple(e["edge"]) for e in A.get("label_parents_added", [])]
    cut = [tuple(e) for e in C["cut"]]
    ops = C["ops"]
    nodes = salient_nodes(G0, G, G2, S, Y, R) | {u for u, _ in prior} | {v for _, v in prior}
    nodes = {v for v in nodes if v in G}
    pos = layout(G, nodes, S, Y)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    for ax in axes.flat:
        ax.set_axis_off()
    sub = lambda H: [(u, v) for u, v in H.edges() if u in nodes and v in nodes]

    # (a) Stage A
    ax = axes[0, 0]
    llm_kept = [e for e in sub(G0) if e not in prior]
    key = [e for e in llm_kept if e[0] in S or e[1] == Y]
    draw_edges(ax, pos, [e for e in llm_kept if e not in key], "#bdbdbd", width=0.6, alpha=0.3)
    draw_edges(ax, pos, key, "#555555", width=1.1, alpha=0.9)
    draw_edges(ax, pos, prior, "#d62728", style="dashed", width=1.4)
    draw_edges(ax, pos, [e for e in gate_added if e[0] in nodes and e[1] in nodes], "#1f77b4", width=1.4)
    draw_edges(ax, pos, [e for e in label_added if e[0] in nodes], "#2ca02c", width=1.6)
    draw_nodes(ax, pos, nodes, S, Y, R, [])
    ax.set_title(f"(a) Stage A: LLM plan, role typing and evidence gate\n{len(prior)} prior-driven edges rejected, "
                 f"{len(gate_added)} added by the gate, {len(label_added)} label parents added", fontsize=8.5)
    ax.legend(handles=[Line2D([], [], color="#7f7f7f", label="LLM edge kept"),
                       Line2D([], [], color="#d62728", ls="--", label="prior-driven (rejected)"),
                       Line2D([], [], color="#1f77b4", label="added: protected-attribute LRT"),
                       Line2D([], [], color="#2ca02c", label="added: label-parent LRT")],
              loc="lower left", fontsize=6.3, frameon=False)

    # (b) Stage B
    ax = axes[0, 1]
    H = unresolved_graph(G, R)
    on_path = [e for e in sub(H) if e[0] in (set(S) | set().union(*[nx.descendants(H, s) for s in S if s in H]))
               and e[1] in (nx.ancestors(H, Y) | {Y})]
    draw_edges(ax, pos, [e for e in sub(G) if e not in on_path], "#d9d9d9", width=0.5, alpha=0.3)
    draw_edges(ax, pos, on_path, "#555555", width=1.1)
    prs = {u: d["prs"] for u, d in B["nodes"].items()}
    mx = max(prs.values()) if prs else 1.0
    sizes = {u: 300 + 1500 * (prs[u] / mx if mx else 0) for u in prs}
    draw_nodes(ax, pos, nodes, S, Y, R, P, sizes, {u: "#000000" for u in P})
    for u, d in B["nodes"].items():
        if u in pos:
            ax.text(pos[u][0] + 0.12, pos[u][1] + 0.22, f"PC {d['pc']:.2f}\nPU {d['max_pu']:.2f}", fontsize=5.6,
                    color="#444444", zorder=5)
    ax.set_title(f"(b) Stage B: proxy audit on unresolved paths (node size = PRS)\n"
                 f"proxies: {', '.join(P) or 'none'}", fontsize=8.5)

    # (c) Stage C
    ax = axes[1, 0]
    draw_edges(ax, pos, [e for e in on_path if e not in cut], "#555555", width=1.0)
    for k_, e in enumerate(cut):
        op = ops.get(f"{e[0]}->{e[1]}", "cut")
        draw_edges(ax, pos, [e], "#d62728" if op == "cut" else "#9467bd", style="solid" if op == "cut" else "dashed",
                   width=2.4)
        if e[0] in pos and e[1] in pos:
            mx_, my_ = (pos[e[0]][0] + pos[e[1]][0]) / 2, (pos[e[0]][1] + pos[e[1]][1]) / 2
            ax.text(mx_, my_ + 0.12 + 0.22 * (k_ % 2), f"w={C['cost'].get(f'{e[0]}->{e[1]}', float('nan')):.3f}", fontsize=5.8,
                    color="#d62728", ha="center", zorder=5)
    draw_nodes(ax, pos, nodes, S, Y, R, P)
    singles = C.get("single_cuts", {})
    ax.set_title(f"(c) Stage C: joint minimum cut on $H=G\\setminus R$ ({len(cut)} edges, weight "
                 f"{C['weight']:.3f}; per-attribute cuts {C.get('sum_single_weight', float('nan')):.3f})",
                 fontsize=8.5)
    ax.legend(handles=[Line2D([], [], color="#555555", label="unresolved path edge"),
                       Line2D([], [], color="#d62728", lw=2.4, label="cut"),
                       Line2D([], [], color="#9467bd", lw=2.4, ls="--", label="surrogate")],
              loc="lower left", fontsize=6.3, frameon=False)

    # (d) Stage D: G' with call groups
    ax = axes[1, 1]
    shown2 = [v for v in G2.nodes if v in nodes or G2.nodes[v].get("hidden")]
    pos2 = layout(G2, shown2, S, Y)
    H2 = G2.subgraph(shown2)
    strong = [e for e in H2.edges() if e[1] == Y or e[0] in S]
    draw_edges(ax, pos2, [e for e in H2.edges() if e not in strong], "#bdbdbd", width=0.6, alpha=0.35)
    draw_edges(ax, pos2, strong, "#333333", width=1.2)
    from caitlyn.stage_d_generate import Generator

    class _L:                                    # no LLM needed: only the call grouping is used
        cfg = {"max_concurrency": 1}
    import pandas as pd
    from caitlyn.data import load_split
    _, train, _ = load_split(a.dataset, a.seed)
    gen = Generator(_L(), meta, train, G2, {"lam": 0.3, "batch_size": 50, "n_exemplars": 10, "grouping": "layer"},
                    a.seed, R=R)
    groups = gen.call_groups([v for v in gen.order if v not in S])
    for g in groups:
        vs = [v for v in g if v in pos2]
        if not vs:
            continue
        xs, ys = [pos2[v][0] for v in vs], [pos2[v][1] for v in vs]
        inA = gen.A is not None and vs[0] in gen.A
        ax.add_patch(FancyBboxPatch((min(xs) - 0.32, min(ys) - 0.62), max(xs) - min(xs) + 0.64,
                                    max(ys) - min(ys) + 0.98, boxstyle="round,pad=0.02",
                                    fc="#cfe2f3" if inA else "#eeeeee", ec="#6fa8dc" if inA else "#bbbbbb",
                                    lw=0.8, zorder=1))
    draw_nodes(ax, pos2, shown2, S, Y, R, P)
    ax.set_title(f"(d) Stage D: edited plan $G'$ and its {len(groups)} call groups per batch\n"
                 f"(blue boxes: nodes of $A$, whose calls never see a protected attribute; dark: edges into $Y$ or out of $S$)", fontsize=8.5)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in ROLE_COLOR.items()], loc="lower center", ncol=5,
               fontsize=7.5, frameon=False)
    hidden = sorted(set(G.nodes) - nodes - set(S))
    if hidden:
        fig.text(0.5, 0.035, "Not shown (no path from a protected attribute to the label): " + ", ".join(hidden),
                 ha="center", fontsize=6.5, color="#555555")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"case_{a.dataset}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", out)

    # decision table
    votes, pvotes = A["role_log"].get("votes", {}), A["role_log"].get("proxy_votes", {})
    rows = []
    for v in sorted(nodes - set(S) - {Y}, key=lambda c: (c not in R, c not in P, c)):
        rd = A["R_details"].get(v, {})
        if v in R:
            rdec = "resolving"
        elif v in A["R_rejected"]:
            rdec = A["R_rejected"][v].replace("not relevant", "irrelevant")
        else:
            rdec = "--"
        bn = B["nodes"].get(v)
        outs = [f"{e[1]} ({ops.get(f'{e[0]}->{e[1]}', 'cut')})" for e in cut if e[0] == v]
        rows.append(" & ".join([
            v.replace("_", "\\_"), f"{votes.get(v, 0)}/{pvotes.get(v, 0)}",
            f"{rd['gain']:.3f}" if "gain" in rd else "--", rdec.replace("_", "\\_"),
            f"{bn['pc']:.2f}" if bn else "--", f"{bn['max_pu']:.2f}" if bn else "--",
            f"{bn['prs']:.3f}" if bn else "--", "\\checkmark" if v in P else "",
            (", ".join(outs) or "--").replace("_", "\\_")]) + " \\\\")
    for s in S:
        outs = [f"{e[1]} ({ops.get(f'{e[0]}->{e[1]}', 'cut')})" for e in cut if e[0] == s]
        pr = [e[1] for e in prior if e[0] == s]
        rows.append(f"{s} (protected) & -- & -- & -- & -- & -- & -- & & " +
                    (", ".join(outs) or "--").replace("_", "\\_") + " \\\\")
    inject("case_table", r"""\begin{table}[t]
\centering
\caption{Decisions of \method{} for the case study (""" + meta.name.replace("_", " ") + r""", \openbackbone, seed """ + str(a.seed) + r"""). Votes: resolving/proxy votes of $\pi_{\text{role}}$ out of $K=5$; gain: AUC gain of the relevance test; PC, PU, PRS: Stage B scores (audited nodes only); last column: edges leaving the node removed by the joint cut, with the operator.}
\label{tab:case}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcclccccl}
\toprule
Column & Votes & Gain & Role test & PC & PU & PRS & Proxy & Removed edges \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    nums = {
        "CasePrior": len(prior), "CasePriorList": ", ".join(f"{u}$\\to${v}" for u, v in prior[:4]).replace("_", "\\_"),
        "CaseGateAdded": len(gate_added), "CaseLabelAdded": len(label_added),
        "CaseLabelAddedList": ", ".join(u for u, _ in label_added).replace("_", "\\_") or "none",
        "CaseR": ", ".join(R).replace("_", "\\_"), "CaseP": ", ".join(P).replace("_", "\\_") or "none",
        "CaseCut": ", ".join(f"{u}$\\to${v}" for u, v in cut).replace("_", "\\_"), "CaseNcut": len(cut),
        "CaseCutAll": C.get("cut_all_n", ""), "CaseGroups": len(groups),
        "CaseEdgesGzero": G0.number_of_edges(), "CaseEdgesG": G.number_of_edges(),
        "CasePriorIntoY": ", ".join(f"{u}$\\to${v}" for u, v in prior if v == Y).replace("_", "\\_") or "none",
        "CaseRrejProxy": ", ".join(f"{c} ({r.split('(')[-1].rstrip(')')})" for c, r in A["R_rejected"].items()
                                   if r.startswith("proxy")).replace("_", "\\_") or "none",
        "CaseJointWeight": f"{C['weight']:.4f}", "CaseSingleWeight": f"{C.get('sum_single_weight', float('nan')):.4f}",
        "CaseLabelParents": ", ".join(sorted({G2.nodes[p].get("surrogate_of", p) for p in G2.predecessors(Y)}))
                            .replace("_", "\\_") or "none",
    }
    inject("case_numbers", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()))
    print(json.dumps({k: v for k, v in nums.items()}, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
