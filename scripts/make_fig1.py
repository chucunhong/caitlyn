"""Figure 1 (framework overview) as a PNG: one panel per stage with a toy plan, the LLM calls, the statistical
decisions with their thresholds, and what is passed to the next stage. Drawn in inches at twice the LLNCS
text width (12.2 cm), so fonts of 10-12 pt print at 5-6 pt.

  python scripts/make_fig1.py            -> paper/figures/fig1_overview.png
"""
import _common  # noqa: F401
from _common import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

C = dict(S="#d62728", P="#ff7f0e", R="#2ca02c", X="#9e9e9e", Y="#222222", llm="#e8f0fb", llm_e="#4a7cc4",
         test="#fdf3e1", test_e="#c99227", out="#eef7ee", out_e="#4f9a52", panel="#fbfbfb", gate="#1f77b4",
         lab="#2ca02c", prior="#d62728", A="#d6e6f6", Ae="#6fa8dc")
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
FW, FH = 9.6, 10.3
NR = 0.15                                                   # node radius (inches)


def box(ax, x, y, w, h, fc, ec, lw=1.0, r=0.06, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, ls=ls))


def text(ax, x, y, s, size=9.4, ha="left", va="top", color="#222222", weight="normal", z=6, **kw):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, weight=weight, zorder=z, linespacing=1.38, **kw)


def node(ax, x, y, lab, role, r=NR, hl=False, dashed=False):
    if dashed:
        ax.add_patch(Circle((x, y), r, fc="white", ec=C[role], lw=1.4, ls="--", zorder=4))
        ax.text(x, y, lab, fontsize=9.5, ha="center", va="center", color=C[role], weight="bold", zorder=5)
        return
    ax.add_patch(Circle((x, y), r, fc=C[role], ec="black" if hl else "white", lw=2.2 if hl else 1.4, zorder=4))
    ax.text(x, y, lab, fontsize=9.5, ha="center", va="center", color="white", weight="bold", zorder=5)


def arrow(ax, p, q, color="#4d4d4d", lw=1.3, ls="-", rad=0.0, z=3, head=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, ls=ls,
                                 shrinkA=12, shrinkB=12, connectionstyle=f"arc3,rad={rad}", zorder=z))


LAB = {"S1": "$S_1$", "S2": "$S_2$"}
POS = {"S1": (0.15, 0.85), "S2": (0.15, 0.2), "X": (0.95, 0.85), "R": (0.95, 0.2), "W": (1.75, 1.12),
       "Y": (2.15, 0.52), "Z": (1.75, -0.06)}


def toy(ax, ox, oy, edges, roles, hl=(), styles=None, skip=(), dashed=()):
    P = {k: (ox + x, oy + y) for k, (x, y) in POS.items()}
    for e in edges:
        arrow(ax, P[e[0]], P[e[1]], **(styles or {}).get(e, {}))
    for k, role in roles.items():
        if k not in skip:
            node(ax, *P[k], LAB.get(k, k), role, hl=k in hl, dashed=k in dashed)
    return P


def panel(ax, x, y, w, h, title):
    box(ax, x, y, w, h, C["panel"], "#b5b5b5", lw=1.1, r=0.08)
    text(ax, x + 0.1, y + h - 0.08, title, size=12.2, weight="bold")


def card(ax, x, y, w, h, kind, title, body):
    fc, ec = {"llm": (C["llm"], C["llm_e"]), "test": (C["test"], C["test_e"]), "out": (C["out"], C["out_e"])}[kind]
    box(ax, x, y, w, h, fc, ec, lw=1.1)
    text(ax, x + 0.08, y + h - 0.06, title, size=10.0, weight="bold", color=ec)
    text(ax, x + 0.08, y + h - 0.30, body, size=9.3)


def main():
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.set_aspect("equal")
    ax.set_axis_off()
    m, g = 0.06, 0.1
    pw = (FW - 2 * m - g) / 2
    strip_h = 1.25
    ph = (FH - 2 * m - strip_h - 2 * g) / 2
    y1 = m + strip_h + g + ph + g                               # top row
    y2 = m + strip_h + g                                        # bottom row
    xa, xb = m, m + pw + g
    roles = dict(S1="S", S2="S", X="P", R="R", Z="X", Y="Y", W="X")
    cw, ch_l, ch_t = pw - 0.2, 1.1, 1.1                          # card width and heights
    gy_off = 2.62                                               # toy plan origin (y) inside a panel

    # ---------------------------------------------------------------- Stage A
    panel(ax, xa, y1, pw, ph, "A   Role-typed plan induction")
    E = [("S1", "X"), ("S2", "X"), ("S1", "R"), ("X", "Y"), ("R", "Y"), ("S2", "Y"), ("Z", "Y"), ("X", "W"),
         ("S2", "R")]
    st = {("S2", "Y"): dict(color=C["prior"], ls="--", lw=1.7), ("Z", "Y"): dict(color=C["lab"], lw=2.0),
          ("S2", "R"): dict(color=C["gate"], lw=1.8, rad=0.35)}
    toy(ax, xa + 0.15, y1 + gy_off, E, roles, styles=st)
    nx_ = xa + 2.75
    text(ax, nx_, y1 + ph - 0.42, "LLM plan $G_0$, BFS from $\\mathcal{S}$", size=9.3, weight="bold", color="#444444")
    for i, (s, col) in enumerate([("—  edge kept", "#4d4d4d"), ("- -  prior-driven: rejected", C["prior"]),
                                  ("—  added by $\\mathcal{S}$-edge LRT", C["gate"]),
                                  ("—  added by label-parent LRT", C["lab"])]):
        text(ax, nx_, y1 + ph - 0.68 - i * 0.22, s, size=9.1, color=col)
    text(ax, nx_, y1 + ph - 1.6, "out: plan $G$ and\nresolving set $R$", size=9.1, color=C["out_e"],
         weight="bold")
    card(ax, xa + 0.1, y1 + 0.1 + ch_t + 0.08, cw, ch_l, "llm", "LLM calls",
         "$\\pi_{\\mathrm{role}}$: K = 5 samples (T = 0.7) of legitimate columns;\n"
         "    a column needs ≥ 3 votes to become a candidate\n"
         "$\\pi_{\\mathrm{edge}}$: BFS from $\\mathcal{S}$ (T = 0); children of $u$ given roles,\n"
         "    partial plan and association scores; $\\pi_{\\mathrm{cycle}}$ breaks cycles")
    card(ax, xa + 0.1, y1 + 0.1, cw, ch_t, "test", "Statistical decisions on real data",
         "$R$: AUC gain for $Y$ given the cell ≥ δ = 0.01 (within-cell\n"
         "    permutations, Holm) and proxy content PC ≤ τ$_p$ = 0.2\n"
         "gate: keep $S_i\\to v$ iff permutation $p_{\\mathrm{Holm}}$ < 0.05 and $a$ ≥ 0.05\n"
         "LRT adds missing $S_i\\to v$ and missing parents of $Y$ (Holm)")

    # ---------------------------------------------------------------- Stage B
    panel(ax, xb, y1, pw, ph, "B   Generator-side proxy audit")
    E = [("S1", "X"), ("S2", "X"), ("S1", "R"), ("X", "Y"), ("R", "Y"), ("Z", "Y"), ("X", "W"), ("S2", "R")]
    toy(ax, xb + 0.15, y1 + gy_off, E, roles, hl=("X",),
        styles={("X", "Y"): dict(color=C["P"], lw=2.4), ("X", "W"): dict(color=C["P"], lw=2.4)})
    bx, by, bw, bh = xb + 2.62, y1 + gy_off - 0.12, 1.98, 1.55
    box(ax, bx, by, bw, bh, "white", "#aaaaaa", r=0.05, z=2)
    text(ax, bx + 0.08, by + bh - 0.06, "call generating $Y$ for M = 100\nreal contexts, J = 5 draws", size=8.6)
    for k, (lab, v, col) in enumerate([("(a)  $X$ shown", 0.66, C["P"]), ("(b)  $X$ = UNKNOWN", 0.44, "#8c8c8c")]):
        yy = by + bh - 0.7 - k * 0.38
        text(ax, bx + 0.08, yy + 0.2, lab, size=8.6)
        ax.add_patch(Rectangle((bx + 0.1, yy - 0.08), 1.5 * v, 0.12, fc=col, zorder=5))
        ax.add_patch(Rectangle((bx + 0.1 + 1.5 * v, yy - 0.08), 1.5 * (1 - v), 0.12, fc="#e0e0e0", zorder=5))
    text(ax, bx + 0.08, by + 0.24, "PU$_{X\\to Y}$ = TV(a, b) − TV(a, a$'$)", size=8.4, color=C["P"],
         weight="bold")
    card(ax, xb + 0.1, y1 + 0.1 + ch_t + 0.08, cw, ch_l, "llm", "LLM calls ($\\pi_{\\mathrm{gen}}$, T = 0.9, 10 fixed exemplars)",
         "for every child $v$ of an audited node $u$ on an unresolved\n"
         "path: (a) $u$ shown, (b) $u$ = UNKNOWN, (a$'$) $u$ shown again;\n"
         "TV(a, a$'$) removes sampling noise\n"
         "cost: 3MJ/50 calls per audited edge")
    card(ax, xb + 0.1, y1 + 0.1, cw, ch_t, "test", "Scores and decision",
         "PC$_u$ = 2 max AUC($x_u \\to S_i$ or cell) − 1  (LightGBM, 5-fold)\n"
         "PRS$_u$ = PC$_u \\cdot$ max$_v$ PU$_{u\\to v}$;  proxies $P$ = {$u \\notin R$: PRS$_u$ > median}\n"
         "a column that encodes $\\mathcal{S}$ but is ignored by the\n"
         "generator gets a low PRS")

    # ---------------------------------------------------------------- Stage C
    panel(ax, xa, y2, pw, ph, "C   Joint path surgery")
    ox, oy = xa + 0.55, y2 + gy_off
    E = [("S1", "X"), ("S2", "X"), ("X", "W"), ("Z", "Y")]
    P = toy(ax, ox, oy, E, roles, dashed=("R",))
    arrow(ax, P["X"], P["Y"], color="#4d4d4d", lw=1.3)
    om = (xa + 0.2, oy + 0.65)
    node(ax, *om, "ω", "X")
    arrow(ax, om, P["S1"], color="#777777", lw=1.1)
    arrow(ax, om, P["S2"], color="#777777", lw=1.1)
    mx, my = (P["X"][0] + P["Y"][0]) / 2, (P["X"][1] + P["Y"][1]) / 2
    ax.plot([mx - 0.12, mx + 0.12], [my + 0.2, my - 0.2], color=C["S"], lw=3.2, zorder=7)
    text(ax, mx + 0.1, my + 0.34, "cut", size=9.6, color=C["S"], weight="bold")
    nx_ = xa + 3.0
    text(ax, nx_, y2 + ph - 0.5, "$H = G \\setminus R$; source ω → $S_i$\nwith capacity ∞", size=9.1,
         color="#444444")
    text(ax, nx_, y2 + ph - 1.08, "one minimum ω–$Y$ cut\nfor all attributes jointly", size=9.1,
         color="#444444")
    text(ax, nx_, y2 + ph - 1.66, "out: edited plan $G'$", size=9.1, color=C["out_e"], weight="bold")
    card(ax, xa + 0.1, y2 + 0.1 + ch_t + 0.08, cw, ch_l, "test", "Edge capacities (no LLM call)",
         "$w_{uv}$ = max(η, Δ$_{uv}$ − γ·1[$u \\in P$]),  η = 10$^{-3}$, γ = 0.01;\n"
         "Δ$_{uv}$: drop in cross-validated fit of $v$ without parent $u$\n"
         "edges cut into one node share its joint loss (iterative\n"
         "reweighting): redundant parents look cheap only one at a time")
    card(ax, xa + 0.1, y2 + 0.1, cw, ch_t, "out", "Operators and guarantee",
         "edge into $Y$, or interior edge with $w_{uv}$ ≤ median of the cut: cut;\n"
         "    else surrogate $u'$, generated with protected parents of $u$ hidden\n"
         "every path $S_i \\leadsto Y$ avoiding $R$ is blocked; the joint cut\n"
         "weighs at most the sum of per-attribute cuts (joint-cut proposition)")

    # ---------------------------------------------------------------- Stage D
    panel(ax, xb, y2, pw, ph, "D   Leakage-controlled synthesis")
    ox, oy = xb + 0.2, y2 + gy_off
    lay = {"S1": (0.1, 0.85), "S2": (0.1, 0.2), "X": (0.75, 0.85), "R": (0.75, 0.2), "W": (1.4, 1.18),
           "Z": (1.4, 0.68), "Y": (2.05, 0.6)}
    Pd = {k: (ox + a, oy + b) for k, (a, b) in lay.items()}
    for gx, g0, g1, inA in [(0.75, -0.04, 1.09, False), (1.4, 0.95, 1.41, False), (1.4, 0.46, 0.9, True),
                            (2.05, 0.38, 0.82, True)]:
        box(ax, ox + gx - 0.24, oy + g0, 0.48, g1 - g0, C["A"] if inA else "#ececec", C["Ae"] if inA else "#b5b5b5",
            lw=1.0, r=0.05, z=2)
    for e in [("S1", "X"), ("S2", "X"), ("S1", "R"), ("X", "W"), ("R", "Y"), ("Z", "Y"), ("S2", "R")]:
        arrow(ax, Pd[e[0]], Pd[e[1]])
    for k, role in roles.items():
        node(ax, *Pd[k], LAB.get(k, k), role)
    nx_ = xb + 2.72
    text(ax, nx_, y2 + ph - 0.5, "$G'$ by topological layer: one\ncall per layer, group and\nbatch of 50 rows",
         size=9.1, color="#444444")
    text(ax, nx_, y2 + ph - 1.22, "blue: nodes of $A$ (reach $Y$\navoiding $R$); their calls\nnever see $\\mathcal{S}$",
         size=9.1, color=C["Ae"])
    card(ax, xb + 0.1, y2 + 0.1 + ch_t + 0.08, cw, ch_l, "llm", "Generation calls ($\\pi_{\\mathrm{gen}}$, T = 0.9, top-$p$ 0.95)",
         "(i) context = parents in $G'$ only;  (ii) $A$ and non-$A$ never share a call\n"
         "(iii) exemplars show pa($v$) and $v$: per row, one of its k = 10\n"
         "     nearest training rows on pa($v$); batches mix cells\n"
         "(iv) roots from π$_\\lambda$(s) = (1−λ) $\\hat p$(s) + λ/|$\\mathcal{C}$|, λ = 0.3")
    card(ax, xb + 0.1, y2 + 0.1, cw, ch_t, "out", "Validation and novelty",
         "domain check with up to 3 retries per value\n"
         "copy of a training row or repeated row: regenerate the last $j$\n"
         "    groups outside an($Y$)∪{$Y$} (round $j$); replace rows still copied\n"
         "    by new rows of the same cell and label → exactly $m$ novel rows")

    # ---------------------------------------------------------------- bottom strip
    sx, sy, sw = m, m, FW - 2 * m
    box(ax, sx, sy, sw, strip_h, "white", "#b5b5b5", r=0.08)
    cols = [("Input", "$D_{\\mathrm{train}}$, label $Y$, protected\n$\\mathcal{S} = (S_1,\\dots,S_k)$, $k \\geq 2$,\n"
                      "cells $\\mathcal{C}$, LLM $f$, rows $m$"),
            ("Output", "$D_{\\mathrm{synth}}$ ($m$ novel rows) and\nthe logged plan, decisions\nand statistics of every stage"),
            ("Target: intersectional unresolved-effect parity",
             "$P(Y \\mid do(s, r)) = P(Y \\mid do(s', r))$ for all cells $s, s'$ and all $r$:\n"
             "exact when calls follow their parents in $G'$; otherwise the gap\n"
             "is at most Σ$_{v \\in A}\\,\\varepsilon_v$, the summed per-call leakage")]
    for (t, b), cx in zip(cols, [sx + 0.12, sx + 2.25, sx + 4.45]):
        text(ax, cx, sy + strip_h - 0.07, t, size=10.0, weight="bold")
        text(ax, cx, sy + strip_h - 0.33, b, size=9.1)
    lx = sx + 0.18
    items = [("S", "protected"), ("P", "proxy"), ("R", "resolving"), ("X", "other"), ("Y", "label")]
    for i, (k, lab) in enumerate(items):
        ax.add_patch(Circle((lx + i * 0.95, sy + 0.17), 0.07, fc=C[k], ec="white", zorder=5))
        text(ax, lx + 0.12 + i * 0.95, sy + 0.25, lab, size=8.8)
    for i, (kind, lab) in enumerate([("llm", "LLM call"), ("test", "statistical test"), ("out", "deterministic")]):
        fc, ec = {"llm": (C["llm"], C["llm_e"]), "test": (C["test"], C["test_e"]), "out": (C["out"], C["out_e"])}[kind]
        box(ax, sx + 5.05 + i * 1.45, sy + 0.1, 0.22, 0.15, fc, ec, r=0.02, z=5)
        text(ax, sx + 5.33 + i * 1.45, sy + 0.25, lab, size=8.8)
    out = ROOT / "paper" / "figures" / "fig1_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
