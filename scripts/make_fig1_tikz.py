"""Fig. 1 of the paper (pipeline overview on a toy plan) as TikZ, written into the AUTO block `fig1_overview`
of paper/main.tex. The same primitives draw an approximate preview with matplotlib in Computer Modern, which
is used to check that every text fits its box (no LaTeX is needed).

  python scripts/make_fig1_tikz.py --preview fig1.png
"""
import argparse
import math
import re

import _common  # noqa: F401
from _common import inject

BASE = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "orange": (1, .5, 0), "violet": (.5, 0, .5),
        "yellow": (1, 1, 0), "black": (0, 0, 0), "white": (1, 1, 1)}
# node style: draw, fill, line width, text colour
NODE = {"prot": ("red!65!black", "red!15", 0.5, "black"), "res": ("green!45!black", "green!15", 0.5, "black"),
        "oth": ("black!45", "black!6", 0.5, "black"), "prx": ("orange!80!black", "orange!25", 0.9, "black"),
        "lab": ("black", "black!75", 0.5, "white"), "faded": ("black!20", "black!2", 0.5, "black!30")}
# edge style: colour, line width, dashed
EDGE = {"e": ("black!55", 0.45, False), "eg": ("black!18", 0.4, False), "gadd": ("blue!60!black", 0.8, False),
        "add": ("green!55!black", 0.9, False), "rej": ("red!70", 0.7, True), "cutE": ("red!80!black", 1.1, False),
        "inf": ("black!80", 0.9, False), "flow": ("blue!65!black", 1.0, False), "pass": ("black!60", 0.8, False),
        "down": ("black!35", 1.6, False), "cutline": ("red!80!black", 0.8, True)}
TOY = {"s1": (0, 1), "s2": (0, 0), "x": (0.85, 1), "r": (0.85, 0), "y": (1.7, 0.5), "z": (1.7, 1.25)}
LBL = {"s1": "$S_1$", "s2": "$S_2$", "x": "$X$", "r": "$R$", "z": "$Z$", "y": "$Y$"}
ROLE = {"s1": "prot", "s2": "prot", "x": "oth", "r": "res", "z": "oth", "y": "lab"}
TXT = "black!85"
K = 2  # TikZ is drawn at K times the size (10pt fonts, whose shapes match the preview) and scaled to \textwidth


def rgb(expr):
    """xcolor expression such as red!65!black or blue!10 as an RGB triple."""
    t = expr.split("!")
    c, i = BASE[t[0]], 1
    while i < len(t):
        p, other = float(t[i]) / 100, BASE[t[i + 1]] if i + 1 < len(t) else BASE["white"]
        c = tuple(p * a + (1 - p) * b for a, b in zip(c, other))
        i += 2
    return c


def to_mpl(s):
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    return s.replace("\\textsc{unknown}", "UNKNOWN").replace("\\enspace", " ").replace("\\,", " ")


def textwidth(s, size):
    """Width in cm of a one-line text in Computer Modern."""
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    return TextPath((0, 0), to_mpl(s), size=size, prop=FontProperties(family="cmr10")).get_extents().width / 72.27 * 2.54


def pt(x, y):
    return f"({x * K:.3f},{y * K:.3f})"


HA = {"center": ("center", "center"), "west": ("left", "center"), "east": ("right", "center"),
      "north": ("center", "top"), "south": ("center", "bottom"), "north west": ("left", "top"),
      "south west": ("left", "bottom"), "south east": ("right", "bottom"), "north east": ("right", "top")}


class Canvas:
    def __init__(self, ax=None):
        self.t, self.ax, self.nodes, self.checks = [], ax, {}, []

    def rect(self, x0, y0, x1, y1, draw=None, fill=None, lw=0.5, r=0.06, dashed=False):
        o = [f"line width={lw * K:g}pt"] + ([f"rounded corners={r * 28.45 * K:.1f}pt"] if r else [])
        o += [f"draw={draw}"] if draw else []
        o += [f"fill={fill}"] if fill else []
        o += ["dashed"] if dashed else []
        self.t.append(f"\\path[{','.join(o)}] {pt(x0, y0)} rectangle {pt(x1, y1)};")
        if self.ax is not None:
            from matplotlib.patches import FancyBboxPatch
            self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                             fc=rgb(fill) if fill else "none", ec=rgb(draw) if draw else "none",
                                             lw=lw, ls="--" if dashed else "-", zorder=1))

    def text(self, x, y, lines, size=5.0, anchor="center", color=TXT, bold=False, maxw=None, fill=None):
        lines = lines if isinstance(lines, list) else [lines]
        font = f"\\fontsize{{{size * K:g}}}{{{size * K * 1.2:.1f}}}\\selectfont" + ("\\bfseries" if bold else "")
        o = f"anchor={anchor},inner sep={'2pt' if fill else '0pt'},font={font},text={color}"
        o += ",align=left" if len(lines) > 1 else ""
        o += f",fill={fill}" if fill else ""
        body = " \\\\ ".join(lines)
        self.t.append(f"\\node[{o}] at {pt(x, y)} {{{body}}};")
        if self.ax is not None:
            ha, va = HA[anchor]
            a = self.ax.text(x, y, "\n".join(to_mpl(s) for s in lines), fontsize=size, ha=ha, va=va,
                             color=rgb(color), weight="bold" if bold else "normal", linespacing=1.2, zorder=6,
                             multialignment="left",
                             bbox=dict(fc=rgb(fill), ec="none", pad=0.5) if fill else None)
            if maxw:
                self.checks.append((a, maxw, lines[0]))

    def node(self, nid, x, y, style, label, size=0.36, font=5.0, square=None):
        d, f, lw, tc = NODE[style]
        square = style == "res" if square is None else square
        extra = f",rectangle,rounded corners={1.2 * K:g}pt" if square and style != "res" else ""
        self.t.append(f"\\node[{style}{extra},minimum size={size * K:g}cm,font=\\fontsize{{{font * K:g}}}{{{font * K:g}}}"
                      f"\\selectfont] ({nid}) at {pt(x, y)} {{{label}}};")
        patch = None
        if self.ax is not None:
            from matplotlib.patches import Circle, FancyBboxPatch
            if square:
                patch = FancyBboxPatch((x - size / 2, y - size / 2), size, size, boxstyle="round,pad=0,rounding_size=0.04",
                                       fc=rgb(f), ec=rgb(d), lw=lw, zorder=4)
            else:
                patch = Circle((x, y), size / 2, fc=rgb(f), ec=rgb(d), lw=lw, zorder=4)
            self.ax.add_patch(patch)
            if label:
                self.ax.text(x, y, to_mpl(label), fontsize=font, ha="center", va="center", color=rgb(tc), zorder=5)
        self.nodes[nid] = (x, y, patch)

    def _arrow(self, pa, pb, style, rad=0.0, patchA=None, patchB=None, head=True):
        from matplotlib.patches import FancyArrowPatch
        c, lw, dashed = EDGE[style]
        self.ax.add_patch(FancyArrowPatch(pa, pb, patchA=patchA, patchB=patchB, shrinkA=0.5, shrinkB=0.5,
                                          arrowstyle=("-|>,head_length=0.7,head_width=0.4" if style == "down" else
                                                      "-|>,head_length=0.5,head_width=0.23") if head else "-",
                                          mutation_scale=6.5, color=rgb(c), lw=lw, ls="--" if dashed else "-",
                                          connectionstyle=f"arc3,rad={rad}", zorder=3))

    def edge(self, a, b, style, bend=0):
        link = "--" if not bend else f" to[bend {'left' if bend > 0 else 'right'}={abs(bend)}] "
        self.t.append(f"\\draw[{style}] ({a}){link}({b});")
        if self.ax is not None:
            (xa, ya, pa), (xb, yb, pb) = self.nodes[a], self.nodes[b]
            self._arrow((xa, ya), (xb, yb), style, -0.59 * math.sin(math.radians(bend)), pa, pb)

    def path(self, pts, style, head=True):
        self.t.append(f"\\draw[{style}{'' if head else ',-'}] " + " -- ".join(pt(x, y) for x, y in pts) + ";")
        if self.ax is not None:
            c, lw, dashed = EDGE[style]
            for p, q in zip(pts[:-2], pts[1:-1]):
                self.ax.plot([p[0], q[0]], [p[1], q[1]], color=rgb(c), lw=lw, ls="--" if dashed else "-", zorder=3,
                             solid_capstyle="butt")
            self._arrow(pts[-2], pts[-1], style, head=head)

    def fill(self, x0, y0, x1, y1, color, r=0.05):
        self.rect(x0, y0, x1, y1, fill=color, lw=0, r=r)

    def disc(self, x, y, rad, color):
        self.t.append(f"\\fill[{color}] {pt(x, y)} circle ({rad * K:g}cm);")
        if self.ax is not None:
            from matplotlib.patches import Circle
            self.ax.add_patch(Circle((x, y), rad, fc=rgb(color), ec="none", zorder=2))


def toy(c, p, ox, oy, s, styles=None, size=0.36, font=5.0, keys=TOY):
    styles = styles or {}
    for k in keys:
        x, y = TOY[k]
        c.node(p + k, ox + s * x, oy + s * y, styles.get(k, ROLE[k]), LBL[k], size, font, square=k == "r")


def toy_xy(k, ox, oy, s):
    return ox + s * TOY[k][0], oy + s * TOY[k][1]


def frame(c, x0, y0, x1, y1, col, title):
    c.rect(x0, y0, x1, y1, draw=col, fill=col.split("!")[0] + "!3", lw=0.8, r=0.12)
    c.text((x0 + x1) / 2, y1 - 0.2, title, size=7.2, bold=True, color=col.split("!")[0] + "!55!black")


def promptbox(c, x0, y0, x1, y1, tag, kind, lines, size=5.0):
    c.rect(x0, y0, x1, y1, draw="black!40", fill="white", lw=0.5, r=0.05)
    tw = 0.16 + textwidth(tag, size)
    c.rect(x1 - tw - 0.06, y1 - 0.1, x1 - 0.06, y1 + 0.1, draw="black!40",
           fill="yellow!35" if kind == "llm" else "blue!12", lw=0.4, r=0.04)
    c.text(x1 - 0.06 - tw / 2, y1, tag, size=size)
    for i, s in enumerate(lines):
        c.text(x0 + 0.07, y1 - 0.21 - i * 0.2, s, size=size, anchor="west", maxw=x1 - x0 - 0.12)


def chip(c, x0, y0, x1, y1, lines, size=5.0, fill="white"):
    c.rect(x0, y0, x1, y1, draw="black!35", fill=fill, lw=0.45, r=0.05)
    c.text((x0 + x1) / 2, (y0 + y1) / 2, lines, size=size, maxw=x1 - x0 - 0.06)


def bars(c, x, y, hs, color="black!45"):
    for i, h in enumerate(hs):
        c.fill(x + i * 0.13, y, x + i * 0.13 + 0.09, y + h, color, r=0)


def draw(c):
    # ------------------------------------------------------------------ A
    frame(c, 0.02, 3.5, 7.35, 6.85, "red!55", "A.\\enspace Role-Typed Plan Induction")
    CX = [1.25, 3.7, 6.1]
    boxes = [("$\\pi_{\\mathrm{role}}$, $K$ samples", "llm",
              ["\\textbf{In:} columns, task", "\\textbf{Out:} legitimate columns",
               "\\textbf{test:} gain $\\geq\\delta$, PC $\\leq\\tau_p$"]),
             ("$\\pi_{\\mathrm{edge}}$, $\\pi_{\\mathrm{cycle}}$", "llm",
              ["\\textbf{In:} graph, rows, $a(u,\\cdot)$", "\\textbf{Out:} children of $u$",
               "BFS from $\\mathbf{S}$, cycles broken"]),
             ("evidence gate", "test",
              ["$S_i\\to v$: permutation, Holm", "missed $S_i\\to v$: also LRT", "parents of $Y$: LRT"])]
    s, oy = 0.75, 4.05
    for i, (cx, (tag, kind, lines)) in enumerate(zip(CX, boxes)):
        promptbox(c, cx - 1.12, 5.62, cx + 1.12, 6.4, tag, kind, lines)
        c.path([(cx, 5.6), (cx, 5.2)], "down")
        ox, p = cx - 0.64, "a%d" % i
        toy(c, p, ox, oy, s)
        if i == 0:
            xr, yr = toy_xy("r", ox, oy, s)
            xx, yx = toy_xy("x", ox, oy, s)
            c.text(xr + 0.24, yr + 0.16, "$\\checkmark$", size=6.5, color="green!45!black")
            c.text(xx + 0.24, yx + 0.16, "$\\times$", size=6.5, color="red!75!black")
        else:
            for a, b in [("s1", "x"), ("s2", "x"), ("s1", "r"), ("x", "y"), ("r", "y")]:
                c.edge(p + a, p + b, "e")
            c.edge(p + "s2", p + "y", "e" if i == 1 else "rej", bend=20)
            if i == 2:
                c.edge(p + "s2", p + "r", "gadd")
                c.edge(p + "z", p + "y", "add")
        if i < 2:
            c.path([(cx + 0.9, oy + 0.5 * s), (CX[i + 1] - 0.9, oy + 0.5 * s)], "flow")
    for cx, cap in zip(CX, ["$R$: proposed and tested", "$G_0$: LLM plan", "$G$: tested plan"]):
        c.text(cx, 3.68, cap, size=5.0)
    # ------------------------------------------------------------------ B
    frame(c, 7.7, 3.5, 12.18, 6.85, "orange!75", "B.\\enspace Generator-Side Proxy Audit")
    for y, lab, hs in [(6.08, "$\\mathrm{pa}(v)$, $X{=}x$", [0.3, 0.12, 0.05]),
                       (5.5, "$X{=}$\\textsc{unknown}",[0.17, 0.15, 0.12])]:
        chip(c, 7.84, y - 0.15, 9.5, y + 0.15, lab)
        c.path([(9.52, y), (9.66, y)], "pass")
        chip(c, 9.68, y - 0.15, 10.28, y + 0.15, "$f(\\pi_{\\mathrm{gen}})$", fill="yellow!35")
        c.path([(10.3, y), (10.42, y)], "pass")
        bars(c, 10.46, y - 0.15, hs)
    c.path([(10.86, 6.2), (10.93, 6.2), (10.93, 5.38), (10.86, 5.38)], "pass", head=False)
    c.text(11.0, 5.79, ["TV $\\Rightarrow$", "$\\mathrm{PU}_{X\\to v}$"], size=5.5, anchor="west")
    chip(c, 7.84, 4.67, 9.5, 4.97, "classifier $X\\to\\mathbf{S}$")
    c.path([(9.52, 4.82), (9.66, 4.82)], "pass")
    c.text(9.72, 4.82, "$\\mathrm{PC}_X=2\\,\\mathrm{AUC}-1$", size=5.5, anchor="west")
    c.text(7.86, 4.2, "$\\mathrm{PRS}_X=\\mathrm{PC}_X\\cdot\\max_v\\mathrm{PU}_{X\\to v}$", size=5.5, anchor="west")
    c.text(7.86, 3.86, "$>$ median of audited columns", size=5.5, anchor="west")
    c.path([(10.55, 4.03), (10.98, 4.03)], "pass")
    c.node("bx", 11.2, 4.03, "prx", "$X$")
    c.text(11.42, 4.03, "proxy", size=5.5, anchor="west")
    # ------------------------------------------------------------------ C
    frame(c, 0.02, 0.0, 4.65, 3.3, "violet!60", "C.\\enspace Joint Path Surgery")
    s, ox, oy = 1.0, 0.95, 0.95
    toy(c, "c", ox, oy, s, {"x": "prx", "r": "faded"})
    c.node("cw", 0.38, 1.45, "lab", "", size=0.16)
    c.text(0.26, 1.45, "$\\omega$", size=6, anchor="east")
    c.edge("cw", "cs1", "inf")
    c.edge("cw", "cs2", "inf")
    for a, b, st in [("s1", "x", "e"), ("s2", "x", "e"), ("s1", "r", "eg"), ("s2", "r", "eg"), ("r", "y", "eg"),
                     ("z", "y", "e"), ("x", "y", "cutE")]:
        c.edge("c" + a, "c" + b, st)
    c.path([(2.06, 2.36), (2.42, 1.18)], "cutline", head=False)
    c.text(2.06, 2.42, "$C^\\star$", size=6, anchor="south", color="red!80!black")
    c.text(1.5, 0.5, "$H=G\\setminus R$ with source $\\omega$", size=5.0, maxw=2.9)
    c.text(1.5, 0.27, "per-attribute cuts pay $X\\to Y$ twice", size=5.0, maxw=2.9)
    c.rect(3.08, 1.2, 4.52, 2.75, draw="black!35", fill="white", lw=0.45, r=0.05)
    for i, s_ in enumerate(["$w_{uv}$: utility lost,", "$-\\gamma$ if $u\\in P$", "refined by $L(C)$",
                            "into $Y$: cut", "interior: cut or", "surrogate $u'$"]):
        c.text(3.15, 2.58 - i * 0.2 - (0.1 if i >= 3 else 0), s_, size=5.0, anchor="west", maxw=1.32)
    # ------------------------------------------------------------------ D
    frame(c, 5.0, 0.0, 12.18, 3.3, "blue!55", "D.\\enspace Leakage-Controlled Synthesis")
    promptbox(c, 5.12, 1.5, 7.3, 2.75, "$\\pi_{\\mathrm{gen}}$", "llm",
              ["\\textbf{In:} the parents in $G'$", "of the group and, per row,", "one of its 10 nearest",
               "training rows (parents,", "targets); \\textbf{Out:} its columns"])
    chip(c, 5.12, 1.0, 7.3, 1.3, "cells $s\\sim\\pi_\\lambda$, batches of 50")
    CXD, s, oy = [8.2, 9.71, 11.22], 0.55, 2.0
    order = ["s1", "s2", "z", "x", "r", "y"]
    steps = [({"s1", "s2", "z"}, set()), ({"x", "r"}, {"s1", "s2", "z"}), ({"y"}, {"s1", "s2", "z", "x", "r"})]
    gp = [("s1", "x"), ("s2", "x"), ("s1", "r"), ("s2", "r"), ("r", "y"), ("z", "y")]
    for i, (cx, (cur, done)) in enumerate(zip(CXD, steps)):
        ox, p = cx - 0.4675, "d%d" % i
        xz, yz = toy_xy("z", ox, oy, s)
        _, yy = toy_xy("y", ox, oy, s)
        c.fill(xz - 0.2, yy - 0.2, xz + 0.2, yz + 0.2, "blue!12")
        for k in cur:
            c.disc(*toy_xy(k, ox, oy, s), 0.21, "yellow!60")
        toy(c, p, ox, oy, s, {k: ("prx" if k == "x" else ROLE[k]) if k in cur | done else "faded" for k in TOY},
            size=0.3, font=4.5)
        for a, b in gp:
            c.edge(p + a, p + b, "e" if b in cur else "eg")
        cols = [k for k in order if k in cur | done]
        for j, k in enumerate(cols):
            x0 = cx - 0.57 + j * 0.19
            c.rect(x0, 1.6, x0 + 0.19, 1.77, draw="black!40", fill="yellow!45" if k in cur else NODE[ROLE[k]][1]
                   if k != "y" else "black!15", lw=0.35, r=0)
            c.rect(x0, 1.43, x0 + 0.19, 1.6, draw="black!40", fill="white", lw=0.35, r=0)
            c.text(x0 + 0.095, 1.685, LBL[k], size=4.0)
        c.text(cx, 1.3, "$V_%d$" % i, size=5.0)
        if i < 2:
            c.path([(cx + 0.64, 2.3), (CXD[i + 1] - 0.64, 2.3)], "flow")
    c.path([(7.32, 2.3), (7.52, 2.3)], "flow")
    c.path([(11.22, 1.22), (11.22, 1.08), (7.95, 1.08), (7.95, 0.92)], "pass")
    chip(c, 7.4, 0.3, 8.55, 0.9, ["copy of $D_{\\mathrm{train}}$", "or a repeat?"])
    c.path([(8.57, 0.6), (8.69, 0.6)], "pass")
    chip(c, 8.71, 0.3, 10.35, 0.9, ["regenerate columns", "outside $\\mathrm{an}(Y)\\cup\\{Y\\}$"])
    c.path([(10.37, 0.6), (10.49, 0.6)], "pass")
    chip(c, 10.51, 0.3, 11.58, 0.9, ["top up: same", "cell and label"])
    c.path([(11.6, 0.6), (11.7, 0.6)], "pass")
    for r_ in range(3):
        for j in range(3):
            c.rect(11.72 + j * 0.13, 0.75 - r_ * 0.13, 11.85 + j * 0.13, 0.88 - r_ * 0.13, draw="black!40",
                   fill="blue!25" if r_ == 0 else "white", lw=0.35, r=0)
    c.text(11.915, 0.26, "$D_{\\mathrm{synth}}$", size=5.0)
    # ------------------------------------------------------------------ outputs passed between stages
    c.path([(7.37, 5.0), (7.68, 5.0)], "pass")
    c.text(7.525, 5.07, "$G,R$", size=5.0, anchor="south")
    c.path([(9.95, 3.48), (9.95, 3.4), (2.3, 3.4), (2.3, 3.32)], "pass")
    c.text(6.0, 3.4, "$P$", size=5.0, fill="white")
    c.path([(4.67, 1.9), (4.98, 1.9)], "pass")
    c.text(4.825, 1.97, "$G'$", size=5.0, anchor="south")
    # ------------------------------------------------------------------ legend
    items = [("node", "prot", "protected"), ("node", "res", "resolving"), ("node", "prx", "proxy"),
             ("halo", None, "current call"), ("shade", None, "no $\\mathbf{S}$ seen"),
             ("edge", "rej", "rejected"), ("edge", "gadd", "added from $\\mathbf{S}$"),
             ("edge", "add", "added parent of $Y$"), ("edge", "cutE", "cut")]
    widths = [(0.52 if k == "edge" else 0.26) + textwidth(t, 5.0) for k, _, t in items]
    gap = 0.2
    x, y = 6.1 - (sum(widths) + gap * (len(items) - 1)) / 2, -0.28
    for (kind, st, t), w in zip(items, widths):
        if kind == "node":
            c.node("lg" + st, x + 0.1, y, st, "", size=0.2, square=st == "res")
            tx = x + 0.26
        elif kind == "halo":
            c.disc(x + 0.1, y, 0.1, "yellow!60")
            tx = x + 0.26
        elif kind == "shade":
            c.fill(x, y - 0.1, x + 0.2, y + 0.1, "blue!12", r=0.03)
            tx = x + 0.26
        else:
            c.path([(x, y), (x + 0.45, y)], st)
            tx = x + 0.52
        c.text(tx, y, t, size=5.0, anchor="west")
        x += w + gap
    print("legend width %.2f cm" % (sum(widths) + gap * (len(items) - 1)))


STYLE = (f"[x=1cm,y=1cm,>={{Stealth[length={3 * K:g}pt,width={2.8 * K:g}pt]}},\n"
         "  nd/.style={circle,draw,inner sep=0pt},\n"
         + "".join(f"  {k}/.style={{nd,draw={d},fill={f},line width={lw * K:g}pt,text={tc}"
                   f"{f',rectangle,rounded corners={1.2 * K:g}pt' if k == 'res' else ''}}},\n"
                   for k, (d, f, lw, tc) in NODE.items())
         + ",\n".join(f"  {k}/.style={{{'-{Stealth[length=%gpt,width=%gpt]}' % (4.5 * K, 5 * K) if k == 'down' else '->'},"
                      f"draw={col},line width={lw * K:g}pt"
                      f"{f',dash pattern=on {3 * K:g}pt off {3 * K:g}pt' if da else ''}}}"
                      for k, (col, lw, da) in EDGE.items()) + "]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="write an approximate PNG preview here")
    ap.add_argument("--no-inject", action="store_true")
    a = ap.parse_args()
    ax = fig = None
    if a.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                             "axes.unicode_minus": False})
        cm = 1 / 2.54
        fig = plt.figure(figsize=(12.4 * cm, 7.5 * cm))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-0.1, 12.3)
        ax.set_ylim(-0.55, 6.95)
        ax.set_aspect("equal")
        ax.axis("off")
    c = Canvas(ax)
    draw(c)
    if a.preview:
        fig.canvas.draw()
        inv = ax.transData.inverted()
        for art, maxw, s in c.checks:
            bb = art.get_window_extent()
            w = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
            if w > maxw:
                print(f"overflow {w:.2f} > {maxw:.2f} cm: {s}")
        fig.savefig(a.preview, dpi=300, facecolor="white")
        print("wrote", a.preview)
    if not a.no_inject:
        inject("fig1_overview", "\\resizebox{\\textwidth}{!}{%\n\\begin{tikzpicture}" + STYLE + "\n" + "\n".join(c.t)
               + "\n\\end{tikzpicture}}")


if __name__ == "__main__":
    main()
