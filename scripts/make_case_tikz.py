"""Case-study figure as TikZ: one real run (all columns) in two panels, Stages A-B (plan, gate, roles, PC/PU)
and Stages C-D (joint cut, edited plan), written into the AUTO block `case_figure` of paper/main.tex. Layout by
role: protected | audited columns on unresolved paths | resolving columns | label, with columns that cannot reach
the label in a bottom row. Edges that decide a stage are coloured; the other edges are light grey. The picture is
drawn at twice the size (fonts near 10pt, as in the other figures) and scaled by 0.5.

  python scripts/make_case_tikz.py --dataset adult --method caitlyn__model-deepseek-v4.1-flash --seed 0
"""
import argparse
import json

import _common  # noqa: F401
from _common import inject
import networkx as nx

from caitlyn.config import RESULTS, load_dataset_meta
from caitlyn.pipeline import json2g

ABBR = {  # Adult
        "marital_status": "mar", "occupation": "occ", "relationship": "rel", "hours_per_week": "hrs",
        "native_country": "nat", "workclass": "work", "education": "edu", "education_num": "eduN",
        "capital_gain": "gain", "capital_loss": "loss",
        # German
        "checking_status": "chk", "credit_history": "hist", "employment": "emp", "housing": "hou",
        "other_installment_plans": "inst", "duration": "dur", "purpose": "purp", "credit_amount": "amt",
        "savings": "sav", "property": "prop", "existing_credits": "ncr", "foreign_worker": "frgn",
        "installment_rate": "rate", "num_dependents": "dep", "other_debtors": "debt", "residence_since": "resid",
        "telephone": "tel"}

STYLES = r"""[x=1cm,y=1cm,>={Stealth[length=3.2pt,width=3pt]},
  nd/.style={circle,draw,line width=0.5pt,minimum size=0.46cm,inner sep=0pt,font=\fontsize{4.8}{5}\selectfont},
  prot/.style={nd,draw=red!65!black,fill=red!15},
  res/.style={nd,rectangle,rounded corners=1.5pt,minimum width=0.46cm,draw=green!45!black,fill=green!15},
  oth/.style={nd,draw=black!45,fill=black!6},
  prx/.style={nd,draw=orange!80!black,fill=orange!25,line width=0.9pt},
  lab/.style={nd,draw=black,fill=black!75,text=white},
  faded/.style={nd,draw=black!20,fill=black!2,text=black!30},
  bg/.style={->,draw=black!18,line width=0.3pt},
  llm/.style={->,draw=black!40,line width=0.4pt},
  e/.style={->,draw=black!55,line width=0.45pt},
  gadd/.style={->,draw=blue!60!black,line width=0.7pt},
  add/.style={->,draw=green!55!black,line width=0.9pt},
  rej/.style={->,draw=red!70,dashed,line width=0.6pt},
  cutE/.style={->,draw=red!80!black,line width=1.1pt},
  sur/.style={->,draw=violet!70,dashed,line width=1.0pt},
  ttl/.style={font=\fontsize{6}{7}\selectfont,anchor=south west,inner sep=0pt},
  ann/.style={font=\fontsize{4.4}{4.6}\selectfont,text=black!85,inner sep=0.4pt,fill=white,rounded corners=0.5pt}]"""


TOP = 3.0  # height of the highest row of nodes
PANEL_DX, PANEL_DY = 6.1, 4.05  # offsets of the 2x2 panels


def double(tikz):
    """Twice every coordinate, dimension and font size (the picture is then scaled by 0.5)."""
    import re
    tikz = re.sub(r"\((-?\d*\.?\d+),(-?\d*\.?\d+)\)", lambda m: f"({2 * float(m[1]):.2f},{2 * float(m[2]):.2f})", tikz)
    tikz = re.sub(r"(?<![\w.=])(-?\d*\.?\d+)(pt|cm)|(?<=[^xy]=)(-?\d*\.?\d+)(pt|cm)",
                  lambda m: f"{2 * float(m[1] or m[3]):g}{m[2] or m[4]}", tikz)
    return re.sub(r"\\fontsize\{([\d.]+)\}\{([\d.]+)\}", lambda m: f"\\fontsize{{{2 * float(m[1]):g}}}{{{2 * float(m[2]):g}}}", tikz)


def layout(S, audited, R, others, G):
    """Role columns; order inside a column by the mean height of neighbours (barycentre sweeps)."""
    half = (len(audited) + 1) // 2
    cols = {0.0: list(S), 1.15: audited[:half], 2.0: audited[half:], 3.05: list(R)}
    top, bot = TOP, 0.3
    pos = {"Y": (4.25, (top + bot) / 2)}

    def place(x, names):
        n = len(names)
        for i, v in enumerate(names):
            pos[v] = (x, (top + bot) / 2 if n == 1 else top - i * (top - bot) / (n - 1))

    for x, names in cols.items():
        place(x, names)
    for _ in range(6):
        for x, names in cols.items():
            if len(names) < 2:
                continue
            def bary(v):
                nb = [pos[u][1] for u in list(G.predecessors(v)) + list(G.successors(v)) if u in pos]
                return -sum(nb) / len(nb) if nb else 0.0
            cols[x] = sorted(names, key=bary)
            place(x, cols[x])
    for i, v in enumerate(others):
        pos[v] = (0.6 + i * (3.4 / max(len(others) - 1, 1)), -0.55)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--method", default="caitlyn__model-deepseek-v4.1-flash")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = load_dataset_meta(a.dataset)
    S, Ylab = list(meta.protected), meta.label
    run = json.loads((RESULTS / a.dataset / a.method / str(a.seed) / "run.json").read_text(encoding="utf-8"))
    st = RESULTS / a.dataset / "_stages" / f"seed{a.seed}"
    A = json.loads((st / f"A_{run['stage_files']['A']}.json").read_text(encoding="utf-8"))
    B = json.loads((st / f"B_{run['stage_files']['B']}.json").read_text(encoding="utf-8"))
    C = json.loads((st / f"C_{run['stage_files']['C']}.json").read_text(encoding="utf-8"))
    rn = lambda v: "Y" if v == Ylab else v
    E = lambda lst: {(rn(u), rn(v)) for u, v in lst}
    G0, G, G2 = (nx.relabel_nodes(json2g(A["G0"]), rn), nx.relabel_nodes(json2g(A["G"]), rn),
                 nx.relabel_nodes(json2g(C["G2"]), rn))
    R, P, audited = A["R"], B["proxies"], list(B["audited"])
    prior = E(e["edge"] for e in A["prior_driven"])
    gate_add = E(e["edge"] for e in A["lrt_added"])
    lab_add = E(e["edge"] for e in A.get("label_parents_added", []))
    cut = E(C["cut"])
    ops = {tuple(rn(x) for x in k.split("->")): o for k, o in C["ops"].items()}
    others = sorted(v for v in G.nodes if v not in set(S) | set(R) | set(audited) | {"Y"})
    pos = layout(S, audited, R, others, G)
    sid = lambda v: v.replace("_", "")
    role = lambda v: "prot" if v in S else "lab" if v == "Y" else "res" if v in R else "oth"
    lab_of = lambda v: "$Y$" if v == "Y" else ABBR.get(v, v)

    H2 = G2.copy()
    H2.remove_nodes_from(R)
    Aset = nx.ancestors(H2, "Y") | {"Y"}
    hidden = [v for v, d in G2.nodes(data=True) if d.get("hidden")]

    out = ["\\begin{tikzpicture}" + STYLES]

    def nodes(p, style_of, extra=None):
        allpos = dict(pos, **(extra or {}))
        return [f"\\node[{style_of(v)}] ({p}{sid(v)}) at ({x:.2f},{y:.2f}) {{{lab_of(v)}}};"
                for v, (x, y) in allpos.items() if v in pos]

    def blocked(u, v, extra):
        """True if the straight segment u-v passes close to another node."""
        allp = dict(pos, **extra)
        if u not in allp or v not in allp:
            return False
        (x0, y0), (x1, y1) = allp[u], allp[v]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy or 1e-9
        for w, (x, y) in allp.items():
            if w in (u, v):
                continue
            t = ((x - x0) * dx + (y - y0) * dy) / L2
            if 0.08 < t < 0.92 and ((x0 + t * dx - x) ** 2 + (y0 + t * dy - y) ** 2) ** 0.5 < 0.27:
                return True
        return False

    def edges(p, es, style, extra=None):
        out_ = []
        for u, v in sorted(es):
            if u not in pos and u not in hidden:
                continue
            link = " to[bend left=16] " if blocked(u, v, extra or {}) else "--"
            out_.append(f"\\draw[{style}] ({p}{sid(u)}){link}({p}{sid(v)});")
        return out_

    def shade(xy):
        return f"\\fill[blue!10,rounded corners=2pt] ({xy[0] - 0.32:.2f},{xy[1] - 0.32:.2f}) rectangle ({xy[0] + 0.32:.2f},{xy[1] + 0.32:.2f});"

    H = G.copy()
    H.remove_nodes_from(R)
    fromS = set(S) | set().union(*[nx.descendants(H, s) for s in S if s in H])
    toY = nx.ancestors(H, "Y") | {"Y"}
    unres = {(u, v) for u, v in H.edges() if u in fromS and v in toY}
    titles = {"a": "(a) Stage A: LLM plan and evidence gate", "b": "(b) Stage B: proxy audit (PC/PU)",
              "c": "(c) Stage C: joint minimum cut on $G\\setminus R$", "d": "(d) Stage D: edited plan $G'$"}
    # each panel draws only the edges its stage acts on
    for p, dx, dy in [("a", 0.0, 0.0), ("b", PANEL_DX, 0.0), ("c", 0.0, -PANEL_DY), ("d", PANEL_DX, -PANEL_DY)]:
        out.append(f"\\begin{{scope}}[shift={{({dx},{dy})}}]")
        if p == "a":
            out += nodes(p, role)
            kept = set(G0.edges()) - prior
            out += edges(p, kept, "llm") + edges(p, prior, "rej") + edges(p, gate_add, "gadd") + edges(p, lab_add, "add")
        elif p == "b":
            out += nodes(p, lambda v: "faded" if v in R else ("prx" if v in P else role(v)))
            out += edges(p, unres, "e")
            for u, d in B["nodes"].items():
                if u in pos:
                    out.append(f"\\node[ann,above=-0.3pt of {p}{sid(u)}] {{{('%.2f' % d['pc']).lstrip('0')}/{('%.2f' % d['max_pu']).lstrip('0')}}};")
        elif p == "c":
            out += nodes(p, lambda v: "faded" if v in R else ("prx" if v in P else role(v)))
            out += edges(p, unres - cut, "llm")
            out += edges(p, {e for e in cut if ops.get(e) == "cut"}, "cutE")
            out += edges(p, {e for e in cut if ops.get(e) == "surrogate"}, "sur")
        else:
            sur_pos = {h: (pos[G2.nodes[h]["surrogate_of"]][0] + 0.55, pos[G2.nodes[h]["surrogate_of"]][1] + 0.35)
                       for h in hidden}
            out += [shade(pos[v]) for v in Aset if v in pos] + [shade(sur_pos[h]) for h in hidden if h in Aset]
            out += nodes(p, lambda v: "prx" if v in P else role(v))
            for h, (x, y) in sur_pos.items():
                lab = "\\widetilde{\\text{" + ABBR.get(G2.nodes[h]["surrogate_of"], G2.nodes[h]["surrogate_of"]) + "}}"
                out.append(f"\\node[nd,draw=violet!70,fill=violet!8,dashed] ({p}{sid(h)}) at ({x:.2f},{y:.2f}) {{${lab}$}};")
            into = {e for e in G2.edges() if e[1] == "Y"}
            out += edges(p, set(G2.edges()) - into, "bg", sur_pos)
            out += [s.replace("[e]", "[e,line width=0.85pt,draw=black!85]") for s in edges(p, into, "e", sur_pos)]
        out.append(f"\\node[ttl] at (-0.3,{TOP + 0.4:.2f}) {{{titles[p]}}};")
        out.append("\\end{scope}")

    items = [("prot", "protected"), ("res", "resolving"), ("prx", "proxy"), ("lab", "label")]
    lines = [("llm", "LLM edge"), ("e", "unresolved path"),("gadd", "added edge from $\\Sset$"), ("add", "added parent of $Y$")]
    if prior:
        lines.append(("rej", "rejected"))
    lines.append(("cutE", "cut (removed)"))
    if hidden:
        lines.append(("sur", "surrogate"))
    leg, x, yl = [], 0.0, 0.0
    for st_, txt in items:                                   # first line: node roles and shading
        leg.append(f"\\node[{st_},minimum size=0.22cm,minimum width=0.22cm] at ({x:.2f},{yl}) {{}}; "
                   f"\\node[anchor=west] at ({x + 0.13:.2f},{yl}) {{{txt}}};")
        x += 0.45 + 0.11 * len(txt)
    leg.append(f"\\fill[blue!10,rounded corners=1.5pt] ({x - 0.11:.2f},{yl - 0.1}) rectangle ({x + 0.11:.2f},{yl + 0.1}); "
               f"\\node[anchor=west] at ({x + 0.13:.2f},{yl}) {{column generated without $\\Sset$}};")
    x, yl = 0.0, -0.32
    for st_, txt in lines:                                   # second line: edges
        leg.append(f"\\draw[{st_}] ({x:.2f},{yl})--({x + 0.42:.2f},{yl}); "
                   f"\\node[anchor=west] at ({x + 0.44:.2f},{yl}) {{{txt}}};")
        x += 0.62 + 0.105 * len(txt.replace("$\\Sset$", "S").replace("$Y$", "Y"))
    out.append(f"\\begin{{scope}}[shift={{(0.0,{-PANEL_DY - (1.1 if others else 0.5)})}},font=\\fontsize{{5.5}}{{6}}\\selectfont,"
               "every node/.append style={inner sep=1pt}]\n" + "\n".join(leg) + "\n\\end{scope}")
    out.append("\\end{tikzpicture}")
    inject("case_figure", "\\scalebox{0.5}{%\n" + double("\n".join(out)) + "}")
    print("legend width ~", round(x + 2.6, 1), "cm | audited", audited, "| R", R, "| others", others, "| A", sorted(Aset))


if __name__ == "__main__":
    main()
