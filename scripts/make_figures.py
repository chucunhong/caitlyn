"""Generate the figures of the paper into paper/figures/*.png (included by paper/main.tex).

  python scripts/make_figures.py
Figures:
  fig_analysis_ab.pdf     (a) TSTR AUC vs. dSP_cap per method and seed; (b) leakage bound vs. twin flip rate
  fig_sensitivity.pdf     (c) sensitivity to n, lambda, k (AUC row and dSP_cap row; no dual axes)
  fig_tradeoff.pdf        fairness/utility trade-off per dataset
  fig_cd.pdf              Nemenyi critical-difference diagrams (utility, fairness)
  fig_plan_<dataset>.pdf  edited plan G' with roles, cut / surrogate and prior-driven edges (seed 0)
"""
import json

import _common  # noqa: F401
from _common import GROUP, METHODS, REAL_DATASETS, ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from caitlyn.config import RESULTS
from caitlyn.eval.stats import plot_cd
from caitlyn.pipeline import json2g
from collect import collect

OUT = ROOT / "paper" / "figures"
TEXTWIDTH = 4.8  # inches, LNCS text width (12.2 cm)

# Role colours = Figure 1 of the paper
ROLE = {"protected": ("#993C1D", "#FAECE7"), "proxy": ("#BA7517", "#FAEEDA"), "resolving": ("#0F6E56", "#E1F5EE"),
        "neutral": ("#888780", "#F1EFE8"), "label": ("#444441", "#FFFFFF")}
PRIOR = "#E24B4A"
# Method groups: fixed categorical order (validated reference palette) + marker as secondary encoding
GROUP_STYLE = {"Real": ("#52514e", "X"), "DGM": ("#2a78d6", "s"), "LLM": ("#eb6834", "^"), "Fair": ("#1baf7a", "D"),
               "Repair": ("#6250d6", "P"), "Ours": ("#e34948", "o")}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "pdf.fonttype": 42, "lines.linewidth": 2})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def plain_label(key):
    return dict((k, v) for k, v, _ in METHODS)[key].replace("\\method", "CAITLYN").replace("$D_{\\text{train}}$", "Real") \
        .replace("\\ ", " ")


def auc_dsp(T, datasets=REAL_DATASETS):
    """Per (method, seed): TSTR XGB AUC and dSP_cap averaged over datasets (German dSP pooled over seeds)."""
    U, X = T["utility"], T["fairness_extra"]
    u = U[(U["mode"] == "tstr") & (U.clf == "xgb") & U.dataset.isin(datasets)]
    x = X[(X.clf == "xgb") & X.dataset.isin(datasets)]
    rows = []
    for (m, s), g in u.groupby(["method", "seed"]):
        dsp = []
        for ds in g.dataset.unique():
            xs = x[(x.method == m) & (x.dataset == ds) & (x.seed.isin([s, "pooled"]))]["dSP"]
            if len(xs):
                dsp.append(xs.mean())
        rows.append(dict(method=m, seed=s, AUC=g["AUC"].mean(), dSP=np.mean(dsp) if dsp else np.nan,
                         n_ds=g.dataset.nunique()))
    return pd.DataFrame(rows)


def scatter_methods(ax, df, show_legend=True):
    keys = [k for k, _, _ in METHODS]
    seen = set()
    for key in keys:
        g = df[df.method == key]
        if g.empty:
            continue
        grp = GROUP[key]
        col, mk = GROUP_STYLE[grp]
        ours = key == "caitlyn"
        ax.scatter(g["dSP"], g["AUC"], s=26 if ours else 14, marker=mk, color=col, edgecolor="white", linewidth=0.6,
                   zorder=3 if ours else 2, label=grp if grp not in seen else None)
        seen.add(grp)
        mx, my = g["dSP"].mean(), g["AUC"].mean()
        if ours or key in ("real", "structsynth", "senna", "fairtabddpm"):
            ax.annotate(plain_label(key), (mx, my), xytext=(4, 3), textcoords="offset points", fontsize=6,
                        color=INK, fontweight="bold" if ours else "normal")
    ax.grid(True, zorder=0)
    ax.set_xlabel(r"$\Delta\mathrm{SP}_\cap$ (lower is better)")
    ax.set_ylabel("TSTR AUC")
    ax.margins(x=0.15, y=0.1)
    if show_legend and seen:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=min(len(seen), 3),
                  handletextpad=0.2, columnspacing=0.8)


def leakage_points():
    d = RESULTS / "_leakage"
    rows = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        for cond, e in r["eps"].items():
            tw = r["twins"].get(cond, {})
            rows.append(dict(dataset=r["dataset"], seed=r["seed"], cond=cond, eps=e["sum"], flip=tw.get("net_flip", np.nan),
                             label_flip=tw.get("net_label_flip", np.nan)))
    return pd.DataFrame(rows)


COND_LABEL = {"full_history": "full history", "parent_only": "+ parent-only", "parent_indep": "+ indep. exemplars",
              "caitlyn": "CAITLYN (all controls)"}
COND_COLOR = {"full_history": "#eb6834", "parent_only": "#eda100", "parent_indep": "#2a78d6", "caitlyn": "#e34948"}


def fig_analysis(T):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9))
    ax = axes[0]
    df = auc_dsp(T)
    if len(df):
        scatter_methods(ax, df, show_legend=False)
    else:
        ax.text(0.5, 0.5, "no results yet", ha="center", va="center", transform=ax.transAxes, color=MUTED)
    ax.set_title("(a) Fairness vs. utility")
    ax = axes[1]
    L = leakage_points()
    if len(L):
        for cond in COND_LABEL:
            g = L[L.cond == cond]
            if g.empty:
                continue
            ax.scatter(g["eps"], g["flip"], s=16, color=COND_COLOR[cond], edgecolor="white", linewidth=0.6,
                       label=COND_LABEL[cond], zorder=3)
        ok = L.dropna(subset=["eps", "flip"])
        if len(ok) >= 3:
            from scipy.stats import spearmanr
            rho = spearmanr(ok["eps"], ok["flip"]).correlation
            ax.text(0.03, 0.95, rf"Spearman $\rho$ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=6.5)
    else:
        ax.text(0.5, 0.5, "no results yet", ha="center", va="center", transform=ax.transAxes, color=MUTED)
    ax.grid(True, zorder=0)
    ax.set_xlabel(r"$\sum_{v\in A}\widehat{\varepsilon}_v$")
    ax.set_ylabel("Twin flip rate (net of null)")
    ax.set_title("(b) Leakage vs. observed")
    fig.tight_layout(w_pad=1.5)
    fig.subplots_adjust(bottom=0.36)
    for ax, ncol in zip(axes, (3, 2)):            # legends below each panel, clear of the axis labels
        h, l = ax.get_legend_handles_labels()
        if h:
            box = ax.get_position()
            fig.legend(h, l, frameon=False, loc="upper center", ncol=ncol, handletextpad=0.2, columnspacing=0.8,
                       bbox_to_anchor=(box.x0 + box.width / 2, 0.08))
    save(fig, "fig_analysis_ab.pdf")


def _sens_rows(T, dataset, prefix, values, default):
    U, X = T["utility"], T["fairness_extra"]
    rows = []
    for v in values:
        m = "caitlyn" if v == default else f"caitlyn__{prefix}{v}"
        auc = U[(U.method == m) & (U.dataset == dataset) & (U["mode"] == "tstr") & (U.clf == "xgb")]["AUC"]
        dsp = X[(X.method == m) & (X.dataset == dataset) & (X.clf == "xgb")]["dSP"]
        rows.append(dict(v=v, auc=auc.mean(), auc_sd=auc.std(), dsp=dsp.mean(), dsp_sd=dsp.std()))
    return pd.DataFrame(rows)


def fig_sensitivity(T, dataset="adult"):
    specs = [("n_train", [100, 500, 2000, "full"], "full", "training size $n$"),
             ("lam", [0.0, 0.3, 0.6, 1.0], 0.3, r"quota $\lambda$"),
             ("k", [1, 2, 3], 3, "protected attributes $k$")]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 2.6), sharex="col")
    for j, (prefix, vals, default, xl) in enumerate(specs):
        d = _sens_rows(T, dataset, prefix, vals, default)
        x = np.arange(len(vals))
        for i, (col, sd, yl, c) in enumerate([("auc", "auc_sd", "TSTR AUC", "#2a78d6"),
                                              ("dsp", "dsp_sd", r"$\Delta\mathrm{SP}_\cap$", "#e34948")]):
            ax = axes[i, j]
            ax.errorbar(x, d[col], yerr=d[sd].fillna(0), color=c, marker="o", ms=4, lw=2, capsize=2, elinewidth=0.8)
            ax.grid(True, axis="y")
            if j == 0:
                ax.set_ylabel(yl)
            if i == 1:
                ax.set_xticks(x, [str(v) for v in vals])
                ax.set_xlabel(xl)
            if d[col].isna().all():
                ax.text(0.5, 0.5, "no results yet", ha="center", va="center", transform=ax.transAxes, color=MUTED,
                        fontsize=6)
    fig.suptitle(f"(c) Sensitivity on {dataset.replace('_', ' ').title()}", fontsize=7.5, y=1.0)
    fig.tight_layout(h_pad=0.4, w_pad=1.0)
    save(fig, "fig_sensitivity.pdf")


def fig_tradeoff(T):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.8), sharey=False)
    for ax, ds in zip(axes, REAL_DATASETS):
        df = auc_dsp(T, [ds])
        if len(df):
            scatter_methods(ax, df, show_legend=False)
        ax.set_title(ds.replace("_", " ").title())
        if ds != "adult":
            ax.set_ylabel("")
    h, l = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    if h:
        fig.legend(h, l, frameon=False, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_tradeoff.pdf")


def fig_cd():
    p = RESULTS / "_stats.json"
    if not p.exists():
        print("skip fig_cd: run make_tables.py after evaluation")
        return
    st = json.loads(p.read_text(encoding="utf-8"))
    panels = [(k, t) for k, t in (("utility_auc", "TSTR AUC (higher is better)"),
                                  ("fair_sp_c", r"SP over combinations (lower is better)")) if k in st]
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(TEXTWIDTH, 1.7 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (k, title) in zip(axes, panels):
        ar = {plain_label(m) if m in dict((a, b) for a, b, _ in METHODS) else m: v for m, v in st[k]["avg_rank"].items()}
        plot_cd(ar, st[k]["cd"], ax, f"{title}; Friedman p = {st[k]['friedman_p']:.1e}", highlight="CAITLYN")
    fig.tight_layout()
    save(fig, "fig_cd.pdf")


def fig_plan(dataset, seed=0):
    p = RESULTS / dataset / "caitlyn" / str(seed) / "plan_edited.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    G, G2 = json2g(d["G"]), json2g(d["G2"])
    S, Y, R, P = d["protected"], d["label"], set(d["R"]), set(d["P"])
    cut = {tuple(e) for e in d["cut"]}
    ops = d["ops"]
    prior = [tuple(e["edge"]) for e in d.get("prior_driven", [])]

    def role(v):
        return "protected" if v in S else "label" if v == Y else "resolving" if v in R else "proxy" if v in P else "neutral"

    H = G.copy()
    H.add_edges_from(prior)                           # rejected edges are drawn for reference only
    try:
        layers = list(nx.topological_generations(H))
    except nx.NetworkXUnfeasible:
        layers = list(nx.topological_generations(G))
    pos = {}
    for i, layer in enumerate(layers):
        layer = sorted(layer, key=lambda v: (role(v), v))
        for j, v in enumerate(layer):
            pos[v] = (j - (len(layer) - 1) / 2, -i)
    width = max(len(l) for l in layers)
    fig, ax = plt.subplots(figsize=(min(TEXTWIDTH * 1.4, 0.9 * width + 1.5), 0.75 * len(layers) + 0.6))
    ax.axis("off")

    def edge(u, v, **kw):
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", shrinkA=11, shrinkB=11, mutation_scale=7, **kw))

    for u, v in G.edges():
        if (u, v) in cut:
            op = ops.get(f"{u}->{v}", "cut")
            edge(u, v, color="#b4b2a9", lw=0.8, linestyle=(0, (2, 2)) if op == "cut" else (0, (1, 1)))
        else:
            edge(u, v, color="#6f6e69", lw=0.8)
    for u, v in prior:
        if u in pos and v in pos:
            edge(u, v, color=PRIOR, lw=0.9, linestyle=(0, (3, 1.5)))
    for v, (x, y) in pos.items():
        ec, fc = ROLE[role(v)]
        ax.scatter([x], [y], s=520, color=fc, edgecolor=ec, linewidth=0.9, zorder=3)
        ax.text(x, y, v.replace("_", "\n") if len(v) > 9 else v, ha="center", va="center", fontsize=5.2,
                color=ec if role(v) != "neutral" else INK, zorder=4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=ROLE[r][1], mec=ROLE[r][0], label=r)
               for r in ("protected", "proxy", "resolving", "neutral")]
    handles += [Line2D([], [], color=PRIOR, lw=0.9, ls=(0, (3, 1.5)), label="prior-driven (rejected)"),
                Line2D([], [], color="#b4b2a9", lw=0.8, ls=(0, (2, 2)), label="cut"),
                Line2D([], [], color="#b4b2a9", lw=0.8, ls=(0, (1, 1)), label="surrogate")]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6)
    save(fig, f"fig_plan_{dataset}.pdf")


if __name__ == "__main__":
    T = collect()
    for k in ("utility", "fairness_extra"):
        if T[k].empty:
            T[k] = pd.DataFrame(columns=["dataset", "method", "seed", "mode", "clf", "AUC", "dSP"])
    fig_analysis(T)
    fig_sensitivity(T)
    fig_tradeoff(T)
    fig_cd()
    for ds in REAL_DATASETS:
        fig_plan(ds)
