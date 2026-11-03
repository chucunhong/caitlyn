"""Generate LaTeX tables for the paper and the supplement from the results directory.

  python scripts/make_tables.py
Writes paper/generated/*.tex. main.tex / supplementary.tex include them with \\IfFileExists, so the
paper compiles with placeholder tables until a generated file exists. Missing numbers print as \\TBD.
"""
import json

import _common  # noqa: F401
from _common import GROUP, LABEL, METHODS, REAL_DATASETS, ROOT, inject
import numpy as np
import pandas as pd

from caitlyn.config import RESULTS, load_dataset_meta, load_run_config
from caitlyn.eval.fairness import METRICS, settings, setting_name
from caitlyn.eval.stats import block_ranks, friedman_nemenyi, wilcoxon_holm
from collect import collect, runs_json

OUT = ROOT / "paper" / "generated"
DS_SHORT = {"adult": "Adult", "credit_scoring": "Credit", "german": "German"}
FAIR_COLS = ["SP", "EO", "EOd", "PP", "PE", "TE", "ABROCA", "epsDF"]


def f(x, d=3, bold=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "\\TBD"
    s = f"{x:.{d}f}"
    return f"\\textbf{{{s}}}" if bold else s


def write(name, text):
    """Tables and macros go into the matching AUTO block of paper/main.tex (single-file paper)."""
    inject(name.replace(".tex", ""), text)


def base(method):
    return method.split("__")[0]


def main_methods(df):
    """Rows of the main tables: plain method directories (no variant suffix)."""
    return df[df["method"].isin([k for k, _, _ in METHODS])]


# =============================================================== Table 2: utility, fidelity, privacy
def table_utility(T):
    U, D = main_methods(T["utility"]), main_methods(T["distribution"])
    rows, vals = [], {}
    for key, label, group in METHODS:
        u = U[(U.method == key)]
        v = {}
        for ds in REAL_DATASETS:
            v[ds] = u[(u.dataset == ds) & (u["mode"] == "tstr") & (u.clf == "xgb")]["AUC"].mean()
        aug = u[(u["mode"] == "aug") & (u.clf == "xgb")].groupby("dataset")["AUC"].mean()
        v["aug"] = aug.mean() if len(aug) else np.nan
        d = D[D.method == key].groupby("dataset").mean(numeric_only=True)
        for k in ("Shape", "Trend", "alpha_precision", "beta_recall", "copy_rate", "DCR_mean", "NN_risk"):
            v[k] = d[k].mean() if k in d and len(d) else np.nan
        vals[key] = v
    # average rank over dataset x classifier x seed blocks (TSTR AUC)
    tstr = U[U["mode"] == "tstr"]
    rank = {}
    if len(tstr):
        _, ranks = block_ranks(tstr, ["dataset", "clf", "seed"], "method", "AUC", True)
        rank = ranks.mean().to_dict()
    synth_keys = [k for k, _, g in METHODS if g not in ("Real", "Repair")]
    best = {}
    for col, hib in [("adult", 1), ("credit_scoring", 1), ("german", 1), ("aug", 1), ("Shape", 1), ("Trend", 1),
                     ("alpha_precision", 1), ("beta_recall", 1), ("DCR_mean", 1)]:
        cand = {k: vals[k][col] for k in synth_keys if np.isfinite(vals[k][col])}
        best[col] = max(cand, key=cand.get) if cand else None
    cand = {k: vals[k]["copy_rate"] for k in synth_keys if np.isfinite(vals[k]["copy_rate"])}
    best["copy_rate"] = min(cand, key=cand.get) if cand else None
    cand = {k: abs(vals[k]["NN_risk"] - 0.5) for k in synth_keys if np.isfinite(vals[k]["NN_risk"])}
    best["NN_risk"] = min(cand, key=cand.get) if cand else None
    cand = {k: rank[k] for k in synth_keys if k in rank}
    best["rank"] = min(cand, key=cand.get) if cand else None
    prev = None
    body = []
    for key, label, group in METHODS:
        v = vals[key]
        if prev is not None and group != prev:
            body.append("\\midrule")
        nd = key in ("real", "reweigh")
        cells = [f(v[ds], 3, best[ds] == key) for ds in REAL_DATASETS]
        cells += ["--"] if nd else [f(v["aug"], 3, best["aug"] == key)]
        for k, d in (("Shape", 3), ("Trend", 3), ("alpha_precision", 3), ("beta_recall", 3), ("copy_rate", 2),
                     ("DCR_mean", 3), ("NN_risk", 2)):
            cells.append("--" if nd else f(v[k], d, best[k] == key))
        cells.append("--" if key == "real" else f(rank.get(key, np.nan), 2, best["rank"] == key))
        grp = group if group != prev else ""
        body.append(f"{grp} & {label} & " + " & ".join(cells) + " \\\\")
        prev = group
    write("tab_utility.tex", r"""\begin{table}[t]
\centering
\caption{Utility, fidelity and privacy (mean over seeds). TSTR AUC with XGBoost; Rank averaged over 4 classifiers. Best synthetic result in bold.}
\label{tab:utility}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccccc}
\toprule
& & \multicolumn{3}{c}{TSTR AUC$\uparrow$} & Aug.\ AUC$\uparrow$ & \multicolumn{4}{c}{Fidelity (avg.\ over datasets)} & \multicolumn{3}{c}{Privacy} & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-10}\cmidrule(lr){11-13}
Type & Method & Adult & Credit & German & avg. & Shape$\uparrow$ & Trend$\uparrow$ & $\alpha$-P$\uparrow$ & $\beta$-R$\uparrow$ & Copy$\downarrow$ & DCR$\uparrow$ & NN-risk & Rank$\downarrow$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # per-classifier supplement table with std
    sup = []
    for clf in load_run_config()["evaluation"]["classifiers"]:
        for key, label, _ in METHODS:
            u = U[(U.method == key) & (U["mode"] == "tstr") & (U.clf == clf)]
            cells = []
            for ds in REAL_DATASETS:
                x = u[u.dataset == ds]
                for m in ("AUC", "BA", "F1"):
                    cells.append(f"{f(x[m].mean())}$\\pm${f(x[m].std(), 3)}" if len(x) else "\\TBD")
            sup.append(f"{clf.upper()} & {label} & " + " & ".join(cells) + " \\\\")
        sup.append("\\midrule")
    write("supp_utility_per_clf.tex", r"""\begin{table}[h]
\centering
\caption{TSTR utility per classifier (mean $\pm$ std over seeds).}
\scriptsize\setlength{\tabcolsep}{2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
& & \multicolumn{3}{c}{Adult} & \multicolumn{3}{c}{Credit scoring} & \multicolumn{3}{c}{German} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Clf & Method & AUC & BA & F1 & AUC & BA & F1 & AUC & BA & F1 \\
\midrule
""" + "\n".join(sup[:-1]) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    return vals, rank


# =============================================================== Table 3: fairness S/C
def fairness_sc(F, X, clf="xgb", datasets=REAL_DATASETS):
    """{method: {metric: (S, C)}} averaged within dataset over seeds, then over datasets."""
    out = {}
    F = F[(F.clf == clf) & (F.dataset.isin(datasets))]
    for key in F.method.unique():
        g = F[F.method == key]
        res = {}
        for m in FAIR_COLS:
            per_ds = g[g.metric == m].groupby(["dataset", "kind"])["value"].mean().unstack()
            res[m] = tuple(per_ds[k].mean() if k in per_ds else np.nan for k in ("S", "C"))
        x = X[(X.method == key) & (X.clf == clf) & (X.dataset.isin(datasets))]
        res["CSP_R"] = x.groupby("dataset")["CSP_R"].mean().mean() if len(x) else np.nan
        res["dSP"] = x.groupby("dataset")["dSP"].mean().mean() if len(x) else np.nan
        out[key] = res
    return out


def fairness_blocks(F, kind, metric, clf="xgb"):
    g = F[(F.clf == clf) & (F.kind == kind) & (F.metric == metric) & F.dataset.isin(REAL_DATASETS)]
    return g.groupby(["dataset", "seed", "method"])["value"].mean().reset_index()


def table_fair(T):
    F, X = main_methods(T["fairness"]), main_methods(T["fairness_extra"])
    sc = fairness_sc(F, X)
    dagger = {}
    for m in FAIR_COLS:
        for kind in ("S", "C"):
            blk = fairness_blocks(F, kind, m)
            if blk.empty or "caitlyn" not in blk.method.values:
                continue
            wide = blk.pivot_table(index=["dataset", "seed"], columns="method", values="value").dropna(axis=1, how="all")
            wide = wide.dropna()
            if len(wide) >= 3 and "caitlyn" in wide:
                for k, p in wilcoxon_holm(wide, "caitlyn", higher_is_better=False).items():
                    if p < 0.05:
                        dagger[(k, m, kind)] = True
    body = []
    for key, label, group in METHODS:
        if key not in sc and key != "caitlyn":
            sc_k = {m: (np.nan, np.nan) for m in FAIR_COLS} | {"CSP_R": np.nan}
        else:
            sc_k = sc.get(key, {m: (np.nan, np.nan) for m in FAIR_COLS} | {"CSP_R": np.nan})
        if key == "caitlyn":
            body.append("\\midrule")
        cells = []
        for m in FAIR_COLS:
            s, c = sc_k[m]
            ds_ = "$^{\\dagger}$" if dagger.get((key, m, "S")) else ""
            dc_ = "$^{\\dagger}$" if dagger.get((key, m, "C")) else ""
            cells.append(f"{f(s, 2)}{ds_}/{f(c, 2)}{dc_}")
        cells.append(f(sc_k["CSP_R"], 3))
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("tab_fair.tex", r"""\begin{table}[t]
\centering
\caption{Fairness of TSTR predictions on real test data (XGBoost, mean over datasets and seeds; lower is better). Each cell reports S/C, the average over single attributes (S) and over combinations (C); $^{\dagger}$ marks baselines that \method{} improves on significantly (Holm-corrected Wilcoxon, $p<0.05$).}
\label{tab:fair}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
Method & SP & EO & EOd & PP & PE & TE & ABROCA & $\varepsilon$-DF & CSP$_R$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    return sc


def table_breakdown(T):
    """Supplement: full breakdown per dataset x classifier (rows = methods, columns = settings)."""
    F = main_methods(T["fairness"])
    U = T["distribution"]
    out = []
    for ds in REAL_DATASETS:
        meta = load_dataset_meta(ds)
        sets = settings(meta.protected)
        heads = [setting_name(a).replace("*", "$\\times$").replace("_", "\\_") for a in sets]
        n1 = len(meta.protected)
        for clf in load_run_config()["evaluation"]["classifiers"]:
            g = F[(F.dataset == ds) & (F.clf == clf)]
            if g.empty:
                continue
            body = []
            for m in FAIR_COLS:
                body.append(f"\\multicolumn{{{len(sets) + 1}}}{{l}}{{\\textit{{{m}}}}} \\\\")
                for key, label, _ in METHODS:
                    gm = g[(g.method == key) & (g.metric == m)]
                    if gm.empty:
                        continue
                    cells = [f(gm[gm.setting == setting_name(a)]["value"].mean(), 3) for a in sets]
                    body.append(f"{label} & " + " & ".join(cells) + " \\\\")
            out.append(rf"""\begin{{table}}[h]
\centering
\caption{{Fairness breakdown, {ds.replace('_', ' ')}, {clf.upper()} (mean over seeds{'; test predictions pooled across seeds' if ds in load_run_config()['evaluation']['pool_seeds_for'] else ''}).}}
\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l{'c' * len(sets)}}}
\toprule
& \multicolumn{{{n1}}}{{c}}{{Single}} & \multicolumn{{{len(sets) - n1}}}{{c}}{{Combination}} \\
Method & {' & '.join(heads)} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    write("supp_breakdown.tex", "\n".join(out) if out else "% no results yet\n")


# =============================================================== Table 5: seesaw
def table_seesaw(T):
    F = T["fairness"]
    X = T["fairness_extra"]
    rows = [("None", "caitlyn__none"), ("Per-attr., sex first", "caitlyn__perattr_order-sex"),
            ("Per-attr., age first", "caitlyn__perattr_order-age"), ("Joint (\\method)", "caitlyn")]
    body = []
    for label, prefix in rows:
        cells = []
        for col in ("sex", "age", "3rd", "dsp"):
            pair = []
            for ds in ("adult", "credit_scoring"):
                meta = load_dataset_meta(ds)
                g = F[(F.dataset == ds) & (F.clf == "xgb") & (F.metric == "SP")]
                g = g[g.method == prefix] if prefix in ("caitlyn", "caitlyn__none") else g[g.method.str.startswith(prefix)]
                if col == "dsp":
                    pair.append(g[g.setting == setting_name(meta.protected)]["value"].mean())
                else:
                    third = [s for s in meta.protected if s not in ("sex", "age")]
                    if col == "3rd" and not third:     # Credit scoring protects sex and age only
                        pair.append(None)
                        continue
                    attr = third[0] if col == "3rd" else col
                    pair.append(g[g.setting == attr]["value"].mean())
            cells.append("/".join("--" if v is None else f(v, 3) for v in pair))
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("tab_seesaw_rows.tex", "\n".join(body) + "\n")


# =============================================================== Table 6: ablation
ABLATIONS = [("", "\\method{} (full)", "caitlyn"), ("A", "w/o evidence gate", "caitlyn__nogate"),
             ("A", "PC plan instead of LLM plan", "caitlyn__pcplan"), ("B", "w/o proxy audit ($\\gamma=0$)", "caitlyn__noproxy"),
             ("C", "per-attribute cuts", "caitlyn__perattr"), ("A", "$R$ from LLM only, no tests", "caitlyn__rllm"),
             ("C", "$R=\\emptyset$", "caitlyn__rempty"), ("D", "full-history context", "caitlyn__fullhist"),
             ("D", "no quotas ($\\lambda=0$)", "caitlyn__lam0")]


def table_ablation(T):
    U, D, X = T["utility"], T["distribution"], T["fairness_extra"]
    dss = ["adult", "credit_scoring"]
    body = []
    for stg, label, method in ABLATIONS:
        auc = U[(U.method == method) & (U["mode"] == "tstr") & (U.clf == "xgb") & U.dataset.isin(dss)].groupby("dataset")["AUC"].mean().mean()
        dsp = X[(X.method == method) & (X.clf == "xgb") & X.dataset.isin(dss)].groupby("dataset")["dSP"].mean().mean()
        tr = D[(D.method == method) & D.dataset.isin(dss)].groupby("dataset")["Trend"].mean().mean() if len(D) else np.nan
        body.append(f"{stg} & {label} & {f(auc)} & {f(dsp)} & {f(tr)} \\\\")
    write("tab_ablation_rows.tex", "\n".join(body) + "\n")


# =============================================================== supplement: R selection, prior edges, cost
def table_rsel(J):
    body = []
    for ds in REAL_DATASETS:
        r = J[(J.dataset == ds) & (J.method == "caitlyn")] if len(J) else J
        if r is None or len(r) == 0:
            body.append(f"{ds.replace('_', ' ')} & \\TBD & \\TBD & \\TBD \\\\")
            continue
        n = len(r)
        cand = pd.Series([c for x in r["candidates"] for c in x]).value_counts()
        sel = pd.Series([c for x in r["R"] for c in x]).value_counts()
        rej = []
        for seed in r["seed"]:
            a = _stage_a(ds, seed)
            if a:
                rej += [f"{k} ({v.split(' (')[0]})" for k, v in a.get("R_rejected", {}).items()]
        rej = pd.Series(rej).value_counts()
        esc = lambda s: str(s).replace("_", "\\_")
        empty = "$\\emptyset$"
        c_txt = ", ".join(f"{esc(k)} ({v}/{n})" for k, v in cand.items())
        s_txt = ", ".join(f"{esc(k)} ({v}/{n})" for k, v in sel.items()) or empty
        r_txt = ", ".join(esc(k) for k in rej.index[:6])
        body.append(f"{ds.replace('_', ' ')} & {c_txt} & {r_txt} & {s_txt} \\\\")
    write("supp_rsel_rows.tex", "\n".join(body) + "\n")


def _stage_a(ds, seed):
    d = RESULTS / ds / "_stages" / f"seed{seed}"
    run = RESULTS / ds / "caitlyn" / str(seed) / "run.json"
    if not run.exists():
        return None
    sig = json.loads(run.read_text(encoding="utf-8"))["stage_files"]["A"]
    p = d / f"A_{sig}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table_prior_edges(J):
    body = []
    for ds in REAL_DATASETS:
        a = _stage_a(ds, 0)
        if not a:
            continue
        for e in a["prior_driven"]:
            rat = str(e.get("rationale", "")).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")[:140]
            body.append(f"{ds.replace('_', ' ')} & {e['edge'][0].replace('_', chr(92) + '_')}$\\to${e['edge'][1].replace('_', chr(92) + '_')} & "
                        f"{f(e.get('assoc'), 3)} & {rat} \\\\")
    write("supp_prior_edges.tex", r"""\begin{table}[h]
\centering
\caption{Prior-driven edges rejected by the evidence gate (seed 0), with the association score and the LLM's rationale.}
\scriptsize
\begin{tabular}{llcp{0.55\textwidth}}
\toprule
Dataset & Edge & $a(u,v)$ & LLM rationale \\
\midrule
""" + ("\n".join(body) or "\\TBD & & & \\\\") + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_cost(J):
    cfg = load_run_config()["llm"]
    pin, pout = cfg["price_in_per_m"], cfg["price_out_per_m"]
    body = []
    for method, label in (("caitlyn", "\\method"), ("structsynth", "StructSynth"), ("cllm", "CLLM")):
        r = J[(J.method == method) & J.dataset.isin(REAL_DATASETS)] if len(J) else J
        if r is None or len(r) == 0:
            body.append(f"{label} & \\TBD & \\TBD & \\TBD & \\TBD & \\TBD \\\\")
            continue
        stages = {}
        for _, x in r.iterrows():
            n_rows = next((v for v in (x.get("rows"), x.get("rows_requested")) if v is not None and v == v and v > 0), 1000)
            rows = n_rows / 1000.0
            for st, u in (x["usage"] or {}).items():
                key = st.split("_")[0] if method == "caitlyn" else "gen"
                s = stages.setdefault(key, np.zeros(4))
                s += np.array([u.get("calls", 0), u.get("tokens_in", 0), u.get("tokens_out", 0), u.get("seconds", 0)]) / rows / len(r)
        for st, v in sorted(stages.items()):
            usd = v[1] / 1e6 * pin + v[2] / 1e6 * pout
            body.append(f"{label} & {st} & {v[0]:.0f} & {v[1] / 1e3:.0f}k / {v[2] / 1e3:.0f}k & {usd:.3f} & {v[3] / 60:.1f} \\\\")
    write("supp_cost.tex", r"""\begin{table}[h]
\centering
\caption{Cost per 1{,}000 generated rows (mean over real datasets and seeds): LLM calls, input/output tokens, US\$ at the configured prices, and LLM time in minutes (uncached). Stages A--C are run once per dataset and seed and amortized over the table size.}
\scriptsize
\begin{tabular}{llcccc}
\toprule
Method & Stage & Calls & Tokens in / out & US\$ & Minutes \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_cells():
    body = []
    for ds in REAL_DATASETS:
        p = ROOT / "data" / "processed" / ds / "cells.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text(encoding="utf-8"))["cells"]["0"]
        for cell in sorted(c["test"]):
            n = c["test"][cell]
            body.append(f"{ds.replace('_', ' ')} & {cell.replace('|', ', ').replace('<', '$<$').replace('>', '$>$')} & "
                        f"{c['train'].get(cell, 0)} & {n}{' (excluded)' if n < 30 else ''} \\\\")
    write("supp_cells.tex", r"""\begin{table}[h]
\centering
\caption{Intersectional cell counts (seed 0). Cells with fewer than 30 test rows are excluded from group metrics.}
\scriptsize
\begin{tabular}{llrr}
\toprule
Dataset & Cell & Train & Test \\
\midrule
""" + ("\n".join(body) or "\\TBD & & & \\\\") + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def numbers(T, vals, sc):
    """Macros for headline numbers used in the text (\\input{generated/numbers.tex})."""
    lines = ["% generated by scripts/make_tables.py"]

    def mac(name, v, d=3):
        lines.append(f"\\newcommand{{\\{name}}}{{{f(v, d)}}}")
    c = sc.get("caitlyn", {})
    for m in ("SP", "EO", "epsDF"):
        mac(f"res{m.replace('eps', 'Eps')}CaitlynC", c.get(m, (np.nan, np.nan))[1], 3)
        base_c = [sc[k][m][1] for k in sc if k not in ("caitlyn", "real") and np.isfinite(sc[k][m][1])]
        mac(f"res{m.replace('eps', 'Eps')}BestBaseC", min(base_c) if base_c else np.nan, 3)
    for ds in REAL_DATASETS:
        mac(f"resAUC{DS_SHORT[ds]}", vals.get("caitlyn", {}).get(ds, np.nan), 3)
        mac(f"resAUC{DS_SHORT[ds]}Real", vals.get("real", {}).get(ds, np.nan), 3)
    write("numbers.tex", "\n".join(lines) + "\n")


def stats_report(T):
    """Friedman/Nemenyi on TSTR AUC and on combination SP; stored as JSON for the CD figures."""
    U = main_methods(T["utility"])
    out = {}
    t = U[U["mode"] == "tstr"]
    if len(t):
        wide, ranks = block_ranks(t, ["dataset", "clf", "seed"], "method", "AUC", True)
        if wide.shape[1] >= 2 and len(wide):
            out["utility_auc"] = friedman_nemenyi(wide, ranks)
    F = main_methods(T["fairness"])
    g = F[(F.kind == "C") & (F.metric == "SP")]
    if len(g):
        g = g.groupby(["dataset", "clf", "seed", "method"])["value"].mean().reset_index()
        wide, ranks = block_ranks(g, ["dataset", "clf", "seed"], "method", "value", False)
        if wide.shape[1] >= 2 and len(wide):
            out["fair_sp_c"] = friedman_nemenyi(wide, ranks)
    (RESULTS / "_stats.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    T = collect()
    J = runs_json()
    if T["utility"].empty:
        print("No metrics.json found: run scripts/evaluate.py first. Writing tables with placeholders.")
        T = {k: pd.DataFrame(columns=["dataset", "method", "seed", "mode", "clf", "AUC", "metric", "kind", "setting",
                                      "value", "dSP", "CSP_R", "Trend"]) for k in T}
    vals, rank = table_utility(T)
    sc = table_fair(T)
    table_breakdown(T)
    table_seesaw(T)
    table_ablation(T)
    table_rsel(J)
    table_prior_edges(J)
    table_cost(J)
    table_cells()
    numbers(T, vals, sc)
    if not T["utility"].empty:
        print(json.dumps(stats_report(T), indent=1, default=float)[:2000])
