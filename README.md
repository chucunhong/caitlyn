# CAITLYN: Multi-Attribute Fair Tabular Synthesis by Surgery on LLM Generation Plans

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Training-free](https://img.shields.io/badge/training-free-brightgreen.svg)](#overview)
[![LLM](https://img.shields.io/badge/LLM-qwen3.8--flash%20%7C%20deepseek--v4.1--flash-purple.svg)](#parameters)

> **CAITLYN** (**C**ross-**A**ttribute **I**ntersectional **T**opology for **L**LM-planned, **Y**ield-preserving, **N**on-discriminatory synthesis): a training-free framework that makes the generation plan of an LLM tabular generator fair for **several protected attributes jointly**.

🚀 [Quick Start](#quick-start) | 📊 [Datasets](#supported-datasets) | 🗂️ [Synthetic Datasets](#synthetic-datasets) | 📈 [Results](#results) | 🔬 [Full Pipeline](#full-pipeline)

---

## 📰 Abstract

LLM-based tabular generators increasingly produce training data. They inherit and amplify demographic bias, and fair generators protect one attribute at a time.

CAITLYN represents generation as an explicit **plan**: a DAG that fixes the order of LLM calls and what each call may see. It then makes this plan fair for several protected attributes jointly, in four steps:

1. It tests the LLM's plan against the data.
2. It detects legitimate (resolving) variables and proxies.
3. It blocks every illegitimate path to the label with **one minimum-weight cut**.
4. It restricts each LLM call to its parents.

We prove exact intersectional parity of unresolved effects when the generator follows the plan, and a leakage bound otherwise.

Results on three credit benchmarks with two LLM backbones:
- Among synthetic-data methods, CAITLYN has the lowest average statistical-parity gap over attribute combinations.
- It copies no training row.
- It stays within 0.06 to 0.15 TSTR AUC of real data on two of the benchmarks.

### Why several attributes at once?

Fairness rarely concerns a single attribute. A synthetic table can show equal positive rates for men and women and for younger and older people, yet still favour older men far more than young women.

Repairing one attribute at a time does not remove this gap, because attributes share proxies. Occupation, for example, depends on both sex and age, so a repair for one attribute leaves the influence of the other in place. Improving fairness for one protected attribute often lowers it for another.

Not every dependency is unfair either: *resolving* variables such as education may legitimately explain part of an outcome gap.

Fair synthesis with a black-box LLM and several protected attributes therefore raises three challenges:
1. Identifying legitimate dependencies without a given causal graph.
2. Finding the columns through which protected information actually reaches the generator, not only those that encode it.
3. Blocking the paths of all protected attributes at once, with as little loss of structure as possible.

---

## 🎯 Key Features

- 🧭 **Role-typed plan induction.** The LLM proposes the legitimate columns and the dependency graph. An evidence gate (Holm-corrected permutation tests and likelihood-ratio tests) keeps only what the data support, and logs stereotype-driven edges as *prior-driven*.
- 🔍 **Generator-side proxy audit.** A column is a proxy when it both *encodes* protected information (proxy capacity, PC) and is *used* by the generator (proxy use, PU). PU is measured by value ablation of the LLM's own calls.
- ✂️ **Joint path surgery.** One minimum `ω → Y` cut blocks all unresolved paths of all protected attributes at once. Capacities are the utility lost per edge, discounted for proxies and refined under the joint loss. The joint cut never costs more than per-attribute cuts.
- 🔒 **Leakage-controlled synthesis.** Each call sees only its parents in the edited plan. Columns that can reach the label are generated without protected attributes in view. Exemplars are matched on parents, small cells are oversampled, and copies of training rows are regenerated without resampling the label.
- 📐 **Guarantees.** A joint cut is never costlier than per-attribute cuts. The edited plan gives exact intersectional unresolved-effect parity (IUP) when the LLM follows it, and per-call leakage bounds the remaining effect otherwise.
- 📊 **Full evaluation.** Every single protected attribute and every combination is evaluated with SP, EO, EOd, PP, PE, TE, ABROCA, ε-DF and CSP<sub>R</sub>. Utility (TSTR and augmentation, four classifiers), fidelity (SDMetrics, α-precision, β-recall) and privacy (copy rate, DCR, NN risk) are also reported.
- 💸 **Cached LLM calls.** Every call is cached in SQLite, so reruns and variants that share stages cost nothing.

---

## 🎯 Overview

**Inputs:**
- a training table with a label `Y` and protected attributes `S = (S_1, …, S_k)` (e.g. sex, race and age);
- a one-sentence task description;
- black-box access to an LLM.

Without human input, CAITLYN selects a **resolving set** `R`: the columns through which protected attributes may legitimately affect `Y`. It then generates `m` rows that train accurate models for real data and satisfy the parity notion below.

**Generation plan.** A plan is a DAG over the columns in which every protected attribute is a root. The LLM generates a row column by column in topological order, and the call for column `v` sees only the values of its parents `pa(v)`.

A path from some `S_i` to `Y` is **unresolved** if none of its intermediate columns is in `R`. For example, sex → relationship → income is unresolved unless relationship is in `R`.

**Fairness target.** CAITLYN follows the idea of explainable discrimination:
- A gap between groups is acceptable when a legitimate qualification explains it, such as education for income.
- A gap that remains among equally qualified people is discrimination.

Demanding equal outcomes across cells regardless of `R`, as statistical parity does, would also erase the effect of education. That lowers utility and treats differently qualified people alike.

CAITLYN instead wants the generated label to depend on the cell only through `R`. Two rows with the same education but different sex and age should receive the same label distribution, while sex and age may still shape education itself. This is **intersectional unresolved-effect parity (IUP)**:

$$\Delta(s,s';r)=\mathrm{TV}\big(P(Y\mid do(\mathbf{S}{=}s,R{=}r)),\,P(Y\mid do(\mathbf{S}{=}s',R{=}r))\big)=0\quad\text{for all cells } s,s' \text{ and values } r.$$

IUP is imposed on every pair of intersectional cells (e.g. female, non-White, ≤30), not on each attribute separately. A gap that appears only at an intersection, such as for young women, therefore also violates it. For classifiers trained on the synthetic data, the observational counterpart of IUP is CSP<sub>R</sub>.

## 🏗️ Architecture

```
 D_train, task, protected S, label Y
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Stage A  Role-typed plan induction                  caitlyn/stage_a_plan │
 │   π_role × K  ──► candidate legitimate columns ──► R (relevance + PC ≤ τp)│
 │   π_edge BFS + π_cycle ──► plan G ──► evidence gate (add / reject edges)  │
 └──────────────────────────────────────────────────────────────────────────┘
        │  G, R
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Stage B  Generator-side proxy audit                caitlyn/stage_b_proxy │
 │   PC_u (encoded)  ×  max_v PU_{u→v} (used by the LLM)  = PRS_u ──► P      │
 └──────────────────────────────────────────────────────────────────────────┘
        │  G, R, P
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Stage C  Joint path surgery                      caitlyn/stage_c_surgery │
 │   H = G \ R + source ω → S_i ;  min-cut ω → Y under capacities w_uv       │
 │   cut edges: removed (into Y, cheap) or replaced by a surrogate ──► G'    │
 └──────────────────────────────────────────────────────────────────────────┘
        │  G'
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Stage D  Leakage-controlled synthesis           caitlyn/stage_d_generate │
 │   cells s ~ π_λ ; one call per topological layer and group (A / not A)    │
 │   parent-only context, parent-matched exemplars, novelty regeneration     │
 └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
     D_synth  (m distinct rows, none copied from D_train)
```

### Key quantities

**Proxy capacity.** How much a column `r` reveals about the protected attributes. `AUC_r` is the best cross-validated AUC for predicting one protected attribute, or the cell, from `r` alone.

$$\mathrm{PC}_r = 2\,\mathrm{AUC}_r - 1$$

**Proxy use.** How much the call that generates `v` changes when `u` is replaced by `UNKNOWN`. It is estimated on `M` real rows with `J` samples per row and condition, and sampling noise is subtracted.

$$\mathrm{PU}_{u\to v}=\mathrm{TV}\big(q_v(\cdot\mid x_{\mathrm{pa}(v)}),\;q_v(\cdot\mid x_{\mathrm{pa}(v)},u{=}\textsc{unknown})\big)$$

**Proxy risk.** A column `u` is a proxy if `PRS_u` exceeds the median over audited columns.

$$\mathrm{PRS}_u=\mathrm{PC}_u\cdot\max_v \mathrm{PU}_{u\to v}$$

**Edge capacity in the cut.** The utility lost by removing `u` from the parents of `v`. The proxy discount `γ` makes edges that leave a proxy cheaper, and the floor `η` keeps capacities positive.

$$w_{uv}=\max\big(\eta,\ \mathrm{perf}_v(\mathrm{pa}(v))-\mathrm{perf}_v(\mathrm{pa}(v)\setminus\{u\})-\gamma\,\mathbb{1}[u\in P]\big)$$

**Joint-loss refinement.** The cut is refined by iterative reweighting toward the smallest

$$L(C)=\sum_v\big[\mathrm{perf}_v(\mathrm{pa}(v))-\mathrm{perf}_v(\mathrm{pa}(v)\setminus C_v)\big].$$

**Cell sampling.** Cells are drawn from a mix of the empirical cell share and the uniform distribution, so that small cells are covered.

$$\pi_\lambda(s)=(1-\lambda)\,\hat p(s)+\lambda/|\mathcal{C}|$$

### Theoretical guarantees

Proofs are in the supplementary material of the paper.

- **A joint cut is never costlier.** Let `C_i` be the cheapest cut that blocks the unresolved paths of `S_i` alone, and `C*` the cheapest cut that blocks those of all protected attributes. Then `w(C*) ≤ Σ_i w(C_i)`. The inequality is strict when two of the `C_i` share an edge, e.g. an edge out of a proxy of two attributes.
- **Exact parity and leakage bound.** Let `A` be the columns from which `Y` can be reached in `G'` without passing through `R`.
  - The LLM is *faithful to the plan* if every call for a column in `A` depends on the protected attributes only through its parents in `G'`. A faithful LLM satisfies IUP.
  - Otherwise, let the leakage `ε_v` of a call be the largest change in its answer distribution when only the cell changes. The unresolved effect is then at most `Σ_{v∈A} ε_v`.
  - Both results concern the synthetic process defined by `G'`, not causal fairness in the real world.

---

## 📁 Project Structure

```
caitlyn/
├── caitlyn/                      # The method
│   ├── llm.py                    # OpenAI-compatible client (Alibaba Cloud Model Studio), SQLite cache, retry
│   ├── data.py                   # Loading, splits (70/30, stratified by label and cell), cells
│   ├── assoc.py                  # Associations (Cramér's V, correlation, correlation ratio), permutation tests, LRT
│   ├── stage_a_plan.py           # Stage A: roles, LLM-guided BFS, cycle breaking, evidence gate
│   ├── stage_b_proxy.py          # Stage B: proxy capacity, proxy use, proxy risk
│   ├── stage_c_surgery.py        # Stage C: capacities, joint min-cut, joint-loss refinement, surrogates
│   ├── stage_d_generate.py       # Stage D: layer-wise calls, matched exemplars, cell sampling, novelty
│   ├── pipeline.py               # Variants and stage caching (results/<ds>/_stages)
│   ├── prompts/*.txt             # π_role, π_edge, π_cycle, π_gen (+ CLLM / StructSynth prompts)
│   └── eval/                     # utility, fidelity, privacy, fairness, stats (Friedman/Nemenyi), leakage
├── configs/
│   ├── run.yaml                  # Every hyperparameter of the paper
│   └── datasets/*.yaml           # Label, protected attributes, task, column descriptions
├── scripts/
│   ├── prepare_data.py           # Preprocess Adult, German credit, Credit scoring
│   ├── run_caitlyn.py            # Run CAITLYN (and its variants)
│   ├── run_baseline.py           # Run or import baselines
│   ├── evaluate.py               # metrics.json + preds.csv for every run
│   ├── collect.py, summarize.py  # Tidy tables of all metrics
│   ├── make_tables.py, make_figures.py, make_case_study.py, make_fig1_tikz.py, make_case_tikz.py
│   ├── plan_oracle.py            # Oracle utility ceiling of a plan (no LLM calls)
│   ├── run_leakage.py            # Per-call leakage probes and twin flip rates
│   └── estimate_cost.py          # Calls / tokens / US$ before running
├── synthetic_dataset/            # ⭐ All synthetic tables used in the paper (see below)
├── requirements.txt
└── .env.example
```

Run outputs are written to `results/<dataset>/<method>/<seed>/` as `synth.csv`, `run.json`, `plan_edited.json`, `metrics.json` and `preds.csv`. Stage A–C artifacts go to `results/<dataset>/_stages/` and are shared across variants.

---

## 🚀 Quick Start

### Step 1: Install

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

cp .env.example .env              # then fill in your Alibaba Cloud Model Studio key
```

`configs/run.yaml → llm.provider` selects how the key is used:
- `token_plan`: subscription key and plan endpoint.
- `payg`: pay-as-you-go key and workspace endpoint.

Never commit `.env`.

### Step 2: Prepare the data

```bash
python scripts/prepare_data.py --dataset all
```

Adult and German credit are downloaded from UCI. For Credit scoring, place the Kaggle release (Davronov, 2021) at `data/raw/credit_scoring.csv`.

### Step 3: Smoke test without API calls

```bash
python scripts/run_caitlyn.py --dataset german --seeds 0 --mock --m 200 --set caitlyn.B=10 caitlyn.M=20 caitlyn.J=2
```

Mock outputs go to `*__mock` folders and are ignored by the table scripts unless `CAITLYN_INCLUDE_MOCK=1` is set.

### Step 4: Run CAITLYN

```bash
# main backbone (qwen3.8-flash), seeds 0 1 2
python scripts/run_caitlyn.py --dataset adult
python scripts/run_caitlyn.py --dataset credit_scoring
python scripts/run_caitlyn.py --dataset german

# second backbone (deepseek-v4.1-flash)
python scripts/run_caitlyn.py --dataset adult --model open

# stop after the plan (Stages A-C only)
python scripts/run_caitlyn.py --dataset german --stages ABC
```

**Output:** `results/<dataset>/caitlyn[__model-…]/<seed>/synth.csv`, together with the edited plan (`plan_edited.json`) and run statistics (`run.json`: R, proxies, cut, usage).

### Step 5: Run baselines

```bash
python scripts/run_baseline.py --dataset adult --method ctgan        # also: real reweigh cllm structsynth tabfairgan
python scripts/run_baseline.py --dataset adult --method export       # files for an external repo
python scripts/run_baseline.py --dataset adult --method import --name senna --seed 0 --file path/to/synth.csv
```

### Step 6: Evaluate and build tables

```bash
python scripts/evaluate.py                       # utility, fidelity, privacy, fairness for every run
python scripts/summarize.py --datasets adult credit_scoring german
python scripts/make_tables.py                    # LaTeX tables (paper/generated/*.tex)
```

---

## ⚙️ Parameters

### Full configuration used in the paper

These values come from [`configs/run.yaml`](configs/run.yaml). Any of them can be overridden at run time with `--set key=value`, e.g. `--set caitlyn.lam=0.6 llm.max_concurrency=3`.

**LLM**

| Component | Value | Key |
|-----------|-------|-----|
| Main backbone | `qwen3.8-flash` | `llm.model` |
| Second backbone | `deepseek-v4.1-flash` | `llm.open_model` (`--model open`) |
| Thinking | off (JSON mode on) | `llm.enable_thinking`, `llm.json_mode` |
| Temperature: π_role / π_edge, π_cycle / π_gen | 0.7 / 0 / 0.9 (top-p 0.95) | `caitlyn.role_temperature`, `struct_temperature`, `gen_temperature`, `gen_top_p` |
| Concurrency | 3 | `llm.max_concurrency` |
| Cache | `results/llm_cache.sqlite` | `llm.cache` |

**Stage A: plan induction**

| Component | Value | Key |
|-----------|-------|-----|
| π_role samples K / majority votes | 5 / 3 | `caitlyn.K`, `role_min_votes` |
| Rows shown to π_role / π_edge | 30 | `caitlyn.n_example_rows` |
| Relevance gain δ | 0.01 AUC | `caitlyn.delta` |
| Proxy-capacity bound τ<sub>p</sub> | 0.2 | `caitlyn.tau_p` |
| Significance α (Holm) | 0.05 | `caitlyn.alpha` |
| Permutations B | max(200, ⌈2m/α⌉) per test family | `caitlyn.B` |
| Association threshold τ for protected edges | 0.05 | `caitlyn.tau` |
| Label is a sink / LRT label-parent completion | yes / yes | `label_is_sink`, `label_parents` |

**Stage B: proxy audit**

| Component | Value | Key |
|-----------|-------|-----|
| Real rows M | 100 | `caitlyn.M` |
| Samples per row and condition J | 5 | `caitlyn.J` |
| Proxy rule | PRS<sub>u</sub> > median over audited columns | — |

**Stage C: surgery**

| Component | Value | Key |
|-----------|-------|-----|
| Capacity floor η | 1e-3 | `caitlyn.eta` |
| Proxy discount γ | 0.01 | `caitlyn.gamma` |
| Joint-loss refinement | on | `caitlyn.refine_cut` |
| Auxiliary predictors | LightGBM, fixed hyperparameters, stratified 5-fold CV | — |

**Stage D: generation**

| Component | Value | Key |
|-----------|-------|-----|
| Rows \|D<sub>synth</sub>\| (every method, every dataset) | 1,000 | `caitlyn.synth_rows` |
| Rows per call | 50 | `caitlyn.batch_size` |
| Grouping | one call per topological layer and group (A / not A) | `caitlyn.grouping` |
| Exemplars | 1 of the 10 nearest training rows on the parents (Gower), per batch row | `caitlyn.match_k` |
| Cell mixing λ | 0.3 | `caitlyn.lam` |
| Novelty rounds / top-ups | 4 / 5 | `caitlyn.novelty_rounds`, `max_topups` |

**Evaluation**

| Component | Value | Key |
|-----------|-------|-----|
| Seeds | 0, 1, 2 | `seeds` |
| Classifiers | XGBoost, random forest, logistic regression, MLP | `evaluation.classifiers` |
| Minimum group size | 30 test rows | `evaluation.min_group` |
| Pooled test predictions | German credit | `evaluation.pool_seeds_for` |

### `scripts/run_caitlyn.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | — | `adult`, `credit_scoring` or `german` |
| `--seeds` | `0 1 2` | Seeds to run |
| `--model` | `qwen3.8-flash` | `open` for `deepseek-v4.1-flash`, or any model id |
| `--stages` | `ABCD` | Stages to run (`ABC` stops before generation) |
| `--m` | 1,000 | Number of synthetic rows |
| `--variant` | `full` | Ablations: `nogate`, `pcplan`, `noproxy`, `perattr`, `rllm`, `rempty`, `fullhist`, `lam0`, `percol`, `leak_*` |
| `--order` | — | Attribute order for `--variant perattr` |
| `--lam`, `--gamma`, `--delta`, `--tau-p`, `--k`, `--n-train` | config | Sensitivity overrides |
| `--set` | — | Override any key of `run.yaml` |
| `--mock` | off | Offline fake LLM |

---

## 📊 Supported Datasets

| Dataset | Rows | Protected attributes | Cells | Label |
|---------|------|----------------------|-------|-------|
| Adult | 45,222 | sex, race (White / non-White), age (≤30, 31–50, >50) | 12 | income > 50K |
| Credit scoring | 8,755 | sex, age (≤36 / >36) | 4 | good / bad credit |
| German credit | 1,000 | sex, age (≤25 / >25) | 4 | good / bad credit risk |

Notes:
- **Splits.** All datasets are split 70/30, stratified by label and cell, over three seeds.
- **Credit scoring.** It has 11.58 positives per negative. Two columns almost determine its label (`Field`, `Changed-phone-number`), so they are generated but dropped before training classifiers.
- **German credit.** `personal_status` is replaced by `sex`.

---

## 🗂️ Synthetic Datasets

All synthetic tables evaluated in the paper are in [`synthetic_dataset/`](synthetic_dataset/):

```
synthetic_dataset/<dataset>/<method>/seed<k>.csv          # synthetic table (columns of data/processed/<dataset>)
synthetic_dataset/<dataset>/caitlyn_*/seed<k>_plan.json   # CAITLYN plan: G (before surgery), G2 (edited plan G'),
                                                          #   R, P (proxies), cut, ops (cut / surrogate), costs
```

Every table has 1,000 rows. CLLM keeps fewer rows after curation (650–929).

| Method | Adult | Credit scoring | German credit |
|--------|-------|----------------|---------------|
| **CAITLYN** (`qwen3.8-flash`) | [seed0](synthetic_dataset/adult/caitlyn_qwen3.8-flash/seed0.csv) · [seed1](synthetic_dataset/adult/caitlyn_qwen3.8-flash/seed1.csv) · [seed2](synthetic_dataset/adult/caitlyn_qwen3.8-flash/seed2.csv) | [seed0](synthetic_dataset/credit_scoring/caitlyn_qwen3.8-flash/seed0.csv) · [seed1](synthetic_dataset/credit_scoring/caitlyn_qwen3.8-flash/seed1.csv) · [seed2](synthetic_dataset/credit_scoring/caitlyn_qwen3.8-flash/seed2.csv) | [seed0](synthetic_dataset/german/caitlyn_qwen3.8-flash/seed0.csv) · [seed1](synthetic_dataset/german/caitlyn_qwen3.8-flash/seed1.csv) · [seed2](synthetic_dataset/german/caitlyn_qwen3.8-flash/seed2.csv) |
| ↳ plans | [0](synthetic_dataset/adult/caitlyn_qwen3.8-flash/seed0_plan.json) · [1](synthetic_dataset/adult/caitlyn_qwen3.8-flash/seed1_plan.json) · [2](synthetic_dataset/adult/caitlyn_qwen3.8-flash/seed2_plan.json) | [0](synthetic_dataset/credit_scoring/caitlyn_qwen3.8-flash/seed0_plan.json) · [1](synthetic_dataset/credit_scoring/caitlyn_qwen3.8-flash/seed1_plan.json) · [2](synthetic_dataset/credit_scoring/caitlyn_qwen3.8-flash/seed2_plan.json) | [0](synthetic_dataset/german/caitlyn_qwen3.8-flash/seed0_plan.json) · [1](synthetic_dataset/german/caitlyn_qwen3.8-flash/seed1_plan.json) · [2](synthetic_dataset/german/caitlyn_qwen3.8-flash/seed2_plan.json) |
| **CAITLYN** (`deepseek-v4.1-flash`) | [seed0](synthetic_dataset/adult/caitlyn_deepseek-v4.1-flash/seed0.csv) · [seed1](synthetic_dataset/adult/caitlyn_deepseek-v4.1-flash/seed1.csv) · [seed2](synthetic_dataset/adult/caitlyn_deepseek-v4.1-flash/seed2.csv) | [seed0](synthetic_dataset/credit_scoring/caitlyn_deepseek-v4.1-flash/seed0.csv) · [seed1](synthetic_dataset/credit_scoring/caitlyn_deepseek-v4.1-flash/seed1.csv) · [seed2](synthetic_dataset/credit_scoring/caitlyn_deepseek-v4.1-flash/seed2.csv) | [seed0](synthetic_dataset/german/caitlyn_deepseek-v4.1-flash/seed0.csv) · [seed1](synthetic_dataset/german/caitlyn_deepseek-v4.1-flash/seed1.csv) · [seed2](synthetic_dataset/german/caitlyn_deepseek-v4.1-flash/seed2.csv) |
| ↳ plans | [0](synthetic_dataset/adult/caitlyn_deepseek-v4.1-flash/seed0_plan.json) · [1](synthetic_dataset/adult/caitlyn_deepseek-v4.1-flash/seed1_plan.json) · [2](synthetic_dataset/adult/caitlyn_deepseek-v4.1-flash/seed2_plan.json) | [0](synthetic_dataset/credit_scoring/caitlyn_deepseek-v4.1-flash/seed0_plan.json) · [1](synthetic_dataset/credit_scoring/caitlyn_deepseek-v4.1-flash/seed1_plan.json) · [2](synthetic_dataset/credit_scoring/caitlyn_deepseek-v4.1-flash/seed2_plan.json) | [0](synthetic_dataset/german/caitlyn_deepseek-v4.1-flash/seed0_plan.json) · [1](synthetic_dataset/german/caitlyn_deepseek-v4.1-flash/seed1_plan.json) · [2](synthetic_dataset/german/caitlyn_deepseek-v4.1-flash/seed2_plan.json) |
| CTGAN | [seed0](synthetic_dataset/adult/ctgan/seed0.csv) · [seed1](synthetic_dataset/adult/ctgan/seed1.csv) · [seed2](synthetic_dataset/adult/ctgan/seed2.csv) | [seed0](synthetic_dataset/credit_scoring/ctgan/seed0.csv) · [seed1](synthetic_dataset/credit_scoring/ctgan/seed1.csv) · [seed2](synthetic_dataset/credit_scoring/ctgan/seed2.csv) | [seed0](synthetic_dataset/german/ctgan/seed0.csv) · [seed1](synthetic_dataset/german/ctgan/seed1.csv) · [seed2](synthetic_dataset/german/ctgan/seed2.csv) |
| CLLM | [seed0](synthetic_dataset/adult/cllm/seed0.csv) · [seed1](synthetic_dataset/adult/cllm/seed1.csv) · [seed2](synthetic_dataset/adult/cllm/seed2.csv) | [seed0](synthetic_dataset/credit_scoring/cllm/seed0.csv) · [seed1](synthetic_dataset/credit_scoring/cllm/seed1.csv) · [seed2](synthetic_dataset/credit_scoring/cllm/seed2.csv) | [seed0](synthetic_dataset/german/cllm/seed0.csv) · [seed1](synthetic_dataset/german/cllm/seed1.csv) · [seed2](synthetic_dataset/german/cllm/seed2.csv) |
| StructSynth | [seed0](synthetic_dataset/adult/structsynth/seed0.csv) · [seed1](synthetic_dataset/adult/structsynth/seed1.csv) · [seed2](synthetic_dataset/adult/structsynth/seed2.csv) | [seed0](synthetic_dataset/credit_scoring/structsynth/seed0.csv) · [seed1](synthetic_dataset/credit_scoring/structsynth/seed1.csv) · [seed2](synthetic_dataset/credit_scoring/structsynth/seed2.csv) | [seed0](synthetic_dataset/german/structsynth/seed0.csv) · [seed1](synthetic_dataset/german/structsynth/seed1.csv) · [seed2](synthetic_dataset/german/structsynth/seed2.csv) |
| TabFairGAN | [seed0](synthetic_dataset/adult/tabfairgan/seed0.csv) · [seed1](synthetic_dataset/adult/tabfairgan/seed1.csv) · [seed2](synthetic_dataset/adult/tabfairgan/seed2.csv) | [seed0](synthetic_dataset/credit_scoring/tabfairgan/seed0.csv) · [seed1](synthetic_dataset/credit_scoring/tabfairgan/seed1.csv) · [seed2](synthetic_dataset/credit_scoring/tabfairgan/seed2.csv) | [seed0](synthetic_dataset/german/tabfairgan/seed0.csv) · [seed1](synthetic_dataset/german/tabfairgan/seed1.csv) · [seed2](synthetic_dataset/german/tabfairgan/seed2.csv) |
| SENNA | [seed0](synthetic_dataset/adult/senna/seed0.csv) | [seed0](synthetic_dataset/credit_scoring/senna/seed0.csv) | [seed0](synthetic_dataset/german/senna/seed0.csv) |

Real data and Real + reweighing release no synthetic table, so they are not included.

---

## 📈 Results

These are the numbers reported in the paper. TSTR means train on synthetic data and test on real data. Fairness is computed on TSTR predictions for the real test set.

### Case study: one run through the four stages

This run uses Adult with `deepseek-v4.1-flash`, seed 0 (Fig. 2 of the paper; plan file: [`seed0_plan.json`](synthetic_dataset/adult/caitlyn_deepseek-v4.1-flash/seed0_plan.json)).

1. **Stage A.**
   - The evidence gate adds 15 edges from protected attributes to the LLM's plan of 25 edges.
   - LRTs add hours per week and native country as parents of income.
   - Of the 9 proposed legitimate columns, 5 pass the tests. Marital status, occupation, relationship and hours per week predict the protected attributes too well (PC 0.31 to 0.74).
2. **Stage B.** Relationship encodes sex best (PC = 0.74), but the generator barely uses it (PU = 0.04). Only marital status and occupation are therefore proxies.
3. **Stage C.**
   - One cut of 7 edges into income (weight 0.0169) blocks every unresolved path.
   - Per-attribute cuts would pay for five shared edges three times (weight 0.0365).
   - DECAF-style removal of every edge on an unresolved path would delete 25 edges.
4. **Edited plan.** Income is generated from workclass and education only.

### Utility, fidelity and privacy

TSTR AUC is computed with XGBoost and averaged over seeds. Rank is the average rank over dataset × classifier blocks (four classifiers, AUC averaged over seeds). DCR is the mean Gower distance to the closest training row; NN-risk is ideal at 0.5. **Bold** is the best synthetic result.

| Type | Method | TSTR Adult | TSTR Credit | TSTR German | Aug. AUC | Shape↑ | Trend↑ | α-P↑ | β-R↑ | Copy↓ | DCR↑ | NN-risk | Rank↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real | D<sub>train</sub> | 0.926 | 0.908 | 0.758 | – | – | – | – | – | – | – | – | – |
| DGM | CTGAN | 0.873 | **0.841** | 0.450 | 0.864 | 0.894 | 0.802 | 0.765 | 0.266 | **0.00** | 0.114 | 0.58 | 5.25 |
| LLM | CLLM | 0.875 | 0.500 | **0.719** | **0.870** | 0.835 | 0.744 | 0.585 | 0.215 | 0.06 | 0.156 | **0.55** | 5.08 |
| LLM | StructSynth | 0.877 | 0.697 | 0.514 | 0.852 | 0.808 | 0.699 | 0.444 | 0.173 | 0.05 | **0.191** | **0.55** | 6.08 |
| Fair | TabFairGAN | 0.840 | 0.796 | 0.465 | 0.859 | 0.936 | 0.848 | **0.916** | 0.404 | 0.10 | 0.093 | 0.59 | 6.83 |
| Fair | SENNA | **0.883** | 0.713 | 0.717 | **0.870** | 0.847 | 0.689 | 0.627 | 0.244 | **0.00** | 0.116 | 0.56 | **4.83** |
| Repair | Real + reweighing | 0.909 | 0.902 | 0.760 | – | – | – | – | – | – | – | – | 2.75 |
| Ours | **CAITLYN** (qwen3.8-flash) | 0.764 | 0.756 | 0.668 | 0.856 | 0.935 | 0.877 | 0.879 | 0.408 | **0.00** | 0.100 | 0.59 | 6.92 |
| Ours | **CAITLYN** (deepseek-v4.1-flash) | 0.641 | 0.812 | 0.697 | 0.859 | **0.956** | **0.904** | 0.874 | **0.491** | **0.00** | 0.074 | 0.62 | 6.17 |

TSTR AUC by dataset:
- **German credit.** CAITLYN trails only CLLM and SENNA (0.697 and 0.668 against 0.719 and 0.717). CTGAN, TabFairGAN and StructSynth fall to 0.45–0.51 with 700 training rows.
- **Credit scoring.** With `deepseek-v4.1-flash`, CAITLYN trails only CTGAN (0.812 against 0.841). CLLM curates away every negative row.
- **Adult.** CAITLYN trails the unconstrained generators and SENNA (0.764 against 0.873–0.883), because the joint cut blocks the strongest predictors of income. An oracle loses the same utility (see [Utility ceiling](#utility-ceiling-oracle)), so this gap is the price of the parity target.

Other results:
- CAITLYN has the best pair trends and β-recall.
- Like CTGAN and SENNA, it copies no training row, against 5–10% for the other synthetic baselines.

### Fairness over attribute combinations (summary)

Each value is the largest group gap among intersectional cells. The pipeline is:
1. XGBoost, mean over seeds.
2. Average over the combinations of each dataset: Adult has s×r, s×a, r×a and s×r×a; Credit scoring and German credit have s×a.
3. Average over the three datasets.

Lower is better. **Bold** marks the lowest value among synthetic-data methods.

| Method | SP↓ | EO↓ | EOd↓ | PP↓ | ABROCA↓ | ε-DF↓ | CSP<sub>R</sub>↓ |
|---|---|---|---|---|---|---|---|
| D<sub>train</sub> | 0.142 | 0.116 | 0.323 | 0.136 | 0.079 | 1.328 | 0.175 |
| Real + reweighing | 0.094 | 0.112 | 0.312 | 0.230 | 0.084 | 0.779 | 0.117 |
| CTGAN | 0.119 | 0.161 | 0.306 | 0.220 | **0.091** | 1.468 | 0.186 |
| CLLM | 0.124 | 0.135 | **0.197** | 0.190 | 0.128 | 1.205 | 0.135 |
| StructSynth | 0.152 | 0.168 | 0.317 | 0.184 | 0.143 | 1.198 | 0.152 |
| TabFairGAN | 0.138 | 0.131 | 0.298 | 0.222 | 0.114 | 1.440 | 0.165 |
| SENNA | 0.186 | 0.180 | 0.253 | **0.165** | 0.131 | 0.881 | 0.187 |
| **CAITLYN** (qwen3.8-flash) | 0.096 | 0.108 | 0.226 | 0.249 | 0.132 | 0.967 | **0.097** |
| **CAITLYN** (deepseek-v4.1-flash) | **0.059** | **0.075** | 0.232 | 0.265 | 0.138 | **0.628** | 0.098 |

Notes:
- CSP<sub>R</sub> is SP within strata of the resolving set R, weighted by P(r). It measures disparity that legitimate variables do not explain. It is averaged over Adult and Credit scoring, because German credit pools its test predictions across seeds.
- CLLM's Credit scoring rows come from a constant predictor (TSTR AUC 0.500).
- Predictive parity is CAITLYN's main loss: equal positive rates across cells with different base rates imply unequal precision.

### Fairness on the full intersection

XGBoost, mean over seeds; lower is better. Adult uses sex×race×age; Credit scoring and German credit use sex×age.

<details open>
<summary><b>Adult (sex × race × age, 12 cells)</b></summary>

| Method | SP | EO | EOd | PP | ABROCA | ε-DF |
|---|---|---|---|---|---|---|
| D<sub>train</sub> | .354 | .505 | .641 | .403 | .115 | 3.30 |
| CTGAN | .324 | .461 | .618 | .638 | .162 | 3.95 |
| CLLM | .278 | **.309** | **.429** | **.433** | .184 | 2.45 |
| StructSynth | .344 | .429 | .600 | .566 | .171 | 3.27 |
| TabFairGAN | .327 | .388 | .565 | .616 | **.133** | 2.08 |
| SENNA | .343 | .429 | .605 | .506 | .167 | 2.31 |
| Real + reweighing | .197 | .506 | .544 | .565 | .121 | 1.51 |
| **CAITLYN** (qwen3.8-flash) | .233 | .407 | .508 | .647 | .202 | 2.35 |
| **CAITLYN** (deepseek-v4.1-flash) | **.152** | .310 | .456 | .704 | .255 | **1.16** |

</details>

<details open>
<summary><b>Credit scoring (sex × age)</b></summary>

| Method | SP | EO | EOd | PP | ABROCA | ε-DF |
|---|---|---|---|---|---|---|
| D<sub>train</sub> | .041 | .007 | .304 | .054 | .078 | 1.00 |
| CTGAN | .053 | .033 | .199 | .057 | .086 | 0.90 |
| CLLM† | .000 | .000 | .000 | .082 | .000 | 0.95 |
| StructSynth | .029 | .022 | .229 | .069 | .106 | 0.68 |
| TabFairGAN | .057 | .030 | .300 | .066 | **.085** | 2.30 |
| SENNA | .103 | .086 | **.114** | **.050** | .127 | **0.23** |
| Real + reweighing | .031 | .007 | .343 | .058 | .066 | 0.80 |
| **CAITLYN** (qwen3.8-flash) | **.021** | **.010** | .191 | .074 | .154 | 0.88 |
| **CAITLYN** (deepseek-v4.1-flash) | .037 | .015 | .273 | .062 | .120 | 0.92 |

† CLLM yields a constant predictor (TSTR AUC 0.500), so its zero gaps are not bolded.

</details>

<details open>
<summary><b>German credit (sex × age; test predictions pooled across seeds)</b></summary>

| Method | SP | EO | EOd | PP | ABROCA | ε-DF |
|---|---|---|---|---|---|---|
| D<sub>train</sub> | .086 | .013 | .226 | .170 | .077 | 0.36 |
| CTGAN | **.032** | .066 | .211 | .204 | **.087** | 0.27 |
| CLLM | .139 | .214 | .304 | .204 | .252 | 0.63 |
| StructSynth | .140 | .168 | .268 | .169 | .207 | 0.41 |
| TabFairGAN | .089 | .087 | .192 | .162 | .169 | 0.41 |
| SENNA | .161 | .211 | .258 | **.135** | .154 | 0.43 |
| Real + reweighing | .093 | .030 | .271 | .188 | .101 | 0.38 |
| **CAITLYN** (qwen3.8-flash) | .083 | .059 | .159 | .179 | .117 | 0.35 |
| **CAITLYN** (deepseek-v4.1-flash) | .035 | **.009** | **.114** | .172 | .117 | **0.21** |

CTGAN has the lowest SP here, but its TSTR AUC (0.450) is below chance.

</details>

Tables 2–4 of the paper also report every **single** attribute (sex, race, age) and the PE and TE gaps.

Findings by dataset:
- **Adult.** CAITLYN with `deepseek-v4.1-flash` has the lowest SP gap and ε-DF among synthetic-data methods in all seven settings (3 single attributes + 4 combinations), except ε-DF for race. CLLM has the lowest EO and EOd gaps in most settings.
- **Credit scoring (sex × age).** CAITLYN has the lowest SP and EO gaps. SENNA has the lowest EOd, PP and ε-DF.
- **German credit (sex × age).** CAITLYN has the lowest EO, EOd and ε-DF. CTGAN, whose AUC is below chance, has the lowest SP.

### Statistical tests: SP over attribute combinations

**Setup.** Friedman test with Nemenyi post-hoc on average ranks:
- 24 blocks: dataset × classifier × combination, with SP averaged over seeds.
- k = 9 methods.

**Result.** Friedman p = 3.1 × 10⁻²³; Nemenyi critical difference CD = 2.45 (α = 0.05).

| Rank | Method | Avg. rank↓ |
|---|---|---|
| 1 | **CAITLYN** (deepseek-v4.1-flash) | **2.17** |
| 2 | **CAITLYN** (qwen3.8-flash) | **2.54** |
| 3 | Real + reweighing | 3.17 |
| 4 | CLLM | 4.17 |
| 5 | TabFairGAN | 4.88 |
| 6 | CTGAN | 5.83 |
| 7 | StructSynth | 6.46 |
| 8 | D<sub>train</sub> | 7.08 |
| 9 | SENNA | 8.71 |

**Which differences exceed the CD:**
- **CAITLYN (deepseek-v4.1-flash)** beats TabFairGAN, CTGAN, StructSynth, real data and SENNA.
- **CAITLYN (qwen3.8-flash)** beats CTGAN, StructSynth, real data and SENNA. Its gap to TabFairGAN (2.33) is just under the CD.

### Utility ceiling (oracle)

This table is in the supplementary material of the paper. The oracle samples each plan ancestrally, using LightGBM conditionals fitted on the training data. It separates the cost of the parity target from the cost of the LLM. Run it with `scripts/plan_oracle.py`.

| Dataset | Backbone | Oracle on G | Oracle on G' | CAITLYN | Real | ΔSP∩ on G | ΔSP∩ on G' | ΔSP∩ Real |
|---|---|---|---|---|---|---|---|---|
| Adult | qwen3.8-flash | 0.887 | 0.768 | 0.764 | 0.898 | 0.321 | 0.150 | 0.384 |
| Adult | deepseek-v4.1-flash | 0.854 | 0.709 | 0.641 | 0.898 | 0.361 | 0.156 | 0.384 |
| Credit scoring | qwen3.8-flash | 0.710 | 0.730 | 0.756 | 0.873 | 0.029 | 0.026 | 0.050 |
| Credit scoring | deepseek-v4.1-flash | 0.682 | 0.802 | 0.812 | 0.873 | 0.025 | 0.037 | 0.050 |
| German credit | qwen3.8-flash | 0.718 | 0.713 | 0.668 | 0.758 | 0.095 | 0.106 | 0.097 |
| German credit | deepseek-v4.1-flash | 0.651 | 0.730 | 0.697 | 0.758 | 0.111 | 0.130 | 0.097 |

On Adult, surgery lowers the oracle AUC from 0.887 to 0.768 while halving ΔSP∩. CAITLYN with qwen3.8-flash reaches this ceiling (0.764), so its gap to the unconstrained generators is the price of the parity target.

---

## 🔬 Baseline Methods

| Method | Type | How it is run |
|---|---|---|
| CTGAN | DGM | `run_baseline.py --method ctgan` (SDV) |
| CLLM | LLM, curated in-context generation | `run_baseline.py --method cllm` (same backbone) |
| StructSynth | LLM, follows an LLM-induced dependency graph | `run_baseline.py --method structsynth` (same backbone) |
| TabFairGAN | Fair GAN | `run_baseline.py --method tabfairgan`; the intersectional cell is the protected group |
| SENNA | Fair LLM, disentangled encoder + k-NN validation | Authors' encoder and loss in fair mode over all protected attributes, with the same backbone; imported with `--method import --name senna` |
| Real + reweighing | Repair (no synthetic table) | `run_baseline.py --method reweigh`, intersectional reweighing |

All LLM-based methods share the backbone (`qwen3.8-flash`) and the number of rows (1,000).

How the SENNA run differs from the original setup, because of compute (one 4 GB GPU):
- The encoder is trained on at most 5,000 rows for 20 epochs, with sequences of 128 tokens (160 on German credit) and batches of 8.
- Rows are verbalised with a generic template, "The ⟨column⟩ is ⟨value⟩."
- The targeted per-batch requests of the fairness governor are omitted.

---

## ⚖️ Evaluation

Implemented in [`caitlyn/eval/`](caitlyn/eval/) and driven by [`scripts/evaluate.py`](scripts/evaluate.py).

**Utility**
- TSTR (train on synthetic, test on real) and augmentation (D<sub>train</sub> ∪ D<sub>synth</sub>).
- Classifiers: XGBoost, random forest, logistic regression and MLP. Metric: AUC.

**Fidelity**
- SDMetrics column shapes and pair trends, α-precision and β-recall.

**Privacy**
- Copy rate, distance to the closest record (DCR) and nearest-neighbour membership risk.

**Fairness**
- Computed for each single protected attribute and each combination. The groups of a combination are its intersectional cells; groups with fewer than 30 test rows are excluded.
- Metrics, each reported as the largest group gap:
  - **SP:** positive rate.
  - **EO:** TPR.
  - **EOd:** TPR + FPR.
  - **PP:** PPV.
  - **PE:** FPR.
  - **TE:** FN/FP ratio.
  - **ABROCA**.
  - **ε-DF:** smoothed ε-differential fairness.
- **ΔSP∩:** SP over the full intersection.
- **CSP<sub>R</sub>:** the P(r)-weighted ΔSP∩ within strata of R.

**Statistics**
- Friedman and Nemenyi tests on ranks (`caitlyn/eval/stats.py`), with optional bootstrap CIs (`--bootstrap`).

---

## 🐛 Troubleshooting

- **`Could not find module lib_lightgbm.dll` (Windows).** Install the *Microsoft Visual C++ Redistributable 2015–2022 (x64)*, which provides `vcomp140.dll`.
- **A path with non-ASCII characters breaks the venv.** Create the venv in an ASCII-only path, e.g. `C:\venvs\caitlyn`.
- **Rate limits or authentication errors.** Keep `llm.max_concurrency` at 3 or below. Interrupted runs resume from `results/llm_cache.sqlite`; do not delete this file between runs.
- **Mock results appear in the tables.** Delete the `*__mock` folders before real runs.
- **Separate environment for the GAN baselines.** CTGAN and TabFairGAN pin their own torch, so install them in a separate venv.

---

## 📋 Requirements

Python 3.10+ (tested with 3.12). See [`requirements.txt`](requirements.txt). Main dependencies:
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `networkx`, `statsmodels`
- `openai` (OpenAI-compatible endpoint), `tenacity`
- `sdmetrics`, `scikit-posthocs`, `matplotlib`
- Optional: `causal-learn` (PC-algorithm plan ablation). Install `sdv` (CTGAN) in a separate venv.

---

## 🙏 Acknowledgments

- Datasets: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), and Credit scoring (Kaggle, Davronov 2021).
- Baselines: CTGAN (SDV), CLLM, StructSynth, TabFairGAN, SENNA.
- Fidelity metrics: [SDMetrics](https://github.com/sdv-dev/SDMetrics).

---

**Star ⭐ this repo if you find it useful!**
