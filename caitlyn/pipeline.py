"""End-to-end CAITLYN pipeline with variants (ablations, seesaw, leakage, sensitivity) and
caching of Stage A-C artifacts, so variants that share stages do not recompute them."""
import hashlib
import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from .common import to_jsonable
from .config import RESULTS
from .stage_a_plan import build_plan, complete_label_parents, evidence_gate, gate_tests, pc_plan, role_candidates, roles_of, select_resolving
from .stage_b_proxy import per_attribute_proxies, proxy_audit
from .stage_c_surgery import surgery
from .stage_d_generate import Generator

# Each variant overrides the defaults below. Names are used as method suffixes: caitlyn__<variant>.
DEFAULTS = dict(r_mode="tests", plan="llm", gate=True, proxy=True, surgery="joint", order=None,
                cut_only=False, context="parent", exemplars="matched", batching="mixed",
                joint_call=False, lam=None, gamma=None, delta=None, tau_p=None, k=None, n_train=None,
                model=None, grouping=None)

VARIANTS = {
    "full": {},
    # ablations (Table 6)
    "nogate": dict(gate=False),
    "pcplan": dict(plan="pc"),
    "noproxy": dict(proxy=False, gamma=0.0),
    "perattr": dict(surgery="per_attr"),
    "rllm": dict(r_mode="llm"),
    "rempty": dict(r_mode="empty"),
    "fullhist": dict(context="full"),
    "lam0": dict(lam=0.0),
    "cutonly": dict(cut_only=True),
    "jointcall": dict(joint_call=True),
    "percol": dict(grouping="column"),         # one call per column instead of per layer group
    "uniformex": dict(exemplars="independent"),  # exemplars drawn uniformly instead of matched on pa(v)
    # seesaw (Table 5): "none" keeps the plan unedited; per-attribute orders are passed with --order
    "none": dict(surgery="none"),
    # generator variants for the leakage analysis (Fig. 2b): controls of Stage D switched off cumulatively
    "leak_full": dict(context="full", exemplars="cell", batching="cell", lam=0.0),
    "leak_parent": dict(exemplars="cell", batching="cell", lam=0.0),
    "leak_indep": dict(batching="cell", lam=0.0),
}


def resolve_variant(name, **overrides):
    opts = dict(DEFAULTS)
    opts.update(VARIANTS.get(name, {}))
    opts.update({k: v for k, v in overrides.items() if v is not None})
    return opts


def method_name(variant, opts):
    parts = [] if variant == "full" else [variant]
    for k in ("n_train", "lam", "k", "delta", "tau_p", "gamma"):
        if opts[k] is not None and k not in VARIANTS.get(variant, {}):
            parts.append(f"{k}{opts[k]}")
    if opts["order"] and variant == "perattr":
        parts.append("order-" + "-".join(opts["order"]))
    if opts["model"]:
        parts.append(f"model-{opts['model']}")
    return "caitlyn" + ("__" + "_".join(parts) if parts else "")


def _sig(d):
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def g2json(G):
    try:
        return nx.node_link_data(G, edges="links")
    except TypeError:
        return nx.node_link_data(G)


def json2g(d):
    try:
        return nx.node_link_graph(d, directed=True, edges="links")
    except TypeError:
        return nx.node_link_graph(d, directed=True)


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def _save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(to_jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")


class Pipeline:
    def __init__(self, llm, meta, train, cfg, seed, variant="full", **overrides):
        self.llm, self.train, self.cfg, self.seed = llm, train, cfg, seed
        self.opts = resolve_variant(variant, **overrides)
        self.variant = variant
        self.c = dict(cfg["caitlyn"])
        for k in ("lam", "gamma", "delta", "tau_p"):
            if self.opts[k] is not None:
                self.c[k] = self.opts[k]
        if self.opts["k"]:
            meta.protected = list(meta.protected)[: int(self.opts["k"])]
        self.meta = meta
        self.method = method_name(variant, self.opts)
        self.stage_dir = RESULTS / meta.name / "_stages" / f"seed{seed}"
        self.timing = {}
        self.R = None
        self.label_parents = bool(self.c.get("label_parents", False)) and self.opts["gate"]
        base = dict(model=llm.model, n=len(train), S=meta.protected, seed=seed)
        a_sig = {**base, **{k: self.opts[k] for k in ("r_mode", "plan", "gate")},
                 **{k: self.c[k] for k in ("delta", "tau_p", "alpha", "B", "tau", "K")}}
        self.sigA0 = _sig(a_sig)                      # Stage A without the label-parent completion
        self.sigA = _sig({**a_sig, "label_parents": True}) if self.label_parents else self.sigA0
        self._set_sigs(self.sigA)

    def _set_sigs(self, sig_a):
        self.sigB = _sig({"A": sig_a, "proxy": self.opts["proxy"], "M": self.c["M"], "J": self.c["J"]})
        self.sigC = _sig({"B": self.sigB, **{k: self.opts[k] for k in ("surgery", "order", "cut_only")},
                          "eta": self.c["eta"], "gamma": self.c["gamma"],
                          **({"refine_cut": True} if self.c.get("refine_cut") and self.opts["surgery"] == "joint" else {})})

    # ------------------------------------------------------------ Stage A
    def stage_a(self):
        p = self.stage_dir / f"A_{self.sigA}.json"
        if (d := _load(p)) is not None:
            d["G"], d["G0"] = json2g(d["G"]), json2g(d["G0"])
            self.R = d["R"]
            return d
        if self.label_parents and (d := _load(self.stage_dir / f"A_{self.sigA0}.json")) is not None:
            # reuse the cached Stage A and add only the label-parent completion
            G, d["label_parents_added"] = complete_label_parents(json2g(d["G"]), self.train, self.meta.protected,
                                                                 self.meta.label, self.seed, self.c["alpha"],
                                                                 int(self.c["lrt_max_rows"]))
            d["G"] = g2json(G)
            _save(p, d)
            d["G"], d["G0"] = G, json2g(d["G0"])
            self.R = d["R"]
            return d
        t0, S, Y, c = time.time(), self.meta.protected, self.meta.label, self.c
        n_jobs = int(c.get("n_jobs", 1))
        cand, llm_prox, role_log = role_candidates(self.llm, self.meta, self.train, c, self.seed)
        if self.opts["r_mode"] == "tests":
            R, rejected, rdet = select_resolving(self.train, S, Y, cand, self.seed, c["delta"], c["tau_p"],
                                                 c["alpha"], int(c["B"]), n_jobs)
        elif self.opts["r_mode"] == "llm":
            R, rejected, rdet = list(cand), {}, {}
        else:
            R, rejected, rdet = [], {r: "R forced empty" for r in cand}, {}
        proxy_cands = sorted(set(llm_prox) | set(rejected))
        roles = roles_of(self.meta, R, proxy_cands, self.train.columns)
        t1 = time.time()
        if self.opts["plan"] == "pc":
            G0, plan_log = pc_plan(self.train, self.meta, self.seed, c["alpha"]), {}
        else:
            G0, plan_log = build_plan(self.llm, self.meta, self.train, roles, c, self.seed)
        t2 = time.time()
        tests = gate_tests(self.train, S, self.seed, int(c["B"]), n_jobs, c["alpha"])
        if self.opts["gate"]:
            G, rej_edges, added, gate_log = evidence_gate(G0, self.train, S, self.seed, c["alpha"], c["tau"],
                                                          int(c["B"]), n_jobs, int(c["lrt_max_rows"]), tests)
        else:
            G, rej_edges, added = G0.copy(), [], []
            gate_log = {"assoc": tests[0], "p": tests[1], "p_holm": tests[2]}
        label_added = []
        if self.label_parents:
            G, label_added = complete_label_parents(G, self.train, S, Y, self.seed, c["alpha"], int(c["lrt_max_rows"]))
        d = dict(candidates=cand, llm_proxy_candidates=llm_prox, R=R, R_rejected=rejected, R_details=rdet,
                 roles=roles, role_log=role_log, plan_log=plan_log, prior_driven=rej_edges, lrt_added=added,
                 label_parents_added=label_added,
                 gate=gate_log, G0=g2json(G0), G=g2json(G),
                 seconds=dict(roles_and_R=t1 - t0, plan=t2 - t1, gate=time.time() - t2),
                 usage={k: v for k, v in self.llm.usage_dict().items() if k.startswith("A_")})
        _save(p, d)
        d["G"], d["G0"] = G, G0
        self.R = R
        return d

    # ------------------------------------------------------------ Stage B
    def stage_b(self, A):
        p = self.stage_dir / f"B_{self.sigB}.json"
        if (d := _load(p)) is not None:
            return d
        t0 = time.time()
        P, log = proxy_audit(self.llm, self.meta, self.train, A["G"], A["R"], self.c, self.seed,
                             extra_candidates=A["llm_proxy_candidates"], skip_pu=not self.opts["proxy"])
        if not self.opts["proxy"]:
            P, log["proxies"] = [], []
        log["seconds"] = time.time() - t0
        log["usage"] = {k: v for k, v in self.llm.usage_dict().items() if k.startswith("B_")}
        _save(p, log)
        return log

    # ------------------------------------------------------------ Stage C
    def stage_c(self, A, B):
        p = self.stage_dir / f"C_{self.sigC}.json"
        if (d := _load(p)) is not None:
            d["G2"] = json2g(d["G2"])
            return d
        t0, S, Y = time.time(), self.meta.protected, self.meta.label
        if self.opts["surgery"] == "none":
            G2, log = A["G"].copy(), {"cut": [], "ops": {}, "weight": 0.0}
        else:
            per_p = None
            if self.opts["surgery"] == "per_attr":
                per_p = {s: per_attribute_proxies(B, self.train, s, A["R"], self.seed) for s in S}
            G2, log = surgery(A["G"], self.train, S, A["R"], Y, B["proxies"], self.c, self.seed,
                              mode="joint" if self.opts["surgery"] == "joint" else "per_attr",
                              S_order=self.opts["order"], per_attr_proxies=per_p, cut_only=self.opts["cut_only"])
        log["seconds"] = time.time() - t0
        log["G2"] = g2json(G2)
        _save(p, log)
        log["G2"] = G2
        return log

    # ------------------------------------------------------------ Stage D
    def generator(self, G2, stage="D_generate", R=None):
        o = self.opts
        return Generator(self.llm, self.meta, self.train, G2, self.c, self.seed, context=o["context"],
                         exemplars=o["exemplars"], batching=o["batching"], joint_call=o["joint_call"],
                         lam=self.c["lam"], stage=stage, R=self.R if R is None else R,
                         grouping=o["grouping"] or self.c.get("grouping", "column"))

    def run(self, outdir: Path, m=None, stages="ABCD"):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.llm.reset_usage()
        A = self.stage_a()
        if self.label_parents and not A.get("label_parents_added"):
            self._set_sigs(self.sigA0)                 # same plan as without the completion: reuse Stages B-C
        B = self.stage_b(A) if "B" in stages or "C" in stages or "D" in stages else None
        C = self.stage_c(A, B) if "C" in stages or "D" in stages else None
        summary = dict(method=self.method, variant=self.variant, opts=self.opts, seed=self.seed,
                       protected=self.meta.protected, R=A["R"], candidates=A["candidates"],
                       proxies=B["proxies"] if B else None, cut=C["cut"] if C else None,
                       ops=C["ops"] if C else None, stage_files=dict(A=self.sigA, B=self.sigB, C=self.sigC))
        if C is not None:
            _save(outdir / "plan_edited.json", {"G2": g2json(C["G2"]), "G": g2json(A["G"]), "R": A["R"],
                                                "P": B["proxies"], "cut": C["cut"], "ops": C["ops"],
                                                "cost": C.get("cost", {}), "prior_driven": A["prior_driven"],
                                                "protected": self.meta.protected, "label": self.meta.label})
        if "D" in stages:
            m = int(m or self.c.get("synth_rows") or len(self.train))
            t0 = time.time()
            gen = self.generator(C["G2"])
            synth = gen.generate(m, seed_tag=f"D|{self.seed}")
            synth.to_csv(outdir / "synth.csv", index=False)
            summary.update(rows_requested=m, rows=len(synth), gen_stats=gen.stats, gen_seconds=time.time() - t0)
        usage = dict(A.get("usage", {}))
        usage.update((B or {}).get("usage", {}))
        usage.update({k: v for k, v in self.llm.usage_dict().items() if k.startswith("D")})
        summary["usage"] = usage
        summary["stage_seconds"] = {"A": A.get("seconds"), "B": B.get("seconds") if B else None,
                                    "C": C.get("seconds") if C else None}
        _save(outdir / "run.json", summary)
        return summary, A, B, C
