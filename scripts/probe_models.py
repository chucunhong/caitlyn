"""Probe Model Studio models before the real runs: availability, JSON validity, latency, tokens and
the maximum useful concurrency (rate limits).

  python scripts/probe_models.py --list
  python scripts/probe_models.py --models qwen-flash qwen-plus deepseek-v4.1-flash glm-5.3 --n 6
  python scripts/probe_models.py --models qwen-flash --concurrency 4 8 16 32 64
Writes results/_probe/<model>.json. Probe calls are not cached and cost a few cents.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import _common  # noqa: F401
import numpy as np

from caitlyn.config import RESULTS, apply_overrides, load_run_config
from caitlyn.data import column_domains, domain_text, load_split, rows_text, validate_value
from caitlyn.llm import LLM, _parse_json
from caitlyn.prompts import render


def gen_prompt(seed, bs=50):
    """A real pi_gen prompt (Adult, label given three parents)."""
    meta, tr, _ = load_split("adult", 0)
    dom = column_domains(tr)
    pa, v = ["education_num", "occupation", "hours_per_week"], "income"
    rows = tr.sample(bs, random_state=seed)[pa].reset_index(drop=True)
    ex = tr.sample(10, random_state=seed + 1)[pa + [v]]
    p = render("gen", target=v, target_description=meta.describe(v), domain=domain_text(dom[v]),
               parent_columns=", ".join(pa), rows_parent_only=rows_text(rows), exemplars=rows_text(ex, with_id=False))
    return p, dom[v], bs


def one_call(llm, prompt, dom, bs, temperature=0.9):
    t0 = time.time()
    try:
        resp = llm._request(prompt, temperature, 0.95, None)
        dt = time.time() - t0
        txt = resp.choices[0].message.content or ""
        u = resp.usage
        try:
            vals = _parse_json(txt).get("values", {})
            ok = sum(validate_value(vals.get(str(i)), dom) is not None for i in range(bs))
            parsed = True
        except Exception:
            ok, parsed = 0, False
        return dict(ok=True, seconds=dt, parsed=parsed, valid_frac=ok / bs,
                    tin=getattr(u, "prompt_tokens", 0), tout=getattr(u, "completion_tokens", 0))
    except Exception as e:
        return dict(ok=False, seconds=time.time() - t0, error=type(e).__name__, msg=str(e)[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=6, help="sequential quality calls per model")
    ap.add_argument("--concurrency", type=int, nargs="*", default=[], help="levels for the rate-limit test")
    ap.add_argument("--per-level", type=int, default=2, help="requests per worker at each concurrency level")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = apply_overrides(load_run_config(), a.set)
    cfg["llm"]["max_retries"] = 1                      # surface 429s instead of retrying them
    out = RESULTS / "_probe"
    out.mkdir(parents=True, exist_ok=True)

    if a.list:
        llm = LLM(cfg)
        ids = sorted(m.id for m in llm.client.models.list())
        keep = [m for m in ids if re.search(r"qwen|deepseek|glm|kimi", m.lower())]
        print(f"{len(ids)} models; text candidates:\n  " + "\n  ".join(keep))
        (out / "models.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")

    for model in a.models:
        llm = LLM(cfg, model=model)
        rep = {"model": model}
        q = [one_call(llm, *gen_prompt(1000 + i)) for i in range(a.n)]
        good = [r for r in q if r["ok"]]
        rep["quality"] = dict(
            calls=len(q), errors=[r for r in q if not r["ok"]][:2],
            parsed=float(np.mean([r["parsed"] for r in good])) if good else 0.0,
            valid_frac=float(np.mean([r["valid_frac"] for r in good])) if good else 0.0,
            latency_s=float(np.median([r["seconds"] for r in good])) if good else None,
            tokens_in=float(np.mean([r["tin"] for r in good])) if good else None,
            tokens_out=float(np.mean([r["tout"] for r in good])) if good else None)
        print(json.dumps(rep["quality"]))
        rep["concurrency"] = []
        for c in a.concurrency:
            jobs = [gen_prompt(2000 + c * 100 + i) for i in range(c * a.per_level)]
            t0 = time.time()
            with ThreadPoolExecutor(c) as pool:
                rs = list(pool.map(lambda j: one_call(llm, *j), jobs))
            wall = time.time() - t0
            n429 = sum(1 for r in rs if not r["ok"] and "RateLimit" in r.get("error", ""))
            nerr = sum(1 for r in rs if not r["ok"])
            row = dict(concurrency=c, requests=len(rs), errors=nerr, rate_limited=n429, wall_s=round(wall, 1),
                       calls_per_min=round(60 * (len(rs) - nerr) / wall, 1),
                       tokens_per_min=round(60 * sum(r.get("tin", 0) + r.get("tout", 0) for r in rs if r["ok"]) / wall))
            rep["concurrency"].append(row)
            print(json.dumps(row))
            if n429 > 0.2 * len(rs):                     # saturated: stop ramping
                break
        (out / f"{model}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
