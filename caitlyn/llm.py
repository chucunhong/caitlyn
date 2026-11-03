"""LLM client for Alibaba Cloud Model Studio (DashScope, OpenAI-compatible) with
sqlite cache, retries, JSON parsing and per-stage token accounting (guide Sec. 3)."""
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import ROOT, load_env


class LLMError(RuntimeError):
    pass


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


_ENTRY = re.compile(r'"(\d+)"\s*:\s*(\[[^\[\]]*\]|"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)')


def _salvage_values(text: str, ids) -> dict | None:
    """Rows of a truncated {"values": {...}} answer (e.g. the model looped until max_tokens): the first
    complete entry of each requested id is kept; None if no entry can be read."""
    vals = {}
    for k, v in _ENTRY.findall(text):
        if int(k) in ids and k not in vals:
            try:
                vals[k] = json.loads(v)
            except json.JSONDecodeError:
                continue
    return {"values": vals, "_salvaged": True} if vals else None


class Cache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT, tin INT, tout INT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            r = self.con.execute("SELECT v, tin, tout FROM c WHERE k=?", (k,)).fetchone()
        return (json.loads(r[0]), r[1], r[2]) if r else None

    def set(self, k, v, tin, tout):
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO c VALUES (?,?,?,?)", (k, json.dumps(v), tin, tout))
            self.con.commit()


class LLM:
    """`call(prompt, temperature, seed_tag, stage)` -> dict parsed from the model's JSON answer.

    Identical (model, temperature, top_p, seed_tag, prompt) are served from the cache, so
    reruns and ablations that share stages are free and reproducible."""

    def __init__(self, cfg: dict, model: str | None = None):
        load_env()
        c = cfg["llm"]
        self.cfg = c
        self.model = {"open": c.get("open_model"), None: c["model"]}.get(model, model)
        self.cache = Cache(ROOT / c["cache"])
        self.usage = defaultdict(lambda: defaultdict(float))
        self.lock = threading.Lock()
        self.sem = threading.Semaphore(int(c.get("max_concurrency", 8)))
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            if self.cfg.get("provider") == "token_plan":     # subscription plan: own key and endpoint
                key = os.environ.get("TOKEN_PLAN_API_KEY", "").strip()
                if not key:
                    raise LLMError("Set TOKEN_PLAN_API_KEY (the plan-specific sk-sp-... key) in .env")
                base = os.environ.get("TOKEN_PLAN_BASE_URL", "").strip() or self.cfg["token_plan_base_url"]
                self._client = OpenAI(api_key=key, base_url=base, timeout=self.cfg.get("timeout", 120))
                return self._client
            names = [self.cfg.get("api_key_env", "ALIBABA_CLOUD_API_KEY"), "ALIBABA_CLOUD_API_KEY", "DASHSCOPE_API_KEY"]
            key = next((os.environ[n].strip() for n in names if os.environ.get(n, "").strip()), None)
            if not key:
                raise LLMError("Missing API key: set ALIBABA_CLOUD_API_KEY in the environment or in .env")
            base = os.environ.get("ALIBABA_CLOUD_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or self.cfg["base_url"]
            host = os.environ.get("API_HOST", "").strip().rstrip("/")
            if host and not os.environ.get("ALIBABA_CLOUD_BASE_URL"):   # e.g. <ws>.ap-southeast-1.maas.aliyuncs.com
                base = (host if host.startswith("http") else "https://" + host)
                if not base.endswith("/compatible-mode/v1"):
                    base += "/compatible-mode/v1"
            if "{workspace}" in base:                 # workspace endpoints: {WorkspaceId}.<region>.maas.aliyuncs.com
                ws = (os.environ.get("ALIBABA_CLOUD_WORKSPACE_ID") or os.environ.get("WORKSPACE_ID", "")).strip()
                if not ws:
                    raise LLMError("Set ALIBABA_CLOUD_WORKSPACE_ID in .env (Model Studio > Settings > Workspace)")
                base = base.replace("{workspace}", ws)
            self._client = OpenAI(api_key=key, base_url=base, timeout=self.cfg.get("timeout", 120))
        return self._client

    def _record(self, stage, tin, tout, dt, cached):
        with self.lock:
            u = self.usage[stage]
            u["calls"] += 1
            u["cached"] += int(cached)
            u["tokens_in"] += tin
            u["tokens_out"] += tout
            u["seconds"] += dt

    def _request(self, prompt, temperature, top_p, seed, max_tokens=None):
        from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError
        kw = dict(model=self.model, temperature=temperature, messages=[{"role": "user", "content": prompt}])
        if top_p is not None:
            kw["top_p"] = top_p
        if max_tokens:
            kw["max_tokens"] = int(max_tokens)
        if self.cfg.get("json_mode", True):
            kw["response_format"] = {"type": "json_object"}
        if seed is not None:
            kw["seed"] = seed
        if self.cfg.get("enable_thinking") is not None:
            kw["extra_body"] = {"enable_thinking": bool(self.cfg["enable_thinking"])}
        delay = 2.0
        for attempt in range(int(self.cfg.get("max_retries", 6))):
            try:
                with self.sem:
                    return self.client.chat.completions.create(**kw)
            except BadRequestError as e:
                if "extra_body" in kw and "thinking" in str(e).lower():
                    kw.pop("extra_body")          # model without a thinking switch
                    continue
                raise
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                time.sleep(delay + np.random.rand())
                delay = min(delay * 2, 60)
        raise LLMError("LLM request failed after retries")

    def call(self, prompt: str, *, temperature: float, seed_tag: str = "0", stage: str = "misc",
             top_p: float | None = None, hint: dict | None = None, max_tokens: int | None = None) -> dict:
        """max_tokens bounds answers that loop; for generation calls (hint kind gen/group) the complete
        rows of a truncated answer are kept."""
        key = hashlib.sha256(f"{self.model}|{temperature}|{top_p}|{seed_tag}|{prompt}".encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            self._record(stage, hit[1], hit[2], 0.0, True)
            return hit[0]
        seed = int(hashlib.md5(seed_tag.encode()).hexdigest()[:8], 16) % (2 ** 31)
        last = None
        for j in range(3):                       # malformed JSON -> ask again with a shifted seed
            t0 = time.time()
            resp = self._request(prompt, temperature, top_p, (seed + j) % (2 ** 31), max_tokens)
            tin = getattr(resp.usage, "prompt_tokens", 0) or 0
            tout = getattr(resp.usage, "completion_tokens", 0) or 0
            self._record(stage, tin, tout, time.time() - t0, False)
            try:
                out = _parse_json(resp.choices[0].message.content or "")
                self.cache.set(key, out, tin, tout)
                return out
            except (json.JSONDecodeError, TypeError) as e:
                last = e
                if hint and hint.get("kind") in ("gen", "group"):
                    out = _salvage_values(resp.choices[0].message.content or "", set(hint.get("ids", [])))
                    if out is not None:
                        self.cache.set(key, out, tin, tout)
                        return out
        raise LLMError(f"Unparseable JSON from model: {last}")

    def usage_dict(self):
        return {s: dict(u) for s, u in self.usage.items()}

    def reset_usage(self):
        self.usage.clear()


class MockLLM(LLM):
    """Offline stand-in for smoke tests (`--mock`): answers every prompt with random valid JSON.
    Never use its outputs for the paper."""

    def __init__(self, cfg: dict, model=None):
        self.cfg, self.model = cfg["llm"], "mock"
        self.usage = defaultdict(lambda: defaultdict(float))
        self.lock = threading.Lock()

    def call(self, prompt, *, temperature, seed_tag="0", stage="misc", top_p=None, hint=None, max_tokens=None):
        rng = np.random.default_rng(int(hashlib.md5((seed_tag + prompt[-200:]).encode()).hexdigest()[:8], 16))
        self._record(stage, len(prompt) // 4, 50, 0.0, False)
        hint = hint or {}
        kind = hint.get("kind")
        if kind == "role":
            cols = hint["columns"]
            pick = list(rng.choice(cols, size=min(len(cols), 3), replace=False))
            return {"resolving": [{"column": c, "rationale": "mock"} for c in pick],
                    "proxy_candidates": [{"column": cols[0], "rationale": "mock"}]}
        if kind == "edge":
            cand = hint["candidates"]
            k = int(rng.integers(0, min(3, len(cand)) + 1)) if cand else 0
            return {"children": [{"column": c, "rationale": "mock"} for c in rng.choice(cand, size=k, replace=False)]}
        if kind == "cycle":
            return {"remove": list(hint["cycle"][0]), "reason": "mock"}
        if kind == "gen":
            dom = hint["domain"]
            vals = {}
            for rid in hint["ids"]:
                if dom["type"] == "numeric":
                    v = float(rng.uniform(dom["min"], dom["max"]))
                    vals[str(rid)] = round(v) if dom.get("integer") else round(v, 3)
                else:
                    vals[str(rid)] = str(rng.choice(dom["values"]))
            return {"values": vals}
        if kind == "group":
            vals = {}
            for rid in hint["ids"]:
                row = []
                for dom in hint["domains"].values():
                    if dom["type"] == "numeric":
                        v = float(rng.uniform(dom["min"], dom["max"]))
                        row.append(round(v) if dom.get("integer") else round(v, 3))
                    else:
                        row.append(str(rng.choice(dom["values"])))
                vals[str(rid)] = row
            return {"values": vals}
        if kind == "rows":
            rows = []
            for rid in hint["ids"]:
                r = {"id": rid}
                for c, dom in hint["domains"].items():
                    if dom["type"] == "numeric":
                        v = float(rng.uniform(dom["min"], dom["max"]))
                        r[c] = round(v) if dom.get("integer") else round(v, 3)
                    else:
                        r[c] = str(rng.choice(dom["values"]))
                rows.append(r)
            return {"rows": rows}
        return {}


def make_llm(cfg, model=None, mock=False):
    return MockLLM(cfg) if mock else LLM(cfg, model)
