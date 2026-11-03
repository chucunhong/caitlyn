"""Configuration loading: run.yaml, dataset yaml files and .env."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_env(path=ROOT / ".env"):
    """Minimal .env reader (KEY=VALUE per line); does not override existing variables."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_run_config(path=CONFIGS / "run.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, items) -> dict:
    """--set caitlyn.B=20 llm.model=qwen-max ... (values parsed as YAML scalars)."""
    for it in items or []:
        key, val = it.split("=", 1)
        d = cfg
        *path, last = key.split(".")
        for k in path:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    return cfg


@dataclass
class DatasetMeta:
    name: str
    label: str
    protected: list
    task: str
    columns: dict
    label_description: str = ""
    truth: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    eval_drop: list = field(default_factory=list)   # columns hidden from downstream classifiers (label leakage)

    def describe(self, col):
        return self.columns.get(col, col)


def dataset_base(name: str) -> str:
    return name


def load_dataset_meta(name: str) -> DatasetMeta:
    with open(CONFIGS / "datasets" / f"{dataset_base(name)}.yaml", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d["name"] = name
    return DatasetMeta(**{k: v for k, v in d.items() if k in DatasetMeta.__dataclass_fields__})
