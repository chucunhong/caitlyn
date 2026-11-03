"""Shared helpers for the scripts: make the repo importable and enumerate result directories."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REAL_DATASETS = ["adult", "credit_scoring", "german"]
PAPER = ROOT / "paper" / "main.tex"


def inject(name, body):
    """Replace the body of the block `%%% AUTO <name> BEGIN ... %%% AUTO <name> END` in paper/main.tex
    (the paper is a single self-contained file; generated tables and macros live in these blocks)."""
    import re
    tex = PAPER.read_text(encoding="utf-8")
    pat = re.compile(rf"(%%% AUTO {re.escape(name)} BEGIN[^\n]*\n).*?(\n%%% AUTO {re.escape(name)} END)", re.S)
    if not pat.search(tex):
        print(f"warning: no AUTO block '{name}' in {PAPER.name}; skipped")
        return
    PAPER.write_text(pat.sub(lambda m: m.group(1) + body.rstrip() + m.group(2), tex, count=1), encoding="utf-8")
    print(f"updated block '{name}' in {PAPER.name}")

# Display names and groups for the tables / figures (order = table order)
METHODS = [
    ("real", "$D_{\\text{train}}$", "Real"),
    ("ctgan", "CTGAN", "DGM"),
    ("tabsyn", "TabSyn", "DGM"),
    ("great", "GReaT", "LLM"),
    ("cllm", "CLLM", "LLM"),
    ("structsynth", "StructSynth", "LLM"),
    ("tabfairgan", "TabFairGAN", "Fair"),
    ("decaf", "DECAF", "Fair"),
    ("golob", "Golob et al.\\ (CF)", "Fair"),
    ("fairtabddpm", "FairTabDDPM", "Fair"),
    ("senna", "SENNA", "Fair"),
    ("reweigh", "Real + reweighing", "Repair"),
    ("caitlyn", "\\method", "Ours"),
    ("caitlyn__model-deepseek-v4.1-flash", "\\method{} (\\openbackbone)", "Ours"),
]
LABEL = {k: v for k, v, _ in METHODS}
GROUP = {k: g for k, _, g in METHODS}
