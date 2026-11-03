"""Prompt templates (caitlyn/prompts/*.txt). `{key}` placeholders are replaced only for the
keys passed, so the literal JSON braces in the templates are left untouched."""
from functools import lru_cache
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"


@lru_cache(None)
def template(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **kw) -> str:
    t = template(name)
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t
