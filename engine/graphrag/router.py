"""
engine/graphrag/router.py
─────────────────────────
Classify a natural-language question into one of 3 GraphRAG templates.
Returns (template_name, params_dict). Never crashes.

Templates:
  explain  — "Why did acc_201 score 78?"
  pattern  — "Have we won accounts like acc_201 before?"
  compare  — "Why is acc_201 scored higher than acc_202?"
  unknown  — no template matched
"""
from __future__ import annotations
import re

_ACCOUNT_RE = re.compile(r"\b(acc_\d+)\b", re.IGNORECASE)

_COMPARE = [
    re.compile(r"why\s+is.+higher.+than", re.I),
    re.compile(r"why\s+is.+lower.+than", re.I),
    re.compile(r"compar.+scor", re.I),
    re.compile(r"differ.+between.+lead", re.I),
    re.compile(r"\bvs\.?\s+acc_", re.I),
]
_EXPLAIN = [
    re.compile(r"why\s+(did|does|is).+scor", re.I),
    re.compile(r"explain.+scor", re.I),
    re.compile(r"what.+driv.+scor", re.I),
    re.compile(r"why.+\d{2,3}\b", re.I),
]
_PATTERN = [
    re.compile(r"won.+like", re.I),
    re.compile(r"similar.+before", re.I),
    re.compile(r"have\s+we\s+(won|seen|closed)", re.I),
    re.compile(r"accounts?\s+like\s+this", re.I),
    re.compile(r"past.+(win|won|deal)", re.I),
    re.compile(r"historical.+pattern", re.I),
]


def classify_question(question: str) -> tuple[str, dict[str, str]]:
    """
    Returns (template_name, params_dict).
    template_name: "explain" | "compare" | "pattern" | "unknown"
    """
    ids = _ACCOUNT_RE.findall(question)
    # compare checked first — overlaps with explain "why" patterns
    for p in _COMPARE:
        if p.search(question):
            return "compare", {
                "lead_id_a": ids[0] if len(ids) > 0 else "",
                "lead_id_b": ids[1] if len(ids) > 1 else "",
            }
    for p in _EXPLAIN:
        if p.search(question):
            return "explain", {"lead_id": ids[0] if ids else ""}
    for p in _PATTERN:
        if p.search(question):
            return "pattern", {"lead_id": ids[0] if ids else ""}
    if ids:
        return "explain", {"lead_id": ids[0]}
    return "unknown", {}
