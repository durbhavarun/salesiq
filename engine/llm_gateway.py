"""
engine/llm_gateway.py
─────────────────────
Single gateway for ALL LLM calls in SalesIQ.
No code outside this file may call any LLM directly.

Pipeline (every call):
  1. PII scrub  — strip personal data before leaving machine
  2. Cache      — return instantly if input seen before
  3. Groq       — primary provider (10s timeout)
  4. Ollama     — local fallback (10s timeout)
  5. Fallback   — structured error dict, never crash
  6. Validate   — confirm JSON matches expected schema
  7. Log        — provider, latency, cache hit/miss

CRITICAL: LLM never touches scores.
Signal types MUST match INTENT_SIGNAL_TYPES in engine/icp_weights.py exactly.
"""
from __future__ import annotations
import hashlib, json, logging, os, re, time
from typing import Any
import requests

logger = logging.getLogger(__name__)

# ── PII patterns ──────────────────────────────────────────────
_COMPILED_PII = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL]"),
    (re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]

def _scrub_pii(text: str) -> str:
    for pat, rep in _COMPILED_PII:
        text = pat.sub(rep, text)
    return text

# ── In-memory cache ───────────────────────────────────────────
_CACHE: dict[str, Any] = {}

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()

# ── Signal types — mirrors INTENT_SIGNAL_TYPES from icp_weights.py ──
_VALID_SIGNAL_TYPES = {
    "credit_rating_change", "expansion_announcement",
    "leadership_change", "ma_activity", "equipment_purchase",
}

# ── Stable-prefix prompts ─────────────────────────────────────
_EXTRACTION_PREFIX = (
    "You are a financial signal extractor for a corporate lending platform.\n"
    "Extract structured signals from the input text below.\n"
    "Return ONLY valid JSON — no markdown, no explanation:\n"
    '{"signals": [{"signal_type": "<type>", "signal_value": "<value>", '
    '"extraction_confidence": <0.0-1.0>, "evidence_snippet": "<verbatim quote max 200 chars>"}]}\n'
    "Valid signal_type values (ONLY these five, exactly as written):\n"
    "  credit_rating_change, expansion_announcement, leadership_change, "
    "ma_activity, equipment_purchase\n"
    "Rules: return empty list [] if nothing relevant; do not invent.\n\n"
    "Input text:\n"
)

_GRAPHRAG_PREFIX = (
    "You are an AI assistant for a corporate lending platform.\n"
    "Answer the question using ONLY the evidence in the subgraph context below.\n"
    "Do not use outside knowledge. If evidence is insufficient, say so.\n"
    "Return ONLY valid JSON — no markdown:\n"
    '{"answer": "<plain-language answer>", "confidence": "<HIGH|MEDIUM|LOW>", '
    '"evidence_used": ["<fact from context>"]}\n\n'
    "Subgraph context:\n"
)

# ── Provider calls ────────────────────────────────────────────
def _call_groq(prompt: str) -> dict | None:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 1024},
            timeout=10,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception as e:
        logger.warning("Groq failed: %s", e)
        return None

def _call_ollama(prompt: str) -> dict | None:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        r = requests.post(f"{base}/api/generate",
            json={"model": os.getenv("OLLAMA_MODEL", "llama3"),
                  "prompt": prompt, "stream": False, "format": "json"},
            timeout=10)
        r.raise_for_status()
        return json.loads(r.json().get("response", "{}"))
    except Exception as e:
        logger.warning("Ollama failed: %s", e)
        return None

def _call_watsonx(prompt: str) -> dict | None:
    """Stub — ibm-watsonx-ai not in requirements.txt (Python 3.13 conflict)."""
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        api_key = os.getenv("WATSONX_API_KEY", "")
        project_id = os.getenv("WATSONX_PROJECT_ID", "")
        if not api_key or not project_id:
            return None
        model = ModelInference(
            model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2"),
            credentials={"apikey": api_key, "url": "https://us-south.ml.cloud.ibm.com"},
            project_id=project_id)
        raw = model.generate_text(prompt=prompt).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning("Watsonx failed: %s", e)
        return None

def _run_chain(prompt: str) -> tuple[dict | None, str]:
    pref = os.getenv("SALESIQ_LLM_PROVIDER", "groq").lower()
    if pref == "none":
        return None, "none"
    if pref == "watsonx":
        r = _call_watsonx(prompt)
        if r is not None:
            return r, "watsonx"
    if pref in ("groq", "watsonx"):
        r = _call_groq(prompt)
        if r is not None:
            return r, "groq"
    if pref in ("groq", "ollama", "watsonx"):
        r = _call_ollama(prompt)
        if r is not None:
            return r, "ollama"
    return None, "none"

# ── Schema validators ─────────────────────────────────────────
def _validate_schema(response: Any) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "not a dict"
    if "signals" not in response:
        return False, "missing 'signals'"
    if not isinstance(response["signals"], list):
        return False, "'signals' not a list"
    for i, item in enumerate(response["signals"]):
        if not isinstance(item, dict):
            return False, f"signals[{i}] not a dict"
        for f, t in [("signal_type",str),("signal_value",str),
                     ("extraction_confidence",float),("evidence_snippet",str)]:
            if f not in item:
                return False, f"signals[{i}] missing '{f}'"
            if t is float and isinstance(item[f], int):
                item[f] = float(item[f])
            elif not isinstance(item[f], t):
                return False, f"signals[{i}]['{f}'] wrong type"
        if item["signal_type"] not in _VALID_SIGNAL_TYPES:
            return False, f"unknown signal_type '{item['signal_type']}'"
        c = item["extraction_confidence"]
        if not (0.0 <= c <= 1.0):
            return False, f"confidence {c} out of range"
    return True, "ok"

def _validate_graphrag(response: Any) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "not a dict"
    for f in ("answer", "confidence", "evidence_used"):
        if f not in response:
            return False, f"missing '{f}'"
    if response["confidence"] not in ("HIGH", "MEDIUM", "LOW"):
        return False, f"invalid confidence '{response['confidence']}'"
    if not isinstance(response["evidence_used"], list):
        return False, "'evidence_used' not a list"
    return True, "ok"

# ── Public API ────────────────────────────────────────────────
def extract_signals(text: str) -> dict[str, Any]:
    """Extract signals from unstructured text. Never raises."""
    t0 = time.monotonic()
    prompt = _EXTRACTION_PREFIX + _scrub_pii(text)
    key = _cache_key(prompt)
    if key in _CACHE:
        lat = (time.monotonic()-t0)*1000
        return {**_CACHE[key], "provider":"cache", "cache_hit":True, "latency_ms":round(lat,1)}
    raw, prov = _run_chain(prompt)
    lat = (time.monotonic()-t0)*1000
    if raw is None:
        return {"signals":[], "provider":"none", "cache_hit":False,
                "latency_ms":round(lat,1), "valid":False,
                "error":"All LLM providers unavailable or disabled"}
    valid, reason = _validate_schema(raw)
    result = {"signals":raw.get("signals",[]), "provider":prov, "cache_hit":False,
              "latency_ms":round(lat,1), "valid":valid,
              "error":None if valid else f"Schema failed: {reason}"}
    if valid:
        _CACHE[key] = {"signals":result["signals"], "valid":True, "error":None}
    return result

def generate_graphrag_answer(question: str, context: str) -> dict[str, Any]:
    """Generate grounded answer constrained to context. Never raises."""
    t0 = time.monotonic()
    prompt = _GRAPHRAG_PREFIX + context + f"\n\nQuestion: {_scrub_pii(question)}"
    key = _cache_key(prompt)
    if key in _CACHE:
        lat = (time.monotonic()-t0)*1000
        return {**_CACHE[key], "provider":"cache", "cache_hit":True, "latency_ms":round(lat,1)}
    raw, prov = _run_chain(prompt)
    lat = (time.monotonic()-t0)*1000
    if raw is None:
        return {"answer":"LLM unavailable — enable Groq or Ollama for answers.",
                "confidence":"LOW", "evidence_used":[], "provider":"none",
                "cache_hit":False, "latency_ms":round(lat,1), "valid":False,
                "error":"All LLM providers unavailable or disabled"}
    valid, reason = _validate_graphrag(raw)
    result = {"answer":raw.get("answer",""), "confidence":raw.get("confidence","LOW"),
              "evidence_used":raw.get("evidence_used",[]), "provider":prov,
              "cache_hit":False, "latency_ms":round(lat,1), "valid":valid,
              "error":None if valid else f"Schema failed: {reason}"}
    if valid:
        _CACHE[key] = {"answer":result["answer"], "confidence":result["confidence"],
                       "evidence_used":result["evidence_used"], "valid":True, "error":None}
    return result
