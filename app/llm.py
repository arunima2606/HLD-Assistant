"""Local LLM client (Ollama) with a deterministic mock mode and a JSON-retry helper."""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from app.config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
NOT_FOUND = "Not found in the provided documents."
EMPTY_EXTRACTION = '{"components":[],"interfaces":[],"ports":[],"signals":[],"dependencies":[]}'


class LLMError(Exception):
    """LLM unreachable or returned an error."""


def _mock(prompt: str, task: str) -> str:
    """Deterministic canned output. Answers extractively from the first context block."""
    if task == "extract":
        return EMPTY_EXTRACTION
    m = re.search(r"Context:\n(\[[^\]\n]+\])\n(.*?)(?:\n\n|\nQuestion:)", prompt, re.S)
    if not m:
        return NOT_FOUND
    snippet = " ".join(m.group(2).split())[:300]
    return f"(mock) {snippet} {m.group(1)}"


def generate(prompt: str, *, task: str = "answer", json_mode: bool = False, system: str | None = None) -> str:
    """Return the model's text. task is 'answer' or 'extract' (used by the mock)."""
    s = get_settings()
    if s.mock_llm:
        return _mock(prompt, task)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload = {"model": s.llm_model, "messages": messages, "stream": False,
               "options": {"temperature": s.llm_temperature, "num_ctx": 4096}}
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(f"{s.ollama_host}/api/chat", json=payload, timeout=s.llm_timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.ConnectionError as exc:
        raise LLMError(f"Cannot reach Ollama at {s.ollama_host}. Start it with `ollama serve` and run "
                       f"`ollama pull {s.llm_model}` (or set MOCK_LLM=1).") from exc
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc


def parse_json(text: str) -> dict:
    """Parse JSON tolerant of markdown fences and surrounding prose."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(t[start:end + 1])


def generate_json(prompt: str, model_cls: type[T], *, retries: int = 1, task: str = "extract") -> T | None:
    """Generate -> parse -> validate. Retries once with a repair hint; returns None on failure."""
    attempt_prompt = prompt
    for attempt in range(retries + 1):
        raw = generate(attempt_prompt, task=task, json_mode=True)
        try:
            return model_cls.model_validate(parse_json(raw))
        except (ValueError, ValidationError) as exc:
            log.warning("Invalid JSON from LLM (attempt %d/%d): %s", attempt + 1, retries + 1, str(exc)[:200])
            attempt_prompt = (prompt + "\n\nYour previous reply was not valid JSON for the required shape. "
                              "Reply with ONLY the JSON object, nothing else.")
    return None
