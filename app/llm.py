"""Thin local-LLM client over Ollama.

Kept deliberately small and swappable: the rest of the app talks to
`generate()` / `generate_json()` and never imports the ollama package
directly, so the runtime can be changed later without touching callers.
"""
from __future__ import annotations

import json
from typing import Optional

import ollama

from app.config import OLLAMA_HOST, LLM_MODEL

_client: Optional["ollama.Client"] = None


def _get_client() -> "ollama.Client":
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def generate(prompt: str, system: Optional[str] = None, temperature: float = 0.0) -> str:
    """Free-text generation. temperature=0 for deterministic, grounded output
    (we want the model to stick to the provided context, not be creative)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = _get_client().chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp["message"]["content"]


def generate_json(prompt: str, system: Optional[str] = None, temperature: float = 0.0) -> dict:
    """Constrained JSON generation via Ollama's format=json. Used for
    citations (Week 2) and structured extraction (Week 3). Returns a parsed
    dict; raises ValueError if the model emits invalid JSON."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = _get_client().chat(
        model=LLM_MODEL,
        messages=messages,
        format="json",
        options={"temperature": temperature},
    )
    content = resp["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n---\n{content}")
