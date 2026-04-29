"""LLM client. Checks providers in this priority order:
  1. GROQ_API_KEY     → groq.com (free, fast, reliable, no card needed)
  2. GEMINI_API_KEY   → Google AI Studio (free but quota can be strict)
  3. OPENROUTER_API_KEY → OpenRouter free models (rate-limited at peak)

Get Groq key in 2 min: console.groq.com → Login with Google → API Keys → Create key
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

PROVIDERS = {
    "groq":       "https://api.groq.com/openai/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
}

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # primary — 70B, 128k context, best quality for verdict
    # No small-model fallback: 8B context windows reject our 50-review prompts.
    # If rate-limited, the runner adds inter-case sleep so the TPM window resets.
]

OPENROUTER_FALLBACKS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free",
]


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = httpx.Client(timeout=180.0)

        groq_key = os.environ.get("GROQ_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        or_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if groq_key:
            self._provider = "groq"
            self.api_key = groq_key
            self.model = model or os.environ.get("GEN_MODEL", "llama-3.3-70b-versatile")
            self._base_url = PROVIDERS["groq"]
        elif gemini_key:
            self._provider = "gemini"
            self.api_key = gemini_key
            self.model = model or os.environ.get("GEN_MODEL", "gemini-2.0-flash")
            self._base_url = PROVIDERS["gemini"]
        elif or_key:
            self._provider = "openrouter"
            self.api_key = or_key
            self.model = model or os.environ.get("GEN_MODEL", OPENROUTER_FALLBACKS[0])
            self._base_url = PROVIDERS["openrouter"]
        else:
            raise LLMError(
                "No API key found.\n"
                "Fastest free option: console.groq.com → Login with Google → API Keys → Create key\n"
                "Then add GROQ_API_KEY=your_key to .env"
            )

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self._provider == "openrouter":
            h["HTTP-Referer"] = "https://github.com/harshtomar/moms-verdict"
            h["X-Title"] = "Moms Verdict - Mumzworld AI Intern"
        return h

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        if self._provider in ("groq", "gemini"):
            models_to_try = [model or self.model]
            if self._provider == "groq":
                models_to_try += [m for m in GROQ_MODELS if m != models_to_try[0]]
        else:
            models_to_try = [model or self.model] + [
                m for m in OPENROUTER_FALLBACKS if m != (model or self.model)
            ]

        last_error: Exception | None = None
        for m in models_to_try:
            try:
                return self._call_once(m, messages, temperature, json_mode, max_tokens)
            except LLMError as e:
                msg = str(e)
                if "429" in msg or "404" in msg:
                    last_error = e
                    continue
                raise

        raise LLMError(
            f"All models exhausted for provider={self._provider}. Last error: {last_error}"
        )

    def _call_once(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool,
        max_tokens: int,
    ) -> LLMResponse:
        import time

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Gemini's compat endpoint does not support response_format reliably.
        if json_mode and self._provider != "gemini":
            body["response_format"] = {"type": "json_object"}

        t0 = time.time()
        resp = self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )
        latency_ms = int((time.time() - t0) * 1000)

        if resp.status_code == 429:
            raise LLMError(f"429 rate-limited: {model} | {resp.text[:200]}")
        if resp.status_code == 404:
            raise LLMError(f"404 not found: {model} | {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code} from {model}: {resp.text[:400]}")

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response shape: {data}") from e

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=model,
            latency_ms=latency_ms,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> tuple[dict, LLMResponse]:
        kwargs.setdefault("json_mode", True)
        resp = self.chat(messages, **kwargs)
        parsed = parse_json_lenient(resp.text)
        return parsed, resp


def parse_json_lenient(text: str) -> dict:
    text = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])
