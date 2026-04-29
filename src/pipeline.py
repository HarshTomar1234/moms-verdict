"""Top-level orchestrator: load → refusal-check → generate EN → generate AR → ground → return.

Designed so the unit of work is `synthesize(product, reviews) -> Verdict` and
nothing else needs to know about LLMs, prompts, or grounding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.grounding import GroundingReport, check_for_refusal, ground_verdict_body
from src.llm import LLMClient
from src.prompts import build_verdict_messages_ar, build_verdict_messages_en
from src.schemas import Product, Refusal, Review, Verdict, VerdictBody


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_products(path: Path | None = None) -> dict[str, Product]:
    path = path or DATA_DIR / "products" / "products.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {p["product_id"]: Product(**p) for p in raw}


def load_reviews(product_id: str, path: Path | None = None) -> list[Review]:
    path = path or DATA_DIR / "reviews" / f"{product_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Review(**r) for r in raw]


# ── Body parsing (model output → VerdictBody) ────────────────────────────────


def _parse_body(payload: dict) -> VerdictBody:
    """The LLM returns a JSON object matching VerdictBody (claims + cross_lingual).
    Pydantic validates. Raises ValidationError if the model produced bad shape."""
    return VerdictBody(**payload)


# ── Synthesis ────────────────────────────────────────────────────────────────


def synthesize(
    product: Product,
    reviews: list[Review],
    *,
    llm: Optional[LLMClient] = None,
    skip_ar: bool = False,
) -> tuple[Verdict, dict]:
    """Returns (verdict, metadata).

    metadata is a dict for the eval runner: latencies, models, grounding reports.
    """
    meta: dict = {
        "n_reviews": len(reviews),
        "n_en": sum(1 for r in reviews if r.language == "en"),
        "n_ar": sum(1 for r in reviews if r.language == "ar"),
    }

    # 1. Deterministic refusal
    refusal = check_for_refusal(product, reviews)
    if refusal is not None:
        meta["refusal_path"] = "deterministic"
        return Verdict(product_id=product.product_id, refusal=refusal, metadata=meta), meta

    # 2. LLM synthesis
    if llm is None:
        llm = LLMClient()

    # EN
    body_en, grounding_en = _generate_grounded_body(
        llm, build_verdict_messages_en(product, reviews), reviews
    )
    meta["latency_ms_en"] = grounding_en["latency_ms"]
    meta["model_en"] = grounding_en["model"]
    meta["grounding_en"] = grounding_en["report"].__dict__

    if skip_ar:
        # Used by evals where we only want to score the EN side.
        return (
            Verdict(
                product_id=product.product_id,
                verdict_en=body_en,
                verdict_ar=body_en,  # placeholder; the schema requires both
                metadata=meta,
            ),
            meta,
        )

    # Brief pause between EN and AR to avoid TPM rate limits on free-tier providers.
    import time as _time; _time.sleep(8)

    # AR — independent prompt, native Arabic system message
    body_ar, grounding_ar = _generate_grounded_body(
        llm, build_verdict_messages_ar(product, reviews), reviews
    )
    meta["latency_ms_ar"] = grounding_ar["latency_ms"]
    meta["model_ar"] = grounding_ar["model"]
    meta["grounding_ar"] = grounding_ar["report"].__dict__

    return (
        Verdict(
            product_id=product.product_id,
            verdict_en=body_en,
            verdict_ar=body_ar,
            metadata=meta,
        ),
        meta,
    )


def _generate_grounded_body(
    llm: LLMClient,
    messages: list[dict],
    reviews: list[Review],
    max_attempts: int = 2,
) -> tuple[VerdictBody, dict]:
    """One LLM call → parse → ground. Retry once on schema or grounding failure.

    Why retry once and not more: if the model fails twice on the same prompt
    it's almost certainly a prompt or schema issue, not a transient one. Better
    to surface the failure than burn the free-tier quota on hopeful retries.
    """
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            payload, raw = llm.chat_json(messages, temperature=0.2)
            body = _parse_body(payload)
            grounded, report = ground_verdict_body(body, reviews)
            return grounded, {
                "latency_ms": raw.latency_ms,
                "model": raw.model,
                "report": report,
                "attempts": attempt + 1,
            }
        except (ValidationError, ValueError, KeyError) as e:
            last_error = e
            # On retry, append a corrective hint to the user message.
            if attempt < max_attempts - 1:
                hint = (
                    "\n\nIMPORTANT: your previous response failed validation: "
                    f"{type(e).__name__}: {str(e)[:300]}. "
                    "Output ONLY a single valid JSON object matching the schema. "
                    "Every claim MUST cite at least one review_id from the input."
                )
                messages = messages.copy()
                messages[-1] = {**messages[-1], "content": messages[-1]["content"] + hint}
    raise RuntimeError(f"Verdict synthesis failed after {max_attempts} attempts: {last_error}")
