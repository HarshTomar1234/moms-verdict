"""Graders. Each grader returns a dict {passed: bool, score: float (0-1), notes: str}.

Grader principles:
- Cheap & deterministic graders run first. LLM-based graders only for what
  can't be checked otherwise (e.g. AR fluency).
- Graders never crash the runner — they catch their own exceptions and return
  passed=False with notes. The runner reports honestly.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.llm import LLMClient
from src.prompts import AR_FLUENCY_GRADER_SYSTEM, AR_FLUENCY_GRADER_USER_TEMPLATE
from src.schemas import (
    Refusal,
    RefusalReason,
    Topic,
    Verdict,
)

from evals.test_cases import ExpectedRefusal, ExpectedVerdict, TestCase


GraderResult = dict[str, Any]


def grade_refusal(case: TestCase, verdict: Verdict) -> GraderResult:
    expected = case.expected
    if not isinstance(expected, ExpectedRefusal):
        return {"name": "refusal_correctness", "skipped": True}

    if verdict.refusal is None:
        return {
            "name": "refusal_correctness",
            "passed": False,
            "score": 0.0,
            "notes": f"expected refusal ({expected.reason.value}); model produced a verdict instead",
        }
    if verdict.refusal.reason != expected.reason:
        return {
            "name": "refusal_correctness",
            "passed": False,
            "score": 0.5,
            "notes": (
                f"refused but with wrong reason: expected {expected.reason.value}, "
                f"got {verdict.refusal.reason.value}"
            ),
        }
    return {
        "name": "refusal_correctness",
        "passed": True,
        "score": 1.0,
        "notes": f"refused correctly with {verdict.refusal.reason.value}",
    }


def grade_topics_present(case: TestCase, verdict: Verdict) -> GraderResult:
    expected = case.expected
    if not isinstance(expected, ExpectedVerdict):
        return {"name": "topics_present", "skipped": True}
    if verdict.refusal:
        return {
            "name": "topics_present",
            "passed": False,
            "score": 0.0,
            "notes": "expected verdict; got refusal",
        }
    seen_topics = {c.topic for c in verdict.verdict_en.claims}
    missing = [t.value for t in expected.must_have_topics if t not in seen_topics]
    if missing:
        return {
            "name": "topics_present",
            "passed": False,
            "score": 1.0 - (len(missing) / max(len(expected.must_have_topics), 1)),
            "notes": f"missing required topics: {missing}",
        }
    return {
        "name": "topics_present",
        "passed": True,
        "score": 1.0,
        "notes": f"all {len(expected.must_have_topics)} required topics covered",
    }


def grade_cross_lingual(case: TestCase, verdict: Verdict) -> GraderResult:
    expected = case.expected
    if not isinstance(expected, ExpectedVerdict):
        return {"name": "cross_lingual", "skipped": True}
    if verdict.refusal:
        return {"name": "cross_lingual", "skipped": True, "notes": "refusal — N/A"}

    insights = verdict.verdict_en.cross_lingual_insights

    # Must-have cross-lingual insight on a specific topic
    if expected.must_have_cross_lingual_topic is not None:
        if not insights:
            return {
                "name": "cross_lingual",
                "passed": False,
                "score": 0.0,
                "notes": f"expected cross-lingual insight on {expected.must_have_cross_lingual_topic.value}; got none",
            }
        # We don't enforce that the insight has a `topic` field (it doesn't),
        # so we check the language asymmetry direction instead.
        if expected.cross_lingual_dominant_lang == "ar":
            if not any(i.ar_evidence_count > i.en_evidence_count for i in insights):
                return {
                    "name": "cross_lingual",
                    "passed": False,
                    "score": 0.3,
                    "notes": "insight present but AR-dominance not surfaced",
                }
        elif expected.cross_lingual_dominant_lang == "en":
            if not any(i.en_evidence_count > i.ar_evidence_count for i in insights):
                return {
                    "name": "cross_lingual",
                    "passed": False,
                    "score": 0.3,
                    "notes": "insight present but EN-dominance not surfaced",
                }

    # Must-NOT-have insights for symmetric topics — best-effort: if there are
    # insights and one of them is on a forbidden topic-keyword, flag.
    forbidden = expected.no_cross_lingual_on_topics
    if forbidden:
        for ins in insights:
            for t in forbidden:
                if t.value.replace("_", " ") in ins.insight.lower():
                    return {
                        "name": "cross_lingual",
                        "passed": False,
                        "score": 0.5,
                        "notes": f"insight on a symmetric topic ({t.value}) shouldn't have been raised",
                    }

    return {
        "name": "cross_lingual",
        "passed": True,
        "score": 1.0,
        "notes": "cross-lingual structure as expected",
    }


def grade_calibration(case: TestCase, verdict: Verdict) -> GraderResult:
    expected = case.expected
    if not isinstance(expected, ExpectedVerdict) or verdict.refusal:
        return {"name": "calibration", "skipped": True}

    high_conf = [c for c in verdict.verdict_en.claims if c.confidence >= 0.85]
    miscalibrated = [
        c for c in high_conf if c.evidence_count < expected.high_confidence_min_evidence
    ]
    if miscalibrated:
        return {
            "name": "calibration",
            "passed": False,
            "score": 1.0 - (len(miscalibrated) / max(len(high_conf), 1)),
            "notes": (
                f"{len(miscalibrated)}/{len(high_conf)} high-confidence claims "
                f"have <{expected.high_confidence_min_evidence} pieces of evidence"
            ),
        }
    return {
        "name": "calibration",
        "passed": True,
        "score": 1.0,
        "notes": f"all {len(high_conf)} high-confidence claims well-calibrated",
    }


def grade_grounding(case: TestCase, verdict: Verdict, valid_review_ids: set[str]) -> GraderResult:
    """Every cited review_id must be in the input set. The pipeline already
    strips invalid citations, so this is a regression check on that."""
    if verdict.refusal:
        return {"name": "grounding", "skipped": True}
    bad: list[tuple[str, str]] = []
    for body_label, body in [("en", verdict.verdict_en), ("ar", verdict.verdict_ar)]:
        for c in body.claims:
            for cite in c.citations:
                if cite not in valid_review_ids:
                    bad.append((body_label, cite))
    if bad:
        return {
            "name": "grounding",
            "passed": False,
            "score": 0.0,
            "notes": f"invalid citations leaked through grounding pass: {bad[:5]}",
        }
    return {"name": "grounding", "passed": True, "score": 1.0, "notes": "all citations valid"}


# ── LLM-based AR fluency grader ───────────────────────────────────────────────


def grade_ar_fluency(case: TestCase, verdict: Verdict, llm: LLMClient | None) -> GraderResult:
    expected = case.expected
    if (
        not isinstance(expected, ExpectedVerdict)
        or not expected.grader_ar_fluency
        or verdict.refusal
        or llm is None
    ):
        return {"name": "ar_fluency", "skipped": True}

    # Concatenate the AR summary + claim texts as the sample.
    body = verdict.verdict_ar
    sample = body.summary + "\n" + "\n".join(c.claim for c in body.claims)

    grader_model = os.environ.get("GRADER_MODEL", "deepseek/deepseek-chat-v3:free")

    try:
        payload, raw = llm.chat_json(
            messages=[
                {"role": "system", "content": AR_FLUENCY_GRADER_SYSTEM},
                {
                    "role": "user",
                    "content": AR_FLUENCY_GRADER_USER_TEMPLATE.format(ar_text=sample),
                },
            ],
            model=grader_model,
            temperature=0.0,
        )
    except Exception as e:
        return {
            "name": "ar_fluency",
            "passed": False,
            "score": 0.0,
            "notes": f"grader call failed: {type(e).__name__}: {str(e)[:200]}",
        }

    score_int = int(payload.get("score", 0))
    label = payload.get("label", "unknown")
    issues = payload.get("issues", [])

    # Pass threshold: 3/5 (mixed or better). Below = translated-feel.
    passed = score_int >= 3
    return {
        "name": "ar_fluency",
        "passed": passed,
        "score": score_int / 5.0,
        "notes": f"score={score_int}/5 ({label}). issues: {issues[:3]}",
        "grader_model": grader_model,
    }
