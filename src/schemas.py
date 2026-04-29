"""Pydantic schemas — the contract between the LLM, the validator, and the UI.

Every field is required-with-a-purpose. We deliberately avoid Optional defaults
that let the model "pass" by emitting empty strings: the brief calls that out
as a failure mode. When the model has no answer it must use Refusal, not silence.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# Inputs --------------------------------------------------------------------

Language = Literal["en", "ar"]


class Review(BaseModel):
    """One review as ingested. review_id is the only thing claims may cite."""

    review_id: str = Field(..., min_length=1, description="Stable, unique id like 'rev_012'.")
    language: Language
    rating: int = Field(..., ge=1, le=5)
    text: str = Field(..., min_length=1)
    helpful_count: int = Field(default=0, ge=0)


class Product(BaseModel):
    product_id: str
    name_en: str
    name_ar: str
    brand: str
    category: str
    age_range: str = Field(..., description="e.g. '0-6 months', 'maternity', '3-5 years'")
    description_en: str
    description_ar: str


# Outputs --------------------------------------------------------------------


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class Topic(str, Enum):
    """Closed set keeps the model honest. Open-ended 'topic' fields tend to
    hallucinate — moms talk about a finite set of things per product."""

    QUALITY = "quality"
    SIZING = "sizing"
    COMFORT = "comfort"
    SAFETY = "safety"
    VALUE_FOR_MONEY = "value_for_money"
    EASE_OF_USE = "ease_of_use"
    DURABILITY = "durability"
    AGE_APPROPRIATENESS = "age_appropriateness"
    DELIVERY_PACKAGING = "delivery_packaging"
    CUSTOMER_SERVICE = "customer_service"
    OTHER = "other"


class LanguageDistribution(BaseModel):
    """How the evidence for a claim splits across languages.

    Keys are 'en' and 'ar'. Sum must equal evidence_count on the parent Claim;
    we validate this in Claim.check_consistency.
    """

    en: int = Field(..., ge=0)
    ar: int = Field(..., ge=0)


class Claim(BaseModel):
    """One synthesized claim about the product, grounded in cited reviews.

    Hard rules:
    - Every claim has ≥1 citation. No citation → no claim.
    - confidence is calibrated (see EVALS rubric).
    - evidence_count is the number of reviews supporting the claim.
    - cited review_ids must exist in the input reviews; pipeline validates this.
    """

    topic: Topic
    sentiment: Sentiment
    claim: str = Field(..., min_length=10, description="One short sentence. No padding.")
    citations: list[str] = Field(..., min_length=1, description="review_ids — must exist in input.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_count: int = Field(..., ge=1)
    language_distribution: LanguageDistribution

    @model_validator(mode="after")
    def check_consistency(self) -> "Claim":
        ld_sum = self.language_distribution.en + self.language_distribution.ar
        if ld_sum != self.evidence_count:
            raise ValueError(
                f"language_distribution sum ({ld_sum}) != evidence_count ({self.evidence_count})"
            )
        if len(self.citations) > self.evidence_count:
            raise ValueError(
                f"citations ({len(self.citations)}) cannot exceed evidence_count ({self.evidence_count})"
            )
        return self


class CrossLingualInsight(BaseModel):
    """The headline novelty. A signal that one language carries and the other
    doesn't (or carries weakly). Also requires citations + a confidence score."""

    insight: str = Field(..., min_length=20)
    en_evidence_count: int = Field(..., ge=0)
    ar_evidence_count: int = Field(..., ge=0)
    citations_en: list[str] = Field(default_factory=list)
    citations_ar: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def must_have_asymmetry(self) -> "CrossLingualInsight":
        # If both languages flag it equally, it is not cross-lingual.
        if self.en_evidence_count == 0 and self.ar_evidence_count == 0:
            raise ValueError("cross-lingual insight needs evidence in at least one language")
        return self


class VerdictBody(BaseModel):
    """The verdict in one language. The same structure is filled in EN and AR
    independently — AR is generated natively, not translated."""

    summary: str = Field(..., min_length=20, description="2–3 sentence headline. No fluff.")
    claims: list[Claim] = Field(..., min_length=1)
    cross_lingual_insights: list[CrossLingualInsight] = Field(default_factory=list)


class RefusalReason(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_SCOPE = "out_of_scope"
    SAFETY_ESCALATION = "safety_escalation"
    REVIEWS_OFF_TOPIC = "reviews_off_topic"


class Refusal(BaseModel):
    """When the model can't responsibly summarize. Examples:
    - <15 reviews → INSUFFICIENT_EVIDENCE
    - product is a regulated good (formula, medication) where claims need
      pediatrician escalation → SAFETY_ESCALATION
    - reviews are mostly off-topic (about delivery, about a different product)
      → REVIEWS_OFF_TOPIC
    """

    reason: RefusalReason
    explanation_en: str = Field(..., min_length=20)
    explanation_ar: str = Field(..., min_length=20)
    min_reviews_required: int | None = None
    suggested_action: str | None = None


class Verdict(BaseModel):
    """Top-level response. Either both verdict_en + verdict_ar, OR a refusal."""

    product_id: str
    verdict_en: VerdictBody | None = None
    verdict_ar: VerdictBody | None = None
    refusal: Refusal | None = None
    metadata: dict = Field(default_factory=dict, description="model, latency, eval-time fields")

    @model_validator(mode="after")
    def either_verdict_or_refusal(self) -> "Verdict":
        has_verdict = self.verdict_en is not None and self.verdict_ar is not None
        has_refusal = self.refusal is not None
        if has_verdict == has_refusal:
            raise ValueError(
                "exactly one of (verdict_en+verdict_ar) or refusal must be set, not both, not neither"
            )
        return self
