"""Citation grounding + off-topic detection.

The model is allowed to make claims, but every claim must cite review_ids that
exist in the input. This module enforces that contract. Bad citations are not
silently dropped — we keep them visible so evals can catch a regressing prompt.

Off-topic detection is intentionally a simple keyword heuristic, not an LLM
call. Determinism beats sophistication for this check, and we document the
tradeoff in the README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas import (
    Claim,
    CrossLingualInsight,
    Product,
    Refusal,
    RefusalReason,
    Review,
    VerdictBody,
)


# ── Refusal triggers (deterministic, run before any LLM call) ─────────────────

MIN_REVIEWS_FOR_VERDICT = 15

# Categories where review-based verdicts are not safe — escalate to a professional.
REGULATED_CATEGORIES = {
    "baby formula",
    "infant formula",
    "formula",
    "medication",
    "medicine",
    "supplements",
    "vitamins",
    "prescription",
}

# Keyword sets for off-topic detection. Bilingual.
DELIVERY_KEYWORDS_EN = {
    "delivery", "shipping", "aramex", "dhl", "skynet", "courier",
    "package", "packaging", "box arrived", "took.*days", "late",
    "refund", "return", "returns process", "customer service",
    "cash on delivery", "wrong product", "dented", "crushed",
}
DELIVERY_KEYWORDS_AR = {
    "توصيل", "تأخر", "أرامكس", "ارامكس", "تأخير", "تغليف", "متأخر",
    "خدمة العملاء", "إرجاع", "ارجاع", "استرداد", "طرد", "صندوق",
}
PRODUCT_NOUN_HINT_EN = {
    "stroller", "diaper", "wheel", "fold", "carrycot", "seat", "leak",
    "absorb", "soft", "rash", "fit", "size", "bottle", "formula",
    "toy", "music", "light", "lights", "feed", "comfort",
}
PRODUCT_NOUN_HINT_AR = {
    "عربية", "العربية", "حفاضة", "حفاضات", "عجلات", "مقاس",
    "كاريكوت", "تسرب", "ناعم", "ناعمة", "حليب", "لعبة", "ولدي", "بنتي",
}


def review_is_off_topic(review: Review) -> bool:
    """Heuristic: a review is off-topic if it mentions delivery/service AND
    does not mention any product-relevant noun. We err on the side of NOT
    flagging — false positives here would suppress legitimate verdicts."""
    text = review.text.lower()
    if review.language == "en":
        delivery_hits = sum(1 for kw in DELIVERY_KEYWORDS_EN if re.search(kw, text))
        product_hits = sum(1 for kw in PRODUCT_NOUN_HINT_EN if kw in text)
    else:
        delivery_hits = sum(1 for kw in DELIVERY_KEYWORDS_AR if kw in text)
        product_hits = sum(1 for kw in PRODUCT_NOUN_HINT_AR if kw in text)
    return delivery_hits >= 1 and product_hits == 0


def off_topic_ratio(reviews: list[Review]) -> float:
    if not reviews:
        return 0.0
    return sum(1 for r in reviews if review_is_off_topic(r)) / len(reviews)


def is_regulated_category(product: Product) -> bool:
    cat = product.category.lower()
    return any(kw in cat for kw in REGULATED_CATEGORIES)


def check_for_refusal(product: Product, reviews: list[Review]) -> Refusal | None:
    """Deterministic refusal logic. Run BEFORE any LLM call.

    Order matters: safety > insufficient_evidence > off_topic. A regulated
    product with 5 reviews returns SAFETY_ESCALATION, not INSUFFICIENT_EVIDENCE.
    """
    if is_regulated_category(product):
        return Refusal(
            reason=RefusalReason.SAFETY_ESCALATION,
            explanation_en=(
                f"This product falls in a regulated category ({product.category}). "
                "We do not synthesise review-based verdicts for infant nutrition, medication, "
                "or supplements — please consult your pediatrician before purchase decisions."
            ),
            explanation_ar=(
                f"هذا المنتج من فئة منظمة ({product.category}). "
                "ما نقدّم خلاصة آراء عملاء في فئات تغذية الرضع والأدوية والمكمّلات — "
                "الرجاء استشارة طبيب الأطفال قبل اتخاذ قرار الشراء."
            ),
            suggested_action="Consult a pediatrician.",
        )

    if len(reviews) < MIN_REVIEWS_FOR_VERDICT:
        return Refusal(
            reason=RefusalReason.INSUFFICIENT_EVIDENCE,
            explanation_en=(
                f"Only {len(reviews)} review(s) available. We require at least "
                f"{MIN_REVIEWS_FOR_VERDICT} for a reliable verdict; a smaller "
                "sample isn't representative of how moms actually find this product."
            ),
            explanation_ar=(
                f"عدد التقييمات المتاحة {len(reviews)} فقط. نحتاج على الأقل "
                f"{MIN_REVIEWS_FOR_VERDICT} تقييم لتلخيص موثوق؛ "
                "العينة الصغيرة ما تمثل تجربة الأمهات الفعلية مع المنتج."
            ),
            min_reviews_required=MIN_REVIEWS_FOR_VERDICT,
        )

    ratio = off_topic_ratio(reviews)
    if ratio > 0.6:
        return Refusal(
            reason=RefusalReason.REVIEWS_OFF_TOPIC,
            explanation_en=(
                f"{int(ratio * 100)}% of reviews discuss delivery, packaging, or customer "
                "service rather than the product itself. We can't reliably synthesise "
                "a product verdict from off-topic feedback."
            ),
            explanation_ar=(
                f"{int(ratio * 100)}% من التقييمات تتحدث عن التوصيل والتغليف وخدمة "
                "العملاء وليس عن المنتج نفسه. ما نقدر نلخّص حكم موثوق عن المنتج "
                "من تقييمات بعيدة عن الموضوع."
            ),
        )

    return None


# ── Citation grounding (run AFTER LLM call) ───────────────────────────────────


@dataclass
class GroundingReport:
    valid_claims: int
    invalid_claims: int
    invalid_citations: list[tuple[str, str]]  # (claim, bad_citation)
    fixed_claims: int  # claims we kept after stripping invalid citations
    dropped_claims: int  # claims with NO valid citations left


def validate_claim_citations(
    claim: Claim, valid_review_ids: set[str]
) -> tuple[Claim | None, list[str]]:
    """Returns (cleaned_claim_or_None, invalid_citations).
    A claim with zero valid citations is dropped (None)."""
    valid_cites = [c for c in claim.citations if c in valid_review_ids]
    invalid_cites = [c for c in claim.citations if c not in valid_review_ids]

    if not valid_cites:
        return None, invalid_cites

    if len(valid_cites) == len(claim.citations):
        return claim, []

    # Some citations were invalid — keep the claim with valid ones, but reduce
    # confidence proportionally and update evidence_count if needed.
    new_evidence_count = max(claim.evidence_count - len(invalid_cites), len(valid_cites))
    # rebalance language_distribution best-effort: cap at evidence_count
    en, ar = claim.language_distribution.en, claim.language_distribution.ar
    if en + ar > new_evidence_count:
        # scale down preserving ratio
        total = en + ar
        en = round(en * new_evidence_count / total)
        ar = new_evidence_count - en

    cleaned = claim.model_copy(
        update={
            "citations": valid_cites,
            "confidence": round(claim.confidence * 0.7, 3),
            "evidence_count": new_evidence_count,
            "language_distribution": {"en": en, "ar": ar},
        }
    )
    return cleaned, invalid_cites


def validate_insight_citations(
    insight: CrossLingualInsight, valid_review_ids: set[str]
) -> CrossLingualInsight | None:
    valid_en = [c for c in insight.citations_en if c in valid_review_ids]
    valid_ar = [c for c in insight.citations_ar if c in valid_review_ids]
    if not valid_en and not valid_ar:
        return None
    return insight.model_copy(
        update={
            "citations_en": valid_en,
            "citations_ar": valid_ar,
            "confidence": round(insight.confidence * (0.7 if (len(valid_en) < len(insight.citations_en) or len(valid_ar) < len(insight.citations_ar)) else 1.0), 3),
        }
    )


def ground_verdict_body(
    body: VerdictBody, reviews: list[Review]
) -> tuple[VerdictBody, GroundingReport]:
    valid_ids = {r.review_id for r in reviews}
    cleaned_claims: list[Claim] = []
    all_invalid: list[tuple[str, str]] = []
    fixed = 0
    dropped = 0

    for claim in body.claims:
        cleaned, invalid_cites = validate_claim_citations(claim, valid_ids)
        if cleaned is None:
            dropped += 1
            for ic in invalid_cites:
                all_invalid.append((claim.claim, ic))
            continue
        if invalid_cites:
            fixed += 1
            for ic in invalid_cites:
                all_invalid.append((claim.claim, ic))
        cleaned_claims.append(cleaned)

    cleaned_insights = []
    for insight in body.cross_lingual_insights:
        ci = validate_insight_citations(insight, valid_ids)
        if ci is not None:
            cleaned_insights.append(ci)

    if not cleaned_claims:
        # If the model produced zero verifiable claims we can't return a body —
        # the pipeline should escalate to refusal.
        raise ValueError("No claims survived citation validation; pipeline should refuse.")

    cleaned_body = body.model_copy(
        update={"claims": cleaned_claims, "cross_lingual_insights": cleaned_insights}
    )
    report = GroundingReport(
        valid_claims=len(cleaned_claims),
        invalid_claims=dropped,
        invalid_citations=all_invalid,
        fixed_claims=fixed,
        dropped_claims=dropped,
    )
    return cleaned_body, report
