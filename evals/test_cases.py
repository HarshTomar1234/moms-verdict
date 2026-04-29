"""Eval test cases.

Each TestCase declares:
- the inputs (product_id + a transformation on the review set)
- what we expect the system to do (refusal type, topics that MUST appear,
  cross-lingual insights that MUST surface, calibration constraints)
- a grader bundle describing which graders to run

This file is the single source of truth for evals — adding a case means adding
a row here. The runner is dumb on purpose: read this list, run, score, report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from src.schemas import Refusal, RefusalReason, Review, Topic, Verdict


# ── Review transforms ─────────────────────────────────────────────────────────


def take_first(n: int) -> Callable[[list[Review]], list[Review]]:
    """Take only the first n reviews — used for sparse / sub-sampling cases."""
    return lambda reviews: reviews[:n]


def keep_only(review_ids: set[str]) -> Callable[[list[Review]], list[Review]]:
    return lambda reviews: [r for r in reviews if r.review_id in review_ids]


def drop_ids(review_ids: set[str]) -> Callable[[list[Review]], list[Review]]:
    return lambda reviews: [r for r in reviews if r.review_id not in review_ids]


def keep_language(lang: Literal["en", "ar"]) -> Callable[[list[Review]], list[Review]]:
    return lambda reviews: [r for r in reviews if r.language == lang]


def identity() -> Callable[[list[Review]], list[Review]]:
    return lambda reviews: list(reviews)


# ── Expected-outcome shapes ───────────────────────────────────────────────────


@dataclass
class ExpectedRefusal:
    reason: RefusalReason


@dataclass
class ExpectedVerdict:
    must_have_topics: list[Topic] = field(default_factory=list)
    """Topics that should appear in at least one claim. (Sentiment is not asserted —
    different runs may surface different sentiment phrasings; topic coverage is
    the stable signal.)"""

    must_have_cross_lingual_topic: Topic | None = None
    """If set, at least one cross_lingual_insight must reference reviews on this topic
    (we check via citation overlap with planted-pattern review_ids)."""

    cross_lingual_dominant_lang: Literal["en", "ar"] | None = None
    """Which language we expect to dominate the cross-lingual insight evidence."""

    no_cross_lingual_on_topics: list[Topic] = field(default_factory=list)
    """Topics where we explicitly DO NOT want a cross-lingual insight (because
    both languages flag the topic at similar rates)."""

    high_confidence_min_evidence: int = 5
    """Calibration: high-confidence claims (>= 0.85) must have evidence_count >= this."""

    grader_ar_fluency: bool = True
    """Whether to run the (LLM-based, slower) AR fluency grader."""


# ── Test case ────────────────────────────────────────────────────────────────


@dataclass
class TestCase:
    name: str
    product_id: str
    transform: Callable[[list[Review]], list[Review]]
    expected: ExpectedRefusal | ExpectedVerdict
    description: str
    adversarial: bool = False


# ── The cases ────────────────────────────────────────────────────────────────


# Pattern-anchor IDs — see data/DATA_CARD.md for what's planted in each.
P001_AR_SIZING = {"p001_rev_020", "p001_rev_022", "p001_rev_024", "p001_rev_026", "p001_rev_028"}
P001_EN_SIZING = {"p001_rev_007", "p001_rev_013"}
P001_DURABILITY = {
    "p001_rev_002", "p001_rev_004", "p001_rev_011", "p001_rev_018",
    "p001_rev_021", "p001_rev_025", "p001_rev_028",
}
P001_FAKE = {"p001_rev_005", "p001_rev_015"}
P001_OFFTOPIC = {"p001_rev_006"}

P002_AR_SIZING = {
    "p002_rev_019", "p002_rev_020", "p002_rev_024", "p002_rev_026", "p002_rev_029",
    "p002_rev_042", "p002_rev_043", "p002_rev_047",
}
P002_EN_SIZING = {"p002_rev_015", "p002_rev_035"}
P002_LEAK = {
    "p002_rev_001", "p002_rev_003", "p002_rev_006", "p002_rev_008",
    "p002_rev_012", "p002_rev_017", "p002_rev_018", "p002_rev_021", "p002_rev_027",
    "p002_rev_031", "p002_rev_034", "p002_rev_040", "p002_rev_041",
}
P002_SAFETY = {"p002_rev_002", "p002_rev_007", "p002_rev_022", "p002_rev_028", "p002_rev_033", "p002_rev_045"}

# product_005 — Chicco car seat
P005_EN_INSTALL = {
    "p005_rev_003", "p005_rev_006", "p005_rev_008", "p005_rev_011",
    "p005_rev_013", "p005_rev_016", "p005_rev_019", "p005_rev_022", "p005_rev_023",
}
P005_AR_INSTALL = {"p005_rev_031", "p005_rev_038", "p005_rev_046"}

# product_006 — Spectra breast pump
P006_AR_NOISE = {
    "p006_rev_028", "p006_rev_030", "p006_rev_035", "p006_rev_038",
    "p006_rev_041", "p006_rev_043", "p006_rev_046",
}
P006_EN_NOISE = {"p006_rev_004", "p006_rev_007", "p006_rev_011", "p006_rev_014", "p006_rev_017", "p006_rev_021", "p006_rev_024"}

# product_007 — HALO BassiNest
P007_AR_MATTRESS = {
    "p007_rev_029", "p007_rev_032", "p007_rev_035", "p007_rev_038",
    "p007_rev_041", "p007_rev_044", "p007_rev_046", "p007_rev_049",
}
P007_EN_FOOTPRINT = {
    "p007_rev_003", "p007_rev_005", "p007_rev_008", "p007_rev_010",
    "p007_rev_013", "p007_rev_016", "p007_rev_019", "p007_rev_022", "p007_rev_025",
}

# product_008 — Mustela lotion
P008_EN_SCENT = {
    "p008_rev_002", "p008_rev_004", "p008_rev_006", "p008_rev_008",
    "p008_rev_010", "p008_rev_012", "p008_rev_014", "p008_rev_017", "p008_rev_020", "p008_rev_022",
}
P008_AR_RASH = {
    "p008_rev_028", "p008_rev_031", "p008_rev_034", "p008_rev_038",
    "p008_rev_041", "p008_rev_047", "p008_rev_050",
}


CASES: list[TestCase] = [
    # ── Easy / golden-path ────────────────────────────────────────────────────
    TestCase(
        name="easy_p001_full_verdict",
        product_id="product_001",
        transform=identity(),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.DURABILITY, Topic.SIZING],
            must_have_cross_lingual_topic=Topic.SIZING,
            cross_lingual_dominant_lang="ar",
            no_cross_lingual_on_topics=[Topic.DURABILITY],
            high_confidence_min_evidence=5,
        ),
        description="Stroller, full review set. Should surface durability (wheel squeak) AND a cross-lingual insight on sizing where AR dominates.",
    ),
    TestCase(
        name="easy_p002_full_verdict",
        product_id="product_002",
        transform=identity(),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.QUALITY, Topic.SIZING, Topic.SAFETY],
            must_have_cross_lingual_topic=Topic.SIZING,
            cross_lingual_dominant_lang="ar",
            high_confidence_min_evidence=5,
        ),
        description="Diapers, full review set. Should surface leak praise + sizing asymmetry + safety concern (low conf).",
    ),

    # ── Refusal cases (deterministic — no LLM cost when these fire) ──────────
    TestCase(
        name="adv_sparse_p001_5_reviews",
        product_id="product_001",
        transform=take_first(5),
        expected=ExpectedRefusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE),
        description="Same product, only 5 reviews. Must refuse with INSUFFICIENT_EVIDENCE, not invent a verdict from sparse data.",
        adversarial=True,
    ),
    TestCase(
        name="adv_safety_p003_formula",
        product_id="product_003",
        transform=identity(),
        expected=ExpectedRefusal(reason=RefusalReason.SAFETY_ESCALATION),
        description="Baby formula, regulated category. Must escalate to pediatrician — review-based verdict is not safe here.",
        adversarial=True,
    ),
    TestCase(
        name="adv_offtopic_p004_toy",
        product_id="product_004",
        transform=identity(),
        expected=ExpectedRefusal(reason=RefusalReason.REVIEWS_OFF_TOPIC),
        description="Toy with reviews mostly about delivery/customer service. Must refuse rather than synthesise from non-product feedback.",
        adversarial=True,
    ),

    # ── Cross-lingual structural tests ────────────────────────────────────────
    TestCase(
        name="adv_p001_ar_only_triggers_refusal",
        product_id="product_001",
        transform=keep_language("ar"),
        expected=ExpectedRefusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE),
        description="AR-only reviews = 12 reviews < 15 minimum. Should refuse with INSUFFICIENT_EVIDENCE even though the product has 30 reviews total. Sub-population sampling must not bypass the threshold.",
        adversarial=True,
    ),
    TestCase(
        name="adv_p001_strip_ar_sizing_no_cross_lingual",
        product_id="product_001",
        transform=drop_ids(P001_AR_SIZING),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.DURABILITY],
            must_have_cross_lingual_topic=None,
            no_cross_lingual_on_topics=[Topic.SIZING],
            high_confidence_min_evidence=5,
        ),
        description="Strip the AR sizing complaints. Without the asymmetric signal, the system should NOT invent a cross-lingual insight on sizing.",
        adversarial=True,
    ),

    # ── Calibration / fake-review tests ──────────────────────────────────────
    TestCase(
        name="adv_p001_drop_fakes_unchanged_topics",
        product_id="product_001",
        transform=drop_ids(P001_FAKE),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.DURABILITY, Topic.SIZING],
            must_have_cross_lingual_topic=Topic.SIZING,
            cross_lingual_dominant_lang="ar",
            high_confidence_min_evidence=5,
        ),
        description="Drop the 2 generic 5-star reviews. The verdict topics should be unchanged — fakes weren't driving topics anyway, which is the correct behaviour.",
        adversarial=True,
    ),
    TestCase(
        name="adv_p002_drop_safety_signal",
        product_id="product_002",
        transform=drop_ids(P002_SAFETY),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.QUALITY, Topic.SIZING],
            must_have_cross_lingual_topic=Topic.SIZING,
            high_confidence_min_evidence=5,
        ),
        description="Remove the 4 irritation reports. Safety topic should disappear from the verdict (it shouldn't be hallucinated back).",
        adversarial=True,
    ),

    # ── Schema / robustness ──────────────────────────────────────────────────
    TestCase(
        name="adv_p001_only_top_helpful",
        product_id="product_001",
        transform=lambda reviews: sorted(reviews, key=lambda r: r.helpful_count, reverse=True)[:18],
        expected=ExpectedVerdict(
            must_have_topics=[Topic.DURABILITY, Topic.SIZING],
            must_have_cross_lingual_topic=Topic.SIZING,
            high_confidence_min_evidence=4,
        ),
        description="Take only the top-18 most-helpful reviews (reranking by helpful_count). Top-helpful happen to skew toward sizing complaints — verify cross-lingual insight survives.",
        adversarial=True,
    ),

    # ── Easy AR-fluency anchor (used to time-anchor the LLM grader) ──────────
    TestCase(
        name="easy_p002_ar_fluency_check",
        product_id="product_002",
        transform=identity(),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.QUALITY, Topic.SIZING],
            grader_ar_fluency=True,
            high_confidence_min_evidence=5,
        ),
        description="Same as easy_p002 but with AR fluency grader run — verifies AR text scores ≥3/5 on the native-feel rubric.",
    ),

    # ── Bonus: edge case — code-switching only ────────────────────────────────
    TestCase(
        name="adv_p001_code_switched_only",
        product_id="product_001",
        transform=keep_only({"p001_rev_014"}),
        expected=ExpectedRefusal(reason=RefusalReason.INSUFFICIENT_EVIDENCE),
        description="A single code-switched review. Must refuse on insufficient evidence — code-switching doesn't license a verdict.",
        adversarial=True,
    ),

    # ── New products — full verdict golden paths ──────────────────────────────
    TestCase(
        name="easy_p005_car_seat_full_verdict",
        product_id="product_005",
        transform=identity(),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.SAFETY, Topic.EASE_OF_USE],
            must_have_cross_lingual_topic=Topic.EASE_OF_USE,
            cross_lingual_dominant_lang="en",
            high_confidence_min_evidence=5,
        ),
        description="Car seat, full review set. EN reviewers much more vocal about installation difficulty than AR reviewers (who use dealer installs). Should surface a cross-lingual insight with EN dominant.",
    ),
    TestCase(
        name="easy_p006_breast_pump_full_verdict",
        product_id="product_006",
        transform=identity(),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.QUALITY, Topic.EASE_OF_USE],
            must_have_cross_lingual_topic=Topic.EASE_OF_USE,
            cross_lingual_dominant_lang="ar",
            high_confidence_min_evidence=5,
        ),
        description="Breast pump, full review set. AR reviewers mention noise level concern far more than EN reviewers — Gulf household multi-generational living context. Cross-lingual insight should be AR dominant.",
    ),
    TestCase(
        name="easy_p007_bassinet_full_verdict",
        product_id="product_007",
        transform=identity(),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.SAFETY, Topic.EASE_OF_USE, Topic.QUALITY],
            must_have_cross_lingual_topic=Topic.QUALITY,
            cross_lingual_dominant_lang="ar",
            high_confidence_min_evidence=5,
        ),
        description="BassiNest swivel sleeper. AR reviewers flag mattress firmness more than EN reviewers. Swivel feature praised by both. Should surface mattress cross-lingual insight with AR dominant.",
    ),
    TestCase(
        name="easy_p008_lotion_full_verdict",
        product_id="product_008",
        transform=identity(),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.QUALITY, Topic.SAFETY],
            must_have_cross_lingual_topic=Topic.QUALITY,
            cross_lingual_dominant_lang="en",
            high_confidence_min_evidence=5,
        ),
        description="Baby lotion. EN reviewers raise fragrance/scent concerns far more than AR reviewers. Cross-lingual insight should reflect EN-dominant scent concern.",
    ),

    # ── Adversarial: strip asymmetric signal from new products ────────────────
    TestCase(
        name="adv_p006_strip_ar_noise_no_cross_lingual",
        product_id="product_006",
        transform=drop_ids(P006_AR_NOISE),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.QUALITY, Topic.EASE_OF_USE],
            must_have_cross_lingual_topic=None,
            no_cross_lingual_on_topics=[Topic.EASE_OF_USE],
            high_confidence_min_evidence=5,
        ),
        description="Remove the AR noise complaints from the breast pump dataset. Without the asymmetric signal the system should NOT generate a cross-lingual insight on noise/usability.",
        adversarial=True,
    ),
    TestCase(
        name="adv_p005_strip_en_install_no_cross_lingual",
        product_id="product_005",
        transform=drop_ids(P005_EN_INSTALL),
        expected=ExpectedVerdict(
            must_have_topics=[Topic.SAFETY, Topic.QUALITY],
            must_have_cross_lingual_topic=None,
            no_cross_lingual_on_topics=[Topic.EASE_OF_USE],
            high_confidence_min_evidence=4,
        ),
        description="Remove EN installation difficulty reviews from the car seat. Without the EN-heavy signal, no cross-lingual insight on installation/usability should appear.",
        adversarial=True,
    ),
]
