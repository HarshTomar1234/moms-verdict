"""Streamlit demo UI — for the 3-minute Loom walkthrough.

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.llm import LLMClient, LLMError
from src.pipeline import load_products, load_reviews, synthesize
from src.schemas import RefusalReason

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Moms Verdict — Mumzworld",
    page_icon="🌟",
    layout="wide",
)

st.title("Moms Verdict")
st.caption("Cross-lingual review intelligence for Mumzworld — EN + AR")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    products = load_products()
    product_labels = {p.product_id: f"{p.name_en} ({p.brand})" for p in products.values()}
    selected_id = st.selectbox(
        "Product",
        options=list(products.keys()),
        format_func=lambda pid: product_labels[pid],
    )
    product = products[selected_id]
    reviews = load_reviews(selected_id)

    max_rev = st.slider(
        "Max reviews to use",
        min_value=1,
        max_value=len(reviews),
        value=len(reviews),
        help="Drag left to test the sparse-evidence refusal (< 15 triggers refusal).",
    )
    reviews_subset = reviews[:max_rev]

    st.metric("Reviews loaded", max_rev)
    n_en = sum(1 for r in reviews_subset if r.language == "en")
    n_ar = sum(1 for r in reviews_subset if r.language == "ar")
    st.caption(f"🇬🇧 EN: {n_en}  ·  🇸🇦 AR: {n_ar}")

    run_btn = st.button("Generate Verdict", type="primary", use_container_width=True)

# ── Product info ──────────────────────────────────────────────────────────────

col_en, col_ar = st.columns(2)
with col_en:
    st.subheader(product.name_en)
    st.caption(f"{product.brand} · {product.category} · {product.age_range}")
    st.write(product.description_en)
with col_ar:
    st.subheader(product.name_ar)
    st.caption(f"{product.brand} · {product.category} · {product.age_range}")
    st.write(product.description_ar)

st.divider()

# ── Reviews preview ───────────────────────────────────────────────────────────

with st.expander(f"Reviews ({max_rev})", expanded=False):
    for r in reviews_subset[:10]:
        flag = "🇬🇧" if r.language == "en" else "🇸🇦"
        stars = "★" * r.rating + "☆" * (5 - r.rating)
        st.markdown(f"**{flag} {stars}** `{r.review_id}` *(helpful: {r.helpful_count})*")
        st.write(r.text)
        st.divider()
    if max_rev > 10:
        st.caption(f"… and {max_rev - 10} more reviews not shown in preview.")

# ── Verdict generation ────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Calling model for English verdict…"):
        t0 = time.time()
        try:
            llm = LLMClient()
            verdict, meta = synthesize(product, reviews_subset, llm=llm)
        except LLMError as e:
            st.error(f"LLM error: {e}\n\nCheck your OPENROUTER_API_KEY in .env")
            st.stop()
        elapsed = time.time() - t0

    st.success(f"Done in {elapsed:.1f}s")

    if verdict.refusal:
        r = verdict.refusal
        reason_map = {
            RefusalReason.INSUFFICIENT_EVIDENCE: ("⚠️ Insufficient Evidence", "orange"),
            RefusalReason.SAFETY_ESCALATION: ("🏥 Safety Escalation", "red"),
            RefusalReason.REVIEWS_OFF_TOPIC: ("🚫 Reviews Off-topic", "orange"),
            RefusalReason.OUT_OF_SCOPE: ("❓ Out of Scope", "grey"),
        }
        label, color = reason_map.get(r.reason, ("Refusal", "grey"))
        st.warning(f"**{label}**")
        st.markdown(f"**English:** {r.explanation_en}")
        st.markdown(f"**العربية:** {r.explanation_ar}")
        if r.min_reviews_required:
            st.caption(f"Minimum required: {r.min_reviews_required} reviews. You provided {max_rev}.")
        if r.suggested_action:
            st.info(r.suggested_action)
    else:
        tab_en, tab_ar, tab_insights, tab_meta = st.tabs(
            ["🇬🇧 English Verdict", "🇸🇦 Arabic Verdict", "🔍 Cross-lingual Insights", "📊 Metadata"]
        )

        # EN
        with tab_en:
            st.subheader("Summary")
            st.info(verdict.verdict_en.summary)
            st.subheader("Claims")
            for c in verdict.verdict_en.claims:
                conf_color = "green" if c.confidence >= 0.7 else "orange" if c.confidence >= 0.4 else "red"
                with st.expander(
                    f"**{c.topic.value.replace('_',' ').title()}** · {c.sentiment.value} · conf {c.confidence:.2f}",
                    expanded=True,
                ):
                    st.write(c.claim)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Evidence", c.evidence_count)
                    c2.metric("EN reviews", c.language_distribution.en)
                    c3.metric("AR reviews", c.language_distribution.ar)
                    st.caption("Citations: " + " · ".join(c.citations))

        # AR
        with tab_ar:
            st.subheader("الملخص")
            st.info(verdict.verdict_ar.summary)
            st.subheader("الادعاءات")
            for c in verdict.verdict_ar.claims:
                with st.expander(
                    f"**{c.topic.value.replace('_',' ')}** · {c.sentiment.value} · ثقة {c.confidence:.2f}",
                    expanded=True,
                ):
                    st.write(c.claim)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("الشواهد", c.evidence_count)
                    c2.metric("تقييمات EN", c.language_distribution.en)
                    c3.metric("تقييمات AR", c.language_distribution.ar)
                    st.caption("المراجع: " + " · ".join(c.citations))

        # Cross-lingual insights
        with tab_insights:
            insights = verdict.verdict_en.cross_lingual_insights
            if not insights:
                st.info("No cross-lingual asymmetry detected for this product/review set.")
            else:
                st.subheader(f"{len(insights)} cross-lingual insight(s) found")
                for ins in insights:
                    with st.expander(f"Confidence {ins.confidence:.2f}", expanded=True):
                        st.write(ins.insight)
                        c1, c2 = st.columns(2)
                        c1.metric("🇬🇧 EN evidence", ins.en_evidence_count)
                        c2.metric("🇸🇦 AR evidence", ins.ar_evidence_count)
                        if ins.citations_en:
                            st.caption("EN cites: " + " · ".join(ins.citations_en))
                        if ins.citations_ar:
                            st.caption("AR cites: " + " · ".join(ins.citations_ar))

        # Metadata
        with tab_meta:
            st.json(meta)

    # Raw JSON toggle
    with st.expander("Raw verdict JSON"):
        st.json(verdict.model_dump(mode="json"))
