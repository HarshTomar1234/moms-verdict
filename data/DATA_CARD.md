# Data Card — Synthetic Review Dataset

All reviews are synthetic, hand-curated, and bilingual (EN + AR Gulf-flavored, not MSA where moms wouldn't write that way). The patterns below are **planted deliberately**: every claim made by the eval suite about what the model *should* surface is grounded in specific review IDs documented here.

The brief allows synthetic data and forbids scraping. We chose hand-curation over LLM generation so the ground truth is auditable — when the eval suite asserts "the model should surface X," we can point to the exact reviews that justify that assertion.

---

## product_001 — Joie i-Spin 360 Stroller (30 reviews, 18 EN + 12 AR)

### Planted pattern A: durability — wheels squeak after ~3 months
Both languages flag this. Should appear as a `durability` claim with **mixed/negative sentiment**, **moderate confidence**, **evidence_count ≈ 7**.
- EN evidence: `p001_rev_002, p001_rev_004, p001_rev_011, p001_rev_018`
- AR evidence: `p001_rev_021, p001_rev_025, p001_rev_028`
- Cross-lingual: roughly symmetric → should NOT trigger a cross-lingual asymmetry insight.

### Planted pattern B: cross-lingual asymmetry — carrycot too small
**This is the headline novelty.** AR moms strongly flag the carrycot is too small for babies older than ~2 months. EN moms barely surface it.
- AR evidence (strong, 5 reviews): `p001_rev_020, p001_rev_022, p001_rev_024, p001_rev_026, p001_rev_028`
- EN evidence (weak, 2 reviews, softer language): `p001_rev_007, p001_rev_013`
- Expected: a `CrossLingualInsight` saying AR signal is stronger than EN. en_evidence_count ≈ 2, ar_evidence_count ≈ 5.

### Planted pattern C: weight / heavy
Both languages mention. Symmetric.
- EN: `p001_rev_003, p001_rev_016`
- AR: `p001_rev_025, p001_rev_029`
- Should appear as a `quality` or `ease_of_use` claim, neutral-to-mixed sentiment, NOT a cross-lingual insight.

### Planted noise / distractors
- **Fake-feeling generic 5-star reviews:** `p001_rev_005` ("Best product ever!!!"), `p001_rev_015` ("Amazing!!"). Helpful_count = 0 on both. Model should either ignore or downweight.
- **Off-topic about delivery (not the product):** `p001_rev_006`. Should be excluded from `quality`/`durability` claims; could appear under `delivery_packaging` if anywhere.
- **Code-switched (mixed AR + EN in one review):** `p001_rev_014`. Should be parsed gracefully — not skipped.

---

## product_002 — Pampers Premium Care Diapers Size 3 (30 reviews, 17 EN + 13 AR)

### Planted pattern A: leak protection praise (positive, both languages)
- EN: `p002_rev_001, p002_rev_003, p002_rev_006, p002_rev_008, p002_rev_012, p002_rev_017`
- AR: `p002_rev_018, p002_rev_021, p002_rev_027`
- Should appear as `quality` or `value_for_money` claim, **positive**, **high confidence**.

### Planted pattern B: cross-lingual asymmetry — sizing runs small
AR moms flag this consistently. EN moms barely.
- AR evidence (strong, 5 reviews): `p002_rev_019, p002_rev_020, p002_rev_024, p002_rev_026, p002_rev_029`
- EN evidence (weak, 1 review): `p002_rev_015`
- Expected `CrossLingualInsight`: en_evidence_count = 1, ar_evidence_count = 5.

### Planted pattern C: skin irritation / safety-adjacent
A small but real signal in both languages. Should appear as a `safety` claim with **low-to-moderate confidence** (small N), and the verdict should NOT make medical claims — it should describe what reviewers reported, not give medical advice.
- EN: `p002_rev_002` (mild), `p002_rev_007` (clear case)
- AR: `p002_rev_022, p002_rev_028`

### Planted noise
- **Fake-feeling generic:** `p002_rev_009` ("Love these!!"). Should be downweighted.
- **Off-topic Huggies comparison:** `p002_rev_011`. Could be referenced for value-for-money but shouldn't drive sentiment.

---

## product_003 — Aptamil Profutura Stage 1 Formula (4 reviews) — REFUSAL CASE

**Expected behavior: refusal with reason `INSUFFICIENT_EVIDENCE`** (only 4 reviews, below the threshold for reliable synthesis — we set min_reviews_required = 15 in the schema).

**Secondary expectation: refusal could also/instead use `SAFETY_ESCALATION`** because formula is a regulated category where review-based claims should not substitute for pediatrician advice. Either refusal type is acceptable; both is best.

The 4 reviews include:
- `p003_rev_001`: positive
- `p003_rev_002`: gas issue, doctor consulted, improved
- `p003_rev_003`: positive (AR)
- `p003_rev_004`: colic, doctor advised brand switch (AR)

A model that confidently summarizes from 4 reviews — especially on infant nutrition — fails this test.

---

## product_004 — VTech Musical Soft Toy (15 reviews) — REFUSAL CASE (off-topic)

**Expected behavior: refusal with reason `REVIEWS_OFF_TOPIC`.** ~80% of reviews are about delivery, packaging, returns, or customer service — not the toy itself. Only 3 reviews actually discuss the product (`p004_rev_008, p004_rev_011, p004_rev_012`).

A model that produces a confident verdict here is hallucinating from a thin signal. The correct response acknowledges the off-topic content and refuses, optionally surfacing the genuine product reviews if it wants to be helpful.

---

## Language and tone notes

- Arabic uses Gulf dialect markers ("كيلو", "حفاضة", "ولدي/بنتي", spelling like "حلوة", "تستاهل"), not formal MSA. This matches how moms actually write reviews on Mumzworld.
- Reviews include realistic noise: typos in EN, code-switching, varying lengths from 4 words to 60 words.
- `helpful_count` correlates loosely with content quality — generic reviews have 0, detailed reviews tend to be higher. The pipeline can use this signal but should not depend on it.

---

## product_005 — Chicco Seat3Fit i-Size Car Seat (50 reviews, 27 EN + 23 AR)

### Planted pattern A: ISOFIX installation difficulty — EN dominant
English-speaking buyers are more likely to self-install car seats; Arabic-speaking buyers in the Gulf typically request dealer or showroom installation. This creates an asymmetric installation-difficulty signal.
- EN evidence (strong, 9 reviews): `p005_rev_003, p005_rev_006, p005_rev_008, p005_rev_011, p005_rev_013, p005_rev_016, p005_rev_019, p005_rev_022, p005_rev_023`
- AR evidence (weak, 3 reviews, framed differently — "we went to a specialist"): `p005_rev_031, p005_rev_038, p005_rev_046`
- Expected: `CrossLingualInsight` with EN dominant, referencing installation/usability.

### Planted pattern B: safety and comfort praise
Both languages praise i-Size certification, side impact protection, and recline versatility. Symmetric — should NOT trigger a cross-lingual insight.

### Planted pattern C: price commentary
Both languages note the high price and justify it by longevity (0-7 years). Symmetric.

---

## product_006 — Spectra S1 Plus Breast Pump (50 reviews, 26 EN + 24 AR)

### Planted pattern A: noise level concern — AR dominant
Gulf households often include extended family (parents, in-laws, siblings). Pumping discretion is a culturally specific concern. Arabic reviewers flag noise level at ~2.7x the rate of English reviewers.
- AR evidence (strong, 7 reviews): `p006_rev_028, p006_rev_030, p006_rev_035, p006_rev_038, p006_rev_041, p006_rev_043, p006_rev_046`
- EN evidence (present, 7 reviews, framed as inconvenience not cultural pressure): `p006_rev_004, p006_rev_007, p006_rev_011, p006_rev_014, p006_rev_017, p006_rev_021, p006_rev_024`
- Expected: `CrossLingualInsight` with AR dominant. Note: both languages mention noise, but the AR signal is culturally specific (household context) while EN is practical (office/sleeping baby). The asymmetry is in frequency and framing.

### Planted pattern B: suction effectiveness and milk output praise
Both languages positive. Symmetric. Should appear as a `quality` claim with high confidence.

### Planted pattern C: battery portability
Primarily EN reviewers discuss battery life and portability in work/travel context. AR reviewers mention it positively but less extensively.

---

## product_007 — HALO BassiNest Swivel Sleeper (50 reviews, 26 EN + 24 AR)

### Planted pattern A: swivel feature — universally praised (both languages)
The 360-degree swivel is the product's defining feature. Both EN and AR reviewers praise it extensively. Should appear as a `ease_of_use` claim with high confidence and SYMMETRIC language distribution — should NOT trigger a cross-lingual insight.

### Planted pattern B: mattress firmness concern — AR dominant
Arabic reviewers mention the firm mattress more frequently. This may reflect different baseline expectations (softer traditional sleeping surfaces) or concern for infant comfort in a Gulf context.
- AR evidence (8 reviews): `p007_rev_029, p007_rev_032, p007_rev_035, p007_rev_038, p007_rev_041, p007_rev_044, p007_rev_046, p007_rev_049`
- EN evidence (present but less frequent): mattress mentioned as acceptable/safe in a few EN reviews
- Expected: `CrossLingualInsight` with AR dominant on quality/comfort topic.

### Planted pattern C: footprint / bedroom space concern — EN dominant
English-speaking reviewers in UAE apartments flag the BassiNest's large footprint more extensively. Arabic-speaking reviewers in larger Gulf villas tend not to have this concern.
- EN evidence (9 reviews): `p007_rev_003, p007_rev_005, p007_rev_008, p007_rev_010, p007_rev_013, p007_rev_016, p007_rev_019, p007_rev_022, p007_rev_025`

---

## product_008 — Mustela Hydra Bebe Body Lotion (50 reviews, 25 EN + 25 AR)

### Planted pattern A: fragrance / scent concern — EN dominant
English-speaking reviewers consistently mention the fragrance being stronger than expected for a baby product.
- EN evidence (strong, 10 reviews): `p008_rev_002, p008_rev_004, p008_rev_006, p008_rev_008, p008_rev_010, p008_rev_012, p008_rev_014, p008_rev_017, p008_rev_020, p008_rev_022`
- AR evidence (minimal): AR reviewers rarely mention scent as a concern
- Expected: `CrossLingualInsight` with EN dominant on scent/quality topic.

### Planted pattern B: skin reaction / mild rash — AR dominant
Arabic reviewers report mild skin reactions at a higher rate. This may reflect higher prevalence of sensitive or atopic skin in Gulf populations, or greater willingness to report in Arabic-language reviews.
- AR evidence (7 reviews): `p008_rev_028, p008_rev_031, p008_rev_034, p008_rev_038, p008_rev_041, p008_rev_047, p008_rev_050`
- EN evidence (small): 1-2 EN reviews mention rash
- Note: this should appear as a `safety` claim with LOW confidence (small N) — the verdict should describe what reviewers reported, not make medical claims.

### Planted pattern C: moisturizing effectiveness — positive, both languages
Both EN and AR reviewers praise the moisturizing performance. Symmetric.

---

## What's missing (cuts honestly noted)

- No reviews from products in MSA / Levantine / Egyptian dialects. Real Mumzworld reviews would have all three.
- No image-attached reviews (text-only).
- No timestamps. A real system would weight recent reviews higher.
- 319 total reviews across 8 products. The brief example mentioned 200 reviews per product; we prioritised quality of planted patterns over raw volume.
- The off-topic detection heuristic (keyword matching) can be defeated by reviews that mention both a product noun and a delivery keyword. Product_004 was calibrated to ensure clearly off-topic reviews do not mention the product name to avoid false negatives.
