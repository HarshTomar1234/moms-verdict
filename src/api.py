"""FastAPI app — single endpoint: POST /verdict

Run:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from src.llm import LLMClient, LLMError
from src.pipeline import load_products, load_reviews, synthesize
from src.schemas import Verdict

app = FastAPI(
    title="Moms Verdict API",
    description="Cross-lingual review intelligence for Mumzworld products.",
    version="0.1.0",
)

_products = load_products()


class VerdictRequest(BaseModel):
    product_id: str
    max_reviews: int | None = None


@app.get("/products")
def list_products():
    return [
        {
            "product_id": pid,
            "name_en": p.name_en,
            "name_ar": p.name_ar,
            "category": p.category,
        }
        for pid, p in _products.items()
    ]


@app.post("/verdict", response_model=None)
def get_verdict(req: VerdictRequest) -> JSONResponse:
    if req.product_id not in _products:
        raise HTTPException(status_code=404, detail=f"Product not found: {req.product_id}")

    product = _products[req.product_id]
    reviews = load_reviews(req.product_id)
    if req.max_reviews:
        reviews = reviews[: req.max_reviews]

    try:
        llm = LLMClient()
        verdict, meta = synthesize(product, reviews, llm=llm)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return JSONResponse(
        content={
            "verdict": verdict.model_dump(mode="json"),
            "meta": meta,
        }
    )


@app.get("/health")
def health():
    return {"status": "ok", "products_loaded": len(_products)}
