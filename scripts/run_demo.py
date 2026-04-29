"""CLI demo: run the verdict pipeline on one product and print results.

    python scripts/run_demo.py --product product_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Arabic prints to the console.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.llm import LLMClient
from src.pipeline import load_products, load_reviews, synthesize


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True, help="e.g. product_001")
    parser.add_argument("--save", help="path to save the JSON verdict")
    parser.add_argument("--skip-ar", action="store_true", help="skip Arabic generation (debug)")
    args = parser.parse_args()

    console = Console()

    products = load_products()
    if args.product not in products:
        console.print(f"[red]Unknown product[/red]: {args.product}")
        console.print(f"Known: {', '.join(products)}")
        sys.exit(1)

    product = products[args.product]
    reviews = load_reviews(args.product)

    console.print(Panel.fit(
        f"[bold]{product.name_en}[/bold]\n"
        f"{product.brand} · {product.category} · {product.age_range}\n"
        f"{len(reviews)} reviews loaded.",
        title="Input",
    ))

    llm = LLMClient()
    verdict, meta = synthesize(product, reviews, llm=llm, skip_ar=args.skip_ar)

    if verdict.refusal:
        console.print(Panel.fit(
            f"[bold red]Refusal[/bold red]: {verdict.refusal.reason.value}\n\n"
            f"EN: {verdict.refusal.explanation_en}\n\n"
            f"AR: {verdict.refusal.explanation_ar}",
            title="Verdict",
        ))
    else:
        # EN
        console.print(Panel.fit(verdict.verdict_en.summary, title="Summary (EN)"))
        t = Table(title="Claims (EN)")
        t.add_column("Topic"); t.add_column("Sent."); t.add_column("Claim", overflow="fold")
        t.add_column("Conf."); t.add_column("N"); t.add_column("EN/AR"); t.add_column("Cites")
        for c in verdict.verdict_en.claims:
            t.add_row(
                c.topic.value, c.sentiment.value, c.claim,
                f"{c.confidence:.2f}", str(c.evidence_count),
                f"{c.language_distribution.en}/{c.language_distribution.ar}",
                ", ".join(c.citations[:3]) + ("..." if len(c.citations) > 3 else ""),
            )
        console.print(t)

        if verdict.verdict_en.cross_lingual_insights:
            console.print("[bold cyan]Cross-lingual insights (EN):[/bold cyan]")
            for ins in verdict.verdict_en.cross_lingual_insights:
                console.print(
                    f"  • {ins.insight}  (EN={ins.en_evidence_count}, AR={ins.ar_evidence_count}, conf={ins.confidence:.2f})"
                )

        # AR (only if not skipped)
        if not args.skip_ar and verdict.verdict_ar is not verdict.verdict_en:
            console.print(Panel.fit(verdict.verdict_ar.summary, title="ملخص (AR)"))
            t2 = Table(title="ادعاءات (AR)")
            t2.add_column("الموضوع"); t2.add_column("النبرة"); t2.add_column("الادعاء", overflow="fold")
            t2.add_column("الثقة"); t2.add_column("العدد"); t2.add_column("EN/AR"); t2.add_column("مراجع")
            for c in verdict.verdict_ar.claims:
                t2.add_row(
                    c.topic.value, c.sentiment.value, c.claim,
                    f"{c.confidence:.2f}", str(c.evidence_count),
                    f"{c.language_distribution.en}/{c.language_distribution.ar}",
                    ", ".join(c.citations[:3]) + ("..." if len(c.citations) > 3 else ""),
                )
            console.print(t2)

    console.print("\n[dim]meta:[/dim]", meta)

    if args.save:
        Path(args.save).write_text(verdict.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Saved to[/green] {args.save}")


if __name__ == "__main__":
    main()
