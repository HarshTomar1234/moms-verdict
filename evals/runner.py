"""Eval runner. Reads test_cases.CASES, runs each, aggregates scores.

Run:
    python -m evals.runner
    python -m evals.runner --case easy_p001_full_verdict
    python -m evals.runner --no-llm-graders   # skip AR fluency to save tokens

Outputs:
- console table of pass/fail per case + per grader
- evals/results/run_<timestamp>.json with full detail
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows so Unicode chars print to the console.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make src importable when run as module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.llm import LLMClient
from src.pipeline import load_products, load_reviews, synthesize
from src.schemas import Verdict

from evals.graders import (
    grade_ar_fluency,
    grade_calibration,
    grade_cross_lingual,
    grade_grounding,
    grade_refusal,
    grade_topics_present,
)
from evals.test_cases import CASES, ExpectedRefusal, ExpectedVerdict, TestCase


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_case(
    case: TestCase,
    llm: LLMClient,
    *,
    run_llm_graders: bool,
) -> dict:
    products = load_products()
    product = products[case.product_id]
    base_reviews = load_reviews(case.product_id)
    reviews = case.transform(base_reviews)
    valid_ids = {r.review_id for r in reviews}

    record: dict = {
        "name": case.name,
        "adversarial": case.adversarial,
        "description": case.description,
        "product_id": case.product_id,
        "n_reviews_after_transform": len(reviews),
        "expected_kind": "refusal" if isinstance(case.expected, ExpectedRefusal) else "verdict",
        "graders": [],
        "error": None,
    }

    try:
        verdict, meta = synthesize(product, reviews, llm=llm)
        record["meta"] = meta
        record["verdict"] = json.loads(verdict.model_dump_json())
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        return record

    # Always-on cheap graders
    record["graders"].append(grade_refusal(case, verdict))
    if isinstance(case.expected, ExpectedVerdict):
        record["graders"].append(grade_topics_present(case, verdict))
        record["graders"].append(grade_cross_lingual(case, verdict))
        record["graders"].append(grade_calibration(case, verdict))
        record["graders"].append(grade_grounding(case, verdict, valid_ids))
        if run_llm_graders:
            record["graders"].append(grade_ar_fluency(case, verdict, llm))

    return record


def aggregate(results: list[dict]) -> dict:
    grader_totals: dict[str, dict] = {}
    cases_passed = 0
    cases_total = 0
    for r in results:
        if r.get("error"):
            cases_total += 1
            continue
        all_passed = True
        for g in r["graders"]:
            if g.get("skipped"):
                continue
            name = g["name"]
            if name not in grader_totals:
                grader_totals[name] = {"pass": 0, "fail": 0, "score_sum": 0.0, "n": 0}
            if g.get("passed"):
                grader_totals[name]["pass"] += 1
            else:
                grader_totals[name]["fail"] += 1
                all_passed = False
            grader_totals[name]["score_sum"] += g.get("score", 0.0)
            grader_totals[name]["n"] += 1
        cases_total += 1
        if all_passed:
            cases_passed += 1

    for name, t in grader_totals.items():
        t["mean_score"] = round(t["score_sum"] / max(t["n"], 1), 3)

    return {
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "graders": grader_totals,
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="run only this case by name")
    parser.add_argument(
        "--no-llm-graders",
        action="store_true",
        help="skip AR-fluency LLM grader (saves tokens for quick iteration)",
    )
    args = parser.parse_args()

    cases = CASES if not args.case else [c for c in CASES if c.name == args.case]
    if not cases:
        print(f"No cases match {args.case!r}. Known: {[c.name for c in CASES]}")
        sys.exit(1)

    console = Console()
    llm = LLMClient()

    import time as _time

    results = []
    for idx, case in enumerate(cases):
        # Wait between cases so Groq's per-minute TPM window resets.
        # Refusal cases are deterministic (no LLM call) so skip the wait for them.
        if idx > 0:
            from evals.test_cases import ExpectedRefusal as _ER
            if not isinstance(case.expected, _ER):
                console.print("[dim]Waiting 15s for Groq TPM window...[/dim]")
                _time.sleep(15)

        console.rule(f"[bold]{case.name}[/bold]")
        rec = run_case(case, llm, run_llm_graders=not args.no_llm_graders)
        results.append(rec)
        if rec.get("error"):
            console.print(f"[red]ERROR[/red]: {rec['error']}")
            continue
        for g in rec["graders"]:
            if g.get("skipped"):
                continue
            mark = "[green]pass[/green]" if g.get("passed") else "[red]FAIL[/red]"
            console.print(f"  {mark} {g['name']:24s} score={g.get('score', 0):.2f}  {g.get('notes', '')}")

    summary = aggregate(results)
    console.rule("[bold]Summary[/bold]")
    table = Table()
    table.add_column("Grader")
    table.add_column("Pass")
    table.add_column("Fail")
    table.add_column("Mean score")
    for name, t in summary["graders"].items():
        table.add_row(name, str(t["pass"]), str(t["fail"]), f"{t['mean_score']:.2f}")
    console.print(table)
    console.print(
        f"\n[bold]Cases:[/bold] {summary['cases_passed']}/{summary['cases_total']} passed all graders\n"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{ts}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[dim]Saved[/dim] {out_path}")


if __name__ == "__main__":
    main()
