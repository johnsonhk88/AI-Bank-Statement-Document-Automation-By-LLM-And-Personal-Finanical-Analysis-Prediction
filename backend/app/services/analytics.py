from __future__ import annotations

import datetime
import re
from collections import defaultdict
from typing import Any

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


def parse_tx_date(value: str | None) -> datetime.date | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text[:32], fmt).date()
        except ValueError:
            continue
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _num(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_transactions(
    rows: list[dict],
    *,
    year: int,
    quarter: int | None,
) -> dict:
    months = defaultdict(lambda: {"credits": 0.0, "debits": 0.0, "count": 0})
    tops: dict[str, dict] = defaultdict(lambda: {"credits": 0.0, "debits": 0.0, "count": 0})
    q_months = {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9), 4: (10, 11, 12)}.get(quarter or 0)

    for raw in rows:
        if not isinstance(raw, dict):
            continue
        dt = parse_tx_date(raw.get("date"))
        if dt is None or dt.year != year:
            continue
        if q_months and dt.month not in q_months:
            continue
        credit = _num(raw.get("credit"))
        debit = _num(raw.get("debit"))
        key = (dt.year, dt.month)
        months[key]["credits"] += credit
        months[key]["debits"] += debit
        months[key]["count"] += 1
        desc = str(raw.get("description") or "Unknown").strip()[:80]
        tops[desc]["credits"] += credit
        tops[desc]["debits"] += debit
        tops[desc]["count"] += 1

    by_month = [
        {
            "year": y,
            "month": m,
            "credits": round(v["credits"], 2),
            "debits": round(v["debits"], 2),
            "net": round(v["credits"] - v["debits"], 2),
            "count": v["count"],
        }
        for (y, m), v in sorted(months.items())
    ]
    credits = round(sum(b["credits"] for b in by_month), 2)
    debits = round(sum(b["debits"] for b in by_month), 2)
    top = sorted(
        (
            {
                "description": name,
                "credits": round(v["credits"], 2),
                "debits": round(v["debits"], 2),
                "count": v["count"],
            }
            for name, v in tops.items()
        ),
        key=lambda x: x["credits"] + x["debits"],
        reverse=True,
    )[:15]
    return {
        "year": year,
        "quarter": quarter,
        "credits": credits,
        "debits": debits,
        "net": round(credits - debits, 2),
        "count": sum(b["count"] for b in by_month),
        "by_month": by_month,
        "top": top,
    }


def summary_markdown(summary: dict) -> str:
    q = f" Q{summary['quarter']}" if summary.get("quarter") else ""
    lines = [
        f"# BankAI {summary['year']}{q} report",
        "",
        f"- Transactions: **{summary['count']}**",
        f"- Credits: **{summary['credits']:.2f}**",
        f"- Debits: **{summary['debits']:.2f}**",
        f"- Net: **{summary['net']:.2f}**",
        "",
        "## By month",
        "",
        "| Month | Credits | Debits | Net | Count |",
        "|------:|--------:|-------:|----:|------:|",
    ]
    for b in summary["by_month"]:
        lines.append(
            f"| {b['year']}-{b['month']:02d} | {b['credits']:.2f} | {b['debits']:.2f} | {b['net']:.2f} | {b['count']} |"
        )
    lines += ["", "## Top descriptions", "", "| Description | Credits | Debits | Count |", "|---|---:|---:|---:|"]
    for t in summary["top"]:
        lines.append(f"| {t['description']} | {t['credits']:.2f} | {t['debits']:.2f} | {t['count']} |")
    lines.append("")
    return "\n".join(lines)
