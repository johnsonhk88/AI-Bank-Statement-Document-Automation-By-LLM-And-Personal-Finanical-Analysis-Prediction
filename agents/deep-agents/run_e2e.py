#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import run_pipeline, build_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Agents bank-statement E2E")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to bank statement PDF")
    parser.add_argument("--question", required=True, help="Finance question for RAG")
    parser.add_argument(
        "--mode",
        choices=("pipeline", "agent"),
        default="pipeline",
        help="pipeline=deterministic tools only; agent=deepagents LLM loop",
    )
    parser.add_argument("--model", default="ollama:llama3.2", help="Model id for --mode agent")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    if args.mode == "pipeline":
        print(run_pipeline(args.pdf, args.question))
        return 0

    agent = build_agent(model=args.model)
    prompt = (
        f"Process bank statement PDF at {args.pdf.resolve()}. "
        f"Extract text, redact PII, store vectors, then answer: {args.question}"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages") if isinstance(result, dict) else None
    if messages:
        print(messages[-1].content if hasattr(messages[-1], "content") else messages[-1])
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
