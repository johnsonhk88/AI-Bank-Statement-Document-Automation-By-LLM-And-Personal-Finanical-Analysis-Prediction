#!/usr/bin/env python3
"""Copy canonical skills/ into each agent harness.

Edit only repo-root ``skills/``. Then:

    python scripts/sync_skills.py --apply   # overwrite harness copies
    python scripts/sync_skills.py --check   # exit 1 if copies drifted

Do not copy ``crewai_skills_loader.py`` into harness trees.
Do not edit harness copies by hand.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "skills"
HARNESSES = [
    REPO / "agents" / "crewai" / "skills",
    REPO / "agents" / "deep-agents" / "skills",
    REPO / "agents" / "hermes" / "skills",
]
SKILL_DIRS = [
    "bank-statement-parsing",
    "financial-analysis",
    "pii-handling",
    "rag-query-handling",
    "output-format",
]


def _read(root: Path, rel: str) -> str | None:
    path = root / rel
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def check() -> int:
    drifted = False
    for dest in HARNESSES:
        for name in SKILL_DIRS:
            rel = f"{name}/SKILL.md"
            src = _read(SOURCE, rel)
            dst = _read(dest, rel)
            if src != dst:
                print(f"DRIFT {dest.relative_to(REPO)}/{rel}")
                drifted = True
    if drifted:
        print("Run: python scripts/sync_skills.py --apply")
        return 1
    print("skills copies match skills/")
    return 0


def apply() -> int:
    for dest in HARNESSES:
        dest.mkdir(parents=True, exist_ok=True)
        for name in SKILL_DIRS:
            src_file = SOURCE / name / "SKILL.md"
            if not src_file.exists():
                print(f"missing source skill: {src_file}", file=sys.stderr)
                return 1
            dst_dir = dest / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_dir / "SKILL.md")
        print(f"synced {dest.relative_to(REPO)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="overwrite harness skill copies")
    group.add_argument("--check", action="store_true", help="fail if harness copies drifted")
    args = parser.parse_args()
    if not SOURCE.is_dir():
        print(f"canonical skills dir missing: {SOURCE}", file=sys.stderr)
        return 1
    return apply() if args.apply else check()


if __name__ == "__main__":
    raise SystemExit(main())
