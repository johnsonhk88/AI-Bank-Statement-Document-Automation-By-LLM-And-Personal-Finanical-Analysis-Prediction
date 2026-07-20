from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOTS = [
    ROOT / "skills",
    ROOT.parent / "hermes" / "skills",
]


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    assert m, "missing YAML frontmatter"
    data = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


@pytest.mark.parametrize("skills_dir", SKILL_ROOTS)
def test_skills_have_name_and_description(skills_dir: Path):
    assert skills_dir.is_dir(), f"missing {skills_dir}"
    skill_files = list(skills_dir.glob("*/SKILL.md"))
    assert len(skill_files) == 5
    for path in skill_files:
        meta = _parse_frontmatter(path.read_text(encoding="utf-8"))
        assert meta.get("name"), f"{path} missing name"
        assert meta.get("description"), f"{path} missing description"
        assert "compatibility" not in meta, f"{path} must not use crewai compatibility field"
