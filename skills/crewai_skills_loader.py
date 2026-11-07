from pathlib import Path
from typing import List, Optional

from crewai.skills import activate_skill, discover_skills


def load_skills(skills_dir: Optional[str] = None) -> List:
    """Load and activate all CrewAI skills from the canonical skills directory.

    Args:
        skills_dir: Path to the skills directory. Defaults to this file's parent
            (repo-root ``skills/``).

    Returns:
        List of activated Skill objects ready to be passed to CrewAI agents.
    """
    base_path = Path(skills_dir) if skills_dir is not None else Path(__file__).parent

    if not base_path.exists():
        raise FileNotFoundError(f"Skills directory not found: {base_path}")

    discovered_skills = discover_skills(base_path)

    if not discovered_skills:
        print("No skills found in the skills directory.")
        return []

    activated_skills = [activate_skill(skill) for skill in discovered_skills]

    print(f"Loaded {len(activated_skills)} CrewAI skills:")
    for skill in activated_skills:
        print(f"  - {skill.name}: {skill.description}")

    return activated_skills


if __name__ == "__main__":
    skills = load_skills()
    print("\nSkills loaded successfully!")
    print(skills)
