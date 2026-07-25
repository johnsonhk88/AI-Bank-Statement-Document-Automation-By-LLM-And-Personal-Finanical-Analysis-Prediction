import sys
from unittest.mock import MagicMock

_mock_modules = [
    "crewai",
    "crewai.skills",
    "instructor",
    "litellm",
    "langchain_huggingface",
    "langchain_qdrant",
    "langchain_text_splitters",
    "app.skills",
    "app.skills.crewai_skills_loader",
]

for _mod in _mock_modules:
    sys.modules[_mod] = MagicMock()

sys.modules["crewai"].Process.sequential = "sequential"

import pytest  # noqa: E402
