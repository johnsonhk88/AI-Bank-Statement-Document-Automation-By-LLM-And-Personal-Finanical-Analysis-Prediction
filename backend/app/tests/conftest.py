import sys
from unittest.mock import MagicMock

_mock_modules = [
    "crewai",
    "crewai.skills",
    "crewai_skills_loader",
    "instructor",
    "litellm",
    "langchain_huggingface",
    "langchain_qdrant",
    "langchain_text_splitters",
]

for _mod in _mock_modules:
    sys.modules[_mod] = MagicMock()

sys.modules["crewai"].Process.sequential = "sequential"

import pytest  # noqa: E402
