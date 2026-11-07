import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.base import AgentResult


class FakeCrewResult:
    def __init__(self, raw_text="mock report text"):
        self.raw = raw_text

    def __str__(self):
        return self.raw


@pytest.mark.asyncio
@patch("app.agents.crewai.adapter.load_skills", return_value=[])
@patch("app.agents.crewai.adapter.extract_transactions", return_value=[])
@patch("app.agents.crewai.adapter.LLMProviderRegistry")
@patch("app.agents.crewai.adapter.Process")
@patch("app.agents.crewai.adapter.LLM")
@patch("app.agents.crewai.adapter.Task")
@patch("app.agents.crewai.adapter.Agent")
@patch("app.agents.crewai.adapter.Crew")
async def test_crewai_adapter_run_returns_agent_result(
    mock_crew, mock_agent, mock_task, mock_llm, mock_process,
    mock_registry_cls, mock_extract, mock_load_skills
):
    mock_registry = MagicMock()
    mock_registry.resolve = MagicMock(
        return_value=("openai/gpt-4o-mini", "https://api.openai.com/v1", "sk-test")
    )
    mock_registry_cls.return_value = mock_registry

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff_async = AsyncMock(
        return_value=FakeCrewResult()
    )
    mock_crew.return_value = mock_crew_instance

    from app.agents.crewai.adapter import CrewAIAdapter

    adapter = CrewAIAdapter()
    result = await adapter.run(
        pdf_path="/tmp/test.pdf",
        question="What is my balance?",
        llm_provider_id="openai",
        llm_model_id="openai/gpt-4o-mini",
        agent_run_item_id=uuid.uuid4(),
    )

    assert isinstance(result, AgentResult)
    assert result.markdown_report == "mock report text"
    assert isinstance(result.transactions, list)
    assert result.raw == {"crewai_raw": "mock report text"}


@pytest.mark.asyncio
@patch("app.agents.crewai.adapter.load_skills", return_value=[])
@patch("app.agents.crewai.adapter.extract_transactions",
       side_effect=Exception("LLM unavailable"))
@patch("app.agents.crewai.adapter.LLMProviderRegistry")
@patch("app.agents.crewai.adapter.Process")
@patch("app.agents.crewai.adapter.LLM")
@patch("app.agents.crewai.adapter.Task")
@patch("app.agents.crewai.adapter.Agent")
@patch("app.agents.crewai.adapter.Crew")
async def test_crewai_adapter_extraction_failure_returns_empty_transactions(
    mock_crew, mock_agent, mock_task, mock_llm, mock_process,
    mock_registry_cls, mock_extract, mock_load_skills
):
    mock_registry = MagicMock()
    mock_registry.resolve = MagicMock(
        return_value=("openai/gpt-4o-mini", "https://api.openai.com/v1", "sk-test")
    )
    mock_registry_cls.return_value = mock_registry

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff_async = AsyncMock(
        return_value=FakeCrewResult("fallback text")
    )
    mock_crew.return_value = mock_crew_instance

    from app.agents.crewai.adapter import CrewAIAdapter

    adapter = CrewAIAdapter()
    result = await adapter.run(
        pdf_path="/tmp/test.pdf",
        question="What is my balance?",
        llm_provider_id="openai",
        llm_model_id="openai/gpt-4o-mini",
        agent_run_item_id=uuid.uuid4(),
    )

    assert isinstance(result, AgentResult)
    assert result.transactions == []
    assert result.markdown_report == "fallback text"
