import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.base import AgentResult, Transaction


def _mock_scalar_result(value):
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _mock_scalars_result(values):
    r = MagicMock()
    r.scalars.return_value.all.return_value = values
    return r


def _build_mocks():
    item_id = uuid.uuid4()
    run_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    mock_item = MagicMock()
    mock_item.id = item_id
    mock_item.run_id = run_id
    mock_item.document_id = doc_id
    mock_item.status = "pending"
    mock_item.error = None
    mock_item.markdown_report = None
    mock_item.transactions = None
    mock_item.started_at = None
    mock_item.finished_at = None

    mock_run = MagicMock()
    mock_run.id = run_id
    mock_run.agent = "crewai"
    mock_run.question = "test question"
    mock_run.llm_provider = "openai"
    mock_run.llm_model = "openai/gpt-4o-mini"
    mock_run.status = "pending"
    mock_run.started_at = None
    mock_run.finished_at = None

    mock_doc = MagicMock()
    mock_doc.storage_path = "2026/07/test.pdf"

    mock_adapter = AsyncMock()
    mock_adapter.run.return_value = AgentResult(
        markdown_report="## Test Report",
        transactions=[
            Transaction(date="2026-01-01", description="Coffee", debit=45.00),
            Transaction(date="2026-01-02", description="Salary", credit=5000.00),
        ],
    )

    return item_id, mock_item, mock_run, mock_doc, mock_adapter


def _make_session_mock(*execute_results):
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(side_effect=list(execute_results))
    mock_session.commit = AsyncMock()
    return mock_session


@patch("app.worker.runner.agent_registry")
@patch("app.worker.runner.AsyncSessionLocal")
def test_run_item_sets_status_succeeded(mock_session_cls, mock_registry):
    item_id, mock_item, mock_run, mock_doc, mock_adapter = _build_mocks()
    mock_registry.get.return_value = mock_adapter

    mock_session = _make_session_mock(
        _mock_scalar_result(mock_item),
        _mock_scalar_result(mock_run),
        _mock_scalar_result(mock_doc),
        _mock_scalar_result(mock_run),
        _mock_scalars_result([mock_item]),
    )
    mock_session_cls.return_value.__aenter__.return_value = mock_session

    from app.worker.celery_app import celery_app
    celery_app.conf.task_always_eager = True

    from app.worker.tasks import run_agent_item
    run_agent_item(str(item_id))

    assert mock_item.status == "succeeded"
    assert mock_item.markdown_report == "## Test Report"
    assert len(mock_item.transactions) == 2
    assert mock_item.transactions[0]["description"] == "Coffee"
    assert mock_item.error is None
    assert mock_item.finished_at is not None
    assert mock_item.started_at is not None
    assert mock_run.status == "succeeded"
    assert mock_session.commit.await_count == 3


@patch("app.worker.runner.agent_registry")
@patch("app.worker.runner.AsyncSessionLocal")
def test_run_item_sets_status_failed_on_exception(mock_session_cls, mock_registry):
    item_id, mock_item, mock_run, mock_doc, _ = _build_mocks()

    mock_adapter = AsyncMock()
    mock_adapter.run.side_effect = RuntimeError("LLM inference timeout after 60s")
    mock_registry.get.return_value = mock_adapter

    mock_session = _make_session_mock(
        _mock_scalar_result(mock_item),
        _mock_scalar_result(mock_run),
        _mock_scalar_result(mock_doc),
        _mock_scalar_result(mock_run),
        _mock_scalars_result([mock_item]),
    )
    mock_session_cls.return_value.__aenter__.return_value = mock_session

    from app.worker.celery_app import celery_app
    celery_app.conf.task_always_eager = True

    from app.worker.tasks import run_agent_item
    run_agent_item(str(item_id))

    assert mock_item.status == "failed"
    assert "LLM inference timeout" in mock_item.error
    assert len(mock_item.error) <= 4000
    assert mock_item.markdown_report is None
    assert mock_item.transactions is None
    assert mock_item.finished_at is not None
    assert mock_item.started_at is not None
    assert mock_run.status == "failed"
    assert mock_session.commit.await_count == 3


@patch("app.worker.runner.agent_registry")
@patch("app.worker.runner.AsyncSessionLocal")
def test_refresh_run_status_partial_when_mixed(mock_session_cls, mock_registry):
    item_id, mock_item, mock_run, mock_doc, mock_adapter = _build_mocks()
    mock_registry.get.return_value = mock_adapter

    second_item = MagicMock()
    second_item.status = "failed"

    mock_session = _make_session_mock(
        _mock_scalar_result(mock_item),
        _mock_scalar_result(mock_run),
        _mock_scalar_result(mock_doc),
        _mock_scalar_result(mock_run),
        _mock_scalars_result([mock_item, second_item]),
    )
    mock_session_cls.return_value.__aenter__.return_value = mock_session

    from app.worker.celery_app import celery_app
    celery_app.conf.task_always_eager = True

    from app.worker.tasks import run_agent_item
    run_agent_item(str(item_id))

    assert mock_item.status == "succeeded"
    assert mock_run.status == "partial"


@patch("app.worker.runner.agent_registry")
@patch("app.worker.runner.AsyncSessionLocal")
def test_run_item_raises_on_missing_item(mock_session_cls, mock_registry):
    missing_id = uuid.uuid4()

    mock_session = _make_session_mock(
        _mock_scalar_result(None),
    )
    mock_session_cls.return_value.__aenter__.return_value = mock_session

    from app.worker.celery_app import celery_app
    celery_app.conf.task_always_eager = True

    from app.worker.tasks import run_agent_item

    with pytest.raises(Exception):
        run_agent_item(str(missing_id))
