import sys
from pathlib import Path
from uuid import UUID

from crewai import Agent, Crew, LLM, Process, Task

from app.agents.base import AgentResult, BaseAgentAdapter
from app.agents.crewai.extractor import extract_transactions
from app.agents.llm_provider import LLMProviderRegistry

BACKEND = Path(__file__).resolve().parent.parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
from app.skills.crewai_skills_loader import load_skills  # noqa: E402


class CrewAIAdapter(BaseAgentAdapter):
    name = "crewai"
    display_name = "CrewAI (Multi-Agent)"
    enabled = True
    description = "CrewAI-based multi-agent bank statement analysis"

    def __init__(self):
        self._llm_registry = LLMProviderRegistry()

    async def run(self, *, pdf_path: str, question: str, llm_provider_id: str, llm_model_id: str, agent_run_item_id: UUID) -> AgentResult:
        litellm_model, base_url, api_key = self._llm_registry.resolve(llm_provider_id, llm_model_id)
        llm = LLM(model=litellm_model, base_url=base_url, api_key=api_key)
        skills = load_skills()

        agent1 = Agent(
            role="Bank Statement Parser",
            goal="Extract transactions",
            backstory="You parse bank statements.",
            llm=llm,
            skills=skills,
        )
        agent2 = Agent(
            role="PII Redactor",
            goal="Redact PII before storage",
            backstory="You protect privacy.",
            llm=llm,
            skills=skills,
        )
        agent3 = Agent(
            role="Vector Store Manager",
            goal="Index content",
            backstory="You manage knowledge bases.",
            llm=llm,
            skills=skills,
        )
        agent4 = Agent(
            role="Financial Analyst",
            goal="Analyze and answer questions",
            backstory="You are a CFA.",
            llm=llm,
            skills=skills,
        )
        agent5 = Agent(
            role="Output Formatter",
            goal="Format final report",
            backstory="You format reports.",
            llm=llm,
            skills=skills,
        )

        task1 = Task(
            description=f"Extract text from PDF: {pdf_path}. Apply bank-statement-parsing skill.",
            expected_output="Structured transaction list",
            agent=agent1,
        )
        task2 = Task(
            description="Redact all PII from extracted text. Apply pii-handling skill.",
            expected_output="Redacted text",
            agent=agent2,
        )
        task3 = Task(
            description="Store redacted text in Qdrant. Use vector_store tool.",
            expected_output="Storage confirmation",
            agent=agent3,
        )
        task4 = Task(
            description=f"Answer: {question}. Use rag query and financial-analysis skill. Cross-check: Opening + Credits - Debits = Closing.",
            expected_output="Markdown financial report",
            agent=agent4,
            output_file=f"/tmp/report_{agent_run_item_id}.md",
        )
        task5 = Task(
            description="Apply output-format skill.",
            expected_output="Final formatted markdown",
            agent=agent5,
        )

        crew = Crew(tasks=[task1, task2, task3, task4, task5], process=Process.sequential, verbose=False)
        result = await crew.kickoff_async(inputs={"pdf_path": pdf_path, "query": question})

        raw_text = getattr(result, "raw", str(result))
        try:
            transactions = await extract_transactions(raw_text, litellm_model, base_url, api_key)
        except Exception:
            transactions = []

        return AgentResult(markdown_report=str(result), transactions=transactions, raw={"crewai_raw": str(result)})
