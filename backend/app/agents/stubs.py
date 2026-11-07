from app.agents.base import AgentResult, BaseAgentAdapter


class DeepAgentsAdapter(BaseAgentAdapter):
    name = "deep-agents"
    display_name = "Deep Agents (LangChain)"
    enabled = False
    description = "Coming soon"

    async def run(self, **kw):
        raise NotImplementedError


class HermesAdapter(BaseAgentAdapter):
    name = "hermes"
    display_name = "Hermes (Docker)"
    enabled = False
    description = "Coming soon"

    async def run(self, **kw):
        raise NotImplementedError
