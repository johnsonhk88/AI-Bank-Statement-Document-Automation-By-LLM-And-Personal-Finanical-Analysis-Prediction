from app.agents.registry import agent_registry
from app.agents.crewai.adapter import CrewAIAdapter
from app.agents.stubs import DeepAgentsAdapter, HermesAdapter

agent_registry.register(CrewAIAdapter())
agent_registry.register(DeepAgentsAdapter())
agent_registry.register(HermesAdapter())
