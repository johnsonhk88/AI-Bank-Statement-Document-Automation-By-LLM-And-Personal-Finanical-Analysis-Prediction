from app.agents.base import AgentInfo, BaseAgentAdapter


class AgentRegistry:
    def __init__(self):
        self._adapters: dict[str, BaseAgentAdapter] = {}

    def register(self, adapter: BaseAgentAdapter):
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> BaseAgentAdapter:
        if name not in self._adapters:
            raise ValueError(f"Unknown agent: {name}")
        return self._adapters[name]

    def list(self) -> list[AgentInfo]:
        return [
            AgentInfo(name=a.name, display_name=a.display_name, enabled=a.enabled, description=a.description)
            for a in self._adapters.values()
        ]

    def is_valid(self, name: str) -> bool:
        return name in self._adapters and self._adapters[name].enabled


agent_registry = AgentRegistry()
