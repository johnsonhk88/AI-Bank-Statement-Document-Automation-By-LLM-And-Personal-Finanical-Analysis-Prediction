interface AgentDropdownProps {
  value: string;
  onChange: (value: string) => void;
}

const agents = [
  { value: "crewai", label: "CrewAI (Multi-Agent)", disabled: false },
  { value: "deep-agents", label: "Deep Agents (LangChain) - coming soon", disabled: true },
  { value: "hermes", label: "Hermes (Docker) - coming soon", disabled: true },
];

export default function AgentDropdown({ value, onChange }: AgentDropdownProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
    >
      {agents.map((a) => (
        <option key={a.value} value={a.value} disabled={a.disabled}>
          {a.label}
        </option>
      ))}
    </select>
  );
}
