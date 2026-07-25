import { useLLMModels } from "../hooks/useLLMModels";

interface ModelDropdownProps {
  providerId: string;
  modelId: string;
  onProviderChange: (providerId: string, modelId: string) => void;
}

export default function ModelDropdown({
  providerId,
  modelId,
  onProviderChange,
}: ModelDropdownProps) {
  const { data, isLoading } = useLLMModels();
  const providers = data?.providers ?? [];

  const selectedProvider = providers.find((p) => p.id === providerId);
  const models = selectedProvider?.models ?? [];

  if (isLoading) {
    return <div className="text-sm text-gray-500">Loading models...</div>;
  }

  const handleProviderChange = (newProviderId: string) => {
    const provider = providers.find((p) => p.id === newProviderId);
    const firstModel = provider?.models[0]?.id ?? "";
    onProviderChange(newProviderId, firstModel);
  };

  const handleModelChange = (newModelId: string) => {
    onProviderChange(providerId, newModelId);
  };

  const selectClasses =
    "rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500";

  return (
    <div className="flex gap-2">
      <select
        value={providerId}
        onChange={(e) => handleProviderChange(e.target.value)}
        className={selectClasses}
      >
        {providers.map((p) => (
          <option key={p.id} value={p.id} disabled={!p.available}>
            {p.display_name}
            {!p.available && p.unavailable_reason
              ? ` (${p.unavailable_reason})`
              : ""}
          </option>
        ))}
      </select>
      <select
        value={modelId}
        onChange={(e) => handleModelChange(e.target.value)}
        className={selectClasses}
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.display_name}
          </option>
        ))}
      </select>
    </div>
  );
}
