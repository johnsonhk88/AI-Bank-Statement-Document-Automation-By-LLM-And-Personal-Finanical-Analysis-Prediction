import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { createAgentRun } from "../api/agentRuns";
import { useLLMModels } from "../hooks/useLLMModels";
import PdfMultiDropzone from "../components/PdfMultiDropzone";
import ExistingDocumentPicker from "../components/ExistingDocumentPicker";
import AgentDropdown from "../components/AgentDropdown";
import ModelDropdown from "../components/ModelDropdown";
import type { DocumentOut } from "../types";

const PROVIDER_KEY = "bankai-provider-id";
const MODEL_KEY = "bankai-model-id";
const DEFAULT_QUESTION =
  "Produce a full analysis report: opening and closing balance, total credits and debits, notable transactions, and a short summary.";

export default function NewBatchPage() {
  const navigate = useNavigate();
  const { data, isLoading: isLoadingModels } = useLLMModels();
  const providers = data?.providers ?? [];

  const [selectedDocs, setSelectedDocs] = useState<DocumentOut[]>([]);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [agent, setAgent] = useState("crewai");
  const [providerId, setProviderId] = useState<string>("");
  const [modelId, setModelId] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedProvider = providers.find((p) => p.id === providerId);

  useEffect(() => {
    if (providers.length === 0) return;

    const savedProvider = localStorage.getItem(PROVIDER_KEY);
    const savedModel = localStorage.getItem(MODEL_KEY);

    const savedProviderExists = providers.some(
      (p) => p.id === savedProvider && p.available,
    );

    if (savedProviderExists && savedModel) {
      const provider = providers.find((p) => p.id === savedProvider);
      const modelExists = provider?.models.some((m) => m.id === savedModel);
      if (modelExists) {
        setProviderId(savedProvider!);
        setModelId(savedModel!);
        return;
      }
    }

    const firstAvailable = providers.find((p) => p.available);
    if (firstAvailable) {
      setProviderId(firstAvailable.id);
      setModelId(firstAvailable.models[0]?.id ?? "");
    }
  }, [providers]);

  const handleProviderChange = (newProviderId: string, newModelId: string) => {
    setProviderId(newProviderId);
    setModelId(newModelId);
  };

  const canSubmit =
    selectedDocs.length > 0 &&
    question.trim().length > 0 &&
    agent &&
    providerId &&
    modelId &&
    !submitting;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;

    setSubmitting(true);
    setError(null);

    try {
      const run = await createAgentRun({
        document_ids: selectedDocs.map((d) => d.id),
        agent,
        question: question.trim(),
        llm_provider_id: providerId,
        llm_model_id: modelId,
      });

      localStorage.setItem(PROVIDER_KEY, providerId);
      localStorage.setItem(MODEL_KEY, modelId);

      navigate(`/batches/${run.id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create batch");
    } finally {
      setSubmitting(false);
    }
  };

  if (isLoadingModels) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-8">
        <p className="text-center text-gray-500">Loading models...</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="mb-2 text-2xl font-bold text-gray-900">New analysis</h1>
      <ol className="mb-6 flex flex-wrap gap-2 text-xs text-gray-600">
        <li className="rounded-full bg-indigo-50 px-3 py-1">1. Source PDF</li>
        <li className="rounded-full bg-indigo-50 px-3 py-1">2. LLM engine</li>
        <li className="rounded-full bg-indigo-50 px-3 py-1">3. Question</li>
        <li className="rounded-full bg-indigo-50 px-3 py-1">4. RAG + report → Postgres</li>
      </ol>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="mb-1 block text-sm font-medium text-gray-700">
            Upload new PDFs
          </label>
          <PdfMultiDropzone
            selectedDocs={selectedDocs}
            onDocsChange={setSelectedDocs}
          />
        </div>

        <div>
          <label className="mb-1 block text-sm font-medium text-gray-700">
            Or select existing statements
          </label>
          <ExistingDocumentPicker
            selectedDocs={selectedDocs}
            onDocsChange={setSelectedDocs}
          />
        </div>

        <div>
          <label className="mb-1 block text-sm font-medium text-gray-700">
            Agent
          </label>
          <AgentDropdown value={agent} onChange={setAgent} />
        </div>

        <div>
          <label className="mb-1 block text-sm font-medium text-gray-700">
            Model
          </label>
          <ModelDropdown
            providerId={providerId}
            modelId={modelId}
            onProviderChange={handleProviderChange}
          />
          {selectedProvider?.kind === "local" && (
            <p className="mt-2 text-xs text-gray-500">
              Local engine — GPU/CPU/RAM:{" "}
              <a
                className="text-indigo-600 hover:underline"
                href="http://localhost:3001/d/bankai-local-llm"
                target="_blank"
                rel="noreferrer"
              >
                Grafana dashboard
              </a>
            </p>
          )}
        </div>

        <div>
          <label
            htmlFor="question"
            className="mb-1 block text-sm font-medium text-gray-700"
          >
            Question
          </label>
          <textarea
            id="question"
            rows={4}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </div>

        {error && (
          <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={!canSubmit}
          className="w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? "Queuing RAG + report..." : "Run analysis"}
        </button>
      </form>
    </div>
  );
}
