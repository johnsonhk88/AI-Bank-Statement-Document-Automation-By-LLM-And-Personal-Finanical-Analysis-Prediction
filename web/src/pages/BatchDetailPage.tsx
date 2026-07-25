import { useParams } from "react-router-dom";
import { useBatchPolling } from "../hooks/useBatchPolling";
import { retryAgentRunItems } from "../api/agentRuns";
import BatchItemCard from "../components/BatchItemCard";

function statusBadge(status: string) {
  const base = "px-2 py-0.5 text-xs font-medium rounded-full";
  switch (status) {
    case "pending":
      return <span className={`${base} bg-gray-100 text-gray-600`}>Pending</span>;
    case "running":
      return <span className={`${base} bg-blue-100 text-blue-700`}>Running</span>;
    case "succeeded":
      return <span className={`${base} bg-green-100 text-green-700`}>Succeeded</span>;
    case "failed":
      return <span className={`${base} bg-red-100 text-red-700`}>Failed</span>;
    case "partial":
      return <span className={`${base} bg-yellow-100 text-yellow-700`}>Partial</span>;
    default:
      return <span className={`${base} bg-gray-100 text-gray-600`}>{status}</span>;
  }
}

export default function BatchDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: batch, isLoading, isError, error } = useBatchPolling(id);

  const handleRetry = async (itemId: string) => {
    if (!id) return;
    try {
      await retryAgentRunItems(id, [itemId]);
    } catch (err) {
      console.error("Retry failed:", err);
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <svg className="animate-spin h-8 w-8 text-blue-500" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">
          {error instanceof Error ? error.message : "Failed to load batch"}
        </p>
      </div>
    );
  }

  if (!batch) return null;

  return (
    <div>
      <div className="mb-6 border rounded-lg p-4 bg-white">
        <h1 className="text-xl font-bold text-gray-900 mb-3">Batch {batch.id}</h1>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-500">Agent:</span>{" "}
            <span className="text-gray-900 capitalize">{batch.agent}</span>
          </div>
          <div>
            <span className="text-gray-500">Model:</span>{" "}
            <span className="text-gray-900">{batch.llm_model}</span>
          </div>
          <div className="col-span-2">
            <span className="text-gray-500">Question:</span>{" "}
            <span className="text-gray-900">{batch.question}</span>
          </div>
          <div>
            <span className="text-gray-500">Status:</span>{" "}
            {statusBadge(batch.status)}
          </div>
          <div>
            <span className="text-gray-500">Created:</span>{" "}
            <span className="text-gray-900">{new Date(batch.created_at).toLocaleString()}</span>
          </div>
        </div>
      </div>

      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        Items ({batch.items?.length ?? 0})
      </h2>

      <div className="space-y-3">
        {batch.items?.map((item) => (
          <BatchItemCard
            key={item.id}
            item={item}
            documentFilename={item.document_id}
            onRetry={handleRetry}
          />
        ))}
      </div>
    </div>
  );
}
