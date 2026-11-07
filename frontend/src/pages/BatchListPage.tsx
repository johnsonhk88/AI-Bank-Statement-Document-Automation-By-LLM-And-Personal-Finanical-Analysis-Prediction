import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { listAgentRuns } from "../api/agentRuns";
import type { AgentRunOut } from "../types";

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

function itemCount(items: AgentRunOut["items"]) {
  if (!items || items.length === 0) return "0/0";
  const succeeded = items.filter((i) => i.status === "succeeded").length;
  return `${succeeded}/${items.length}`;
}

export default function BatchListPage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["agent-runs"],
    queryFn: () => listAgentRuns(50, 0),
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Batches</h1>
        <Link
          to="/batches/new"
          className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700"
        >
          New Batch
        </Link>
      </div>

      {isLoading && (
        <div className="flex justify-center py-12">
          <svg className="animate-spin h-8 w-8 text-blue-500" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
      )}

      {isError && (
        <div className="text-center py-12">
          <p className="text-red-600 text-sm">
            {error instanceof Error ? error.message : "Failed to load batches"}
          </p>
        </div>
      )}

      {data && data.items.length === 0 && (
        <div className="text-center py-16">
          <p className="text-gray-500 mb-3">No batches yet</p>
          <Link to="/batches/new" className="text-sm text-blue-600 hover:text-blue-800">
            Create your first batch
          </Link>
        </div>
      )}

      {data && data.items.length > 0 && (
        <div className="overflow-x-auto border border-gray-200 rounded-lg bg-white">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-600">Created</th>
                <th className="px-4 py-3 text-left font-medium text-gray-600">Agent</th>
                <th className="px-4 py-3 text-left font-medium text-gray-600">Model</th>
                <th className="px-4 py-3 text-left font-medium text-gray-600">Question</th>
                <th className="px-4 py-3 text-left font-medium text-gray-600">Status</th>
                <th className="px-4 py-3 text-left font-medium text-gray-600">Items</th>
                <th className="px-4 py-3 text-left font-medium text-gray-600">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {data.items.map((run) => (
                <tr key={run.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 whitespace-nowrap text-gray-700">
                    {new Date(run.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-gray-700 capitalize">{run.agent}</td>
                  <td className="px-4 py-3 whitespace-nowrap text-gray-700">
                    {run.llm_model.split("/").pop()}
                  </td>
                  <td className="px-4 py-3 text-gray-700 max-w-xs truncate">{run.question}</td>
                  <td className="px-4 py-3 whitespace-nowrap">{statusBadge(run.status)}</td>
                  <td className="px-4 py-3 whitespace-nowrap text-gray-600">{itemCount(run.items)}</td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <Link
                      to={`/batches/${run.id}`}
                      className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
