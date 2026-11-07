import { useState } from "react";
import type { AgentRunItemOut } from "../types";
import MarkdownViewer from "./MarkdownViewer";
import TransactionsTable from "./TransactionsTable";

interface BatchItemCardProps {
  item: AgentRunItemOut;
  documentFilename?: string;
  onRetry?: (itemId: string) => void;
}

function statusBadge(status: string) {
  switch (status) {
    case "pending":
      return <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-gray-100 text-gray-600">Pending</span>;
    case "running":
      return (
        <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-700 inline-flex items-center gap-1.5">
          <svg className="animate-spin h-3 w-3 text-blue-500" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Running
        </span>
      );
    case "succeeded":
      return <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-green-100 text-green-700">Succeeded</span>;
    case "failed":
      return <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-red-100 text-red-700">Failed</span>;
    case "partial":
      return <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-yellow-100 text-yellow-700">Partial</span>;
    default:
      return <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-gray-100 text-gray-600">{status}</span>;
  }
}

export default function BatchItemCard({ item, documentFilename, onRetry }: BatchItemCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border rounded-lg p-4 bg-white">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-900">
            {documentFilename || item.document_id}
          </span>
          {statusBadge(item.status)}
        </div>
        <div className="flex items-center gap-2">
          {(item.status === "succeeded" || item.status === "failed" || item.status === "partial") && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-xs text-blue-600 hover:text-blue-800"
            >
              {expanded ? "Hide details" : "View details"}
            </button>
          )}
          {item.status === "failed" && (
            <button
              onClick={() => onRetry?.(item.id)}
              className="px-2 py-0.5 text-xs font-medium rounded bg-red-50 text-red-600 hover:bg-red-100 border border-red-200"
            >
              Retry
            </button>
          )}
        </div>
      </div>

      {item.status === "failed" && item.error && (
        <p className="mt-2 text-sm text-red-600">{item.error}</p>
      )}

      {expanded && (
        <div className="mt-4 space-y-4 border-t pt-4">
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Report
            </h4>
            <MarkdownViewer content={item.markdown_report} />
          </div>
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Transactions
            </h4>
            <TransactionsTable transactions={item.transactions} />
          </div>
        </div>
      )}
    </div>
  );
}
