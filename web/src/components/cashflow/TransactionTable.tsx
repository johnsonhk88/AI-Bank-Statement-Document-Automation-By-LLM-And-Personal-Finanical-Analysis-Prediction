import { useState } from "react";
import type { CashFlowTransaction } from "../../types";

type TransactionTableProps = {
  transactions: CashFlowTransaction[];
  total: number;
  limit: number;
  offset: number;
  currency: string;
  onPageChange: (newOffset: number) => void;
  onCategoryFilter: (category: string | null) => void;
  activeCategory: string | null;
};

function formatCurrency(value: number, currency: string): string {
  return new Intl.NumberFormat("en-HK", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
  }).format(value);
}

export function TransactionTable({
  transactions,
  total,
  limit,
  offset,
  currency,
  onPageChange,
  onCategoryFilter,
  activeCategory,
}: TransactionTableProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(total / limit);

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="p-4 border-b border-gray-100 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">
          Transactions
          {activeCategory && (
            <span className="ml-2 px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full">
              {activeCategory}
              <button
                type="button"
                onClick={() => onCategoryFilter(null)}
                className="ml-1 hover:text-blue-900"
              >
                x
              </button>
            </span>
          )}
        </h3>
        <span className="text-xs text-gray-400">
          {total.toLocaleString()} total
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-500 border-b border-gray-100">
              <th className="px-4 py-2">Date</th>
              <th className="px-4 py-2">Description</th>
              <th className="px-4 py-2">Merchant</th>
              <th className="px-4 py-2">Category</th>
              <th className="px-4 py-2 text-right">Amount</th>
              <th className="px-4 py-2 text-center">Receipt</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((tx) => (
              <>
                <tr
                  key={tx.id}
                  className={`border-b border-gray-50 hover:bg-gray-50 cursor-pointer ${
                    expandedId === tx.id ? "bg-blue-50" : ""
                  }`}
                  onClick={() =>
                    setExpandedId(expandedId === tx.id ? null : tx.id)
                  }
                >
                  <td className="px-4 py-2.5 whitespace-nowrap text-gray-600">
                    {tx.date}
                  </td>
                  <td className="px-4 py-2.5 max-w-[200px] truncate">
                    {tx.description}
                  </td>
                  <td className="px-4 py-2.5 text-gray-600">{tx.merchant}</td>
                  <td className="px-4 py-2.5">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onCategoryFilter(tx.category);
                      }}
                      className="px-2 py-0.5 text-xs rounded-full bg-gray-100 hover:bg-blue-100 hover:text-blue-700"
                    >
                      {tx.category}
                    </button>
                  </td>
                  <td
                    className={`px-4 py-2.5 text-right font-mono ${
                      tx.amount >= 0 ? "text-emerald-600" : "text-red-600"
                    }`}
                  >
                    {formatCurrency(tx.amount, currency)}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    {tx.receipt_id ? (
                      <span
                        className="inline-block w-2 h-2 rounded-full bg-emerald-400"
                        title={`Matched (${tx.match_confidence ?? "?"})`}
                      />
                    ) : (
                      <span className="text-gray-300">-</span>
                    )}
                  </td>
                </tr>

                {/* Line items drill-down */}
                {expandedId === tx.id && tx.line_items.length > 0 && (
                  <tr key={`${tx.id}-items`}>
                    <td colSpan={6} className="bg-gray-50 px-8 py-3">
                      <p className="text-xs font-semibold text-gray-500 mb-2">
                        Receipt Line Items ({tx.line_items.length})
                      </p>
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-gray-400">
                            <th className="text-left py-1">Item</th>
                            <th className="text-left py-1">Sub-category</th>
                            <th className="text-right py-1">Qty</th>
                            <th className="text-right py-1">Unit Price</th>
                            <th className="text-right py-1">Total</th>
                          </tr>
                        </thead>
                        <tbody>
                          {tx.line_items.map((li) => (
                            <tr
                              key={li.id}
                              className="border-t border-gray-100"
                            >
                              <td className="py-1">{li.item_name}</td>
                              <td className="py-1 text-gray-500">
                                {li.sub_category}
                              </td>
                              <td className="py-1 text-right">{li.quantity}</td>
                              <td className="py-1 text-right">
                                {formatCurrency(li.unit_price, currency)}
                              </td>
                              <td className="py-1 text-right font-medium">
                                {formatCurrency(li.total, currency)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </td>
                  </tr>
                )}

                {expandedId === tx.id && tx.line_items.length === 0 && (
                  <tr key={`${tx.id}-empty`}>
                    <td
                      colSpan={6}
                      className="bg-gray-50 px-8 py-3 text-xs text-gray-400"
                    >
                      No receipt line items linked to this transaction
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() => onPageChange(Math.max(0, offset - limit))}
            className="text-xs px-3 py-1 rounded border border-gray-300 disabled:opacity-40 hover:bg-gray-50"
          >
            Previous
          </button>
          <span className="text-xs text-gray-500">
            Page {currentPage} of {totalPages}
          </span>
          <button
            type="button"
            disabled={offset + limit >= total}
            onClick={() => onPageChange(offset + limit)}
            className="text-xs px-3 py-1 rounded border border-gray-300 disabled:opacity-40 hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
