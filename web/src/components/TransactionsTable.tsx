import type { Transaction } from "../types";

interface TransactionsTableProps {
  transactions: Transaction[] | null | undefined;
}

const fmt = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export default function TransactionsTable({
  transactions,
}: TransactionsTableProps) {
  if (!transactions || transactions.length === 0) {
    return (
      <p className="text-sm text-gray-400 italic">No transactions extracted</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm border border-gray-200 rounded-lg">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-3 py-2 text-left font-medium text-gray-600">Date</th>
            <th className="px-3 py-2 text-left font-medium text-gray-600">Description</th>
            <th className="px-3 py-2 text-right font-medium text-gray-600">Credit</th>
            <th className="px-3 py-2 text-right font-medium text-gray-600">Debit</th>
            <th className="px-3 py-2 text-right font-medium text-gray-600">Balance</th>
            <th className="px-3 py-2 text-left font-medium text-gray-600">Currency</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {transactions.map((tx, i) => (
            <tr key={i} className="hover:bg-gray-50">
              <td className="px-3 py-2 whitespace-nowrap text-gray-700">{tx.date}</td>
              <td className="px-3 py-2 text-gray-700 max-w-xs truncate">{tx.description}</td>
              <td className="px-3 py-2 text-right whitespace-nowrap text-green-600">
                {tx.credit != null ? fmt.format(tx.credit) : ""}
              </td>
              <td className="px-3 py-2 text-right whitespace-nowrap text-red-600">
                {tx.debit != null ? fmt.format(tx.debit) : ""}
              </td>
              <td className="px-3 py-2 text-right whitespace-nowrap text-gray-700">
                {tx.balance != null ? fmt.format(tx.balance) : ""}
              </td>
              <td className="px-3 py-2 whitespace-nowrap text-gray-500">{tx.currency}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
