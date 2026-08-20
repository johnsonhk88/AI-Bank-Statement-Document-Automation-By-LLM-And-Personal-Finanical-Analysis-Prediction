import type { CashFlowSummary } from "../../types";

type SummaryCardsProps = {
  data: CashFlowSummary;
};

function formatAmount(value: number, currency: string): string {
  return new Intl.NumberFormat("en-HK", {
    style: "currency",
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function SummaryCards({ data }: SummaryCardsProps) {
  const cards = [
    {
      label: "Income",
      value: data.total_income,
      color: "text-emerald-600",
      bg: "bg-emerald-50",
      border: "border-emerald-200",
    },
    {
      label: "Expenses",
      value: data.total_expenses,
      color: "text-red-600",
      bg: "bg-red-50",
      border: "border-red-200",
    },
    {
      label: "Net",
      value: data.net,
      color: data.net >= 0 ? "text-emerald-600" : "text-red-600",
      bg: data.net >= 0 ? "bg-emerald-50" : "bg-red-50",
      border: data.net >= 0 ? "border-emerald-200" : "border-red-200",
    },
    {
      label: "Transactions",
      value: data.transaction_count,
      color: "text-blue-600",
      bg: "bg-blue-50",
      border: "border-blue-200",
      isCount: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div
          key={card.label}
          className={`rounded-xl border ${card.border} ${card.bg} p-5`}
        >
          <p className="text-sm text-gray-500 font-medium">{card.label}</p>
          <p className={`text-2xl font-bold mt-1 ${card.color}`}>
            {card.isCount
              ? card.value.toLocaleString()
              : formatAmount(card.value, data.currency)}
          </p>
        </div>
      ))}
    </div>
  );
}
