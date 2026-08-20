import type {
  SpendingAlertsResponse,
  CategorySpendingAlert,
  SpendingTrend,
} from "../../types";

type SpendingAlertsProps = {
  data: SpendingAlertsResponse;
  currency: string;
};

const TREND_CONFIG: Record<
  SpendingTrend,
  { label: string; icon: string; color: string; bg: string; border: string }
> = {
  accelerating: {
    label: "Rising",
    icon: "\u2191",
    color: "text-red-700",
    bg: "bg-red-50",
    border: "border-red-200",
  },
  decelerating: {
    label: "Falling",
    icon: "\u2193",
    color: "text-emerald-700",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
  },
  stable: {
    label: "Stable",
    icon: "\u2192",
    color: "text-gray-600",
    bg: "bg-gray-50",
    border: "border-gray-200",
  },
};

function formatAmount(value: number, currency: string): string {
  return new Intl.NumberFormat("en-HK", {
    style: "currency",
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function MiniHistogram({ months }: { months: CategorySpendingAlert["months"] }) {
  const values = months.map((m) => m.amount);
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;

  // Show last 6 months as tiny bars
  const recent = values.slice(-6);

  return (
    <div className="flex items-end gap-px h-6">
      {recent.map((v, i) => {
        const height = Math.max(((v - min) / range) * 100, 8);
        return (
          <div
            key={i}
            className="w-1.5 bg-gray-300 rounded-sm"
            style={{ height: `${height}%` }}
          />
        );
      })}
    </div>
  );
}

function AlertCard({
  alert,
  currency,
}: {
  alert: CategorySpendingAlert;
  currency: string;
}) {
  const cfg = TREND_CONFIG[alert.trend];
  const avgMonthly =
    alert.months.reduce((sum, m) => sum + m.amount, 0) / alert.months.length;

  return (
    <div className={`rounded-xl border ${cfg.border} ${cfg.bg} p-4`}>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-gray-800">
          {alert.category}
        </h4>
        <span
          className={`inline-flex items-center gap-1 text-xs font-bold px-2 py-0.5 rounded-full ${cfg.color} ${cfg.bg} border ${cfg.border}`}
        >
          {cfg.icon} {cfg.label}
        </span>
      </div>

      <div className="flex items-end justify-between mb-2">
        <div>
          <p className="text-xs text-gray-500">Avg / month</p>
          <p className="text-base font-bold text-gray-800">
            {formatAmount(avgMonthly, currency)}
          </p>
        </div>
        <MiniHistogram months={alert.months} />
      </div>

      <p className="text-xs text-gray-500 leading-relaxed">
        {alert.recommendation}
      </p>
    </div>
  );
}

export function SpendingAlerts({ data, currency }: SpendingAlertsProps) {
  // Sort: alerts first, then by category name
  const sorted = [...data.categories].sort((a, b) => {
    if (a.alert !== b.alert) return a.alert ? -1 : 1;
    return a.category.localeCompare(b.category);
  });

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-700">
          Spending Trends by Category
        </h3>
        {data.alert_count > 0 && (
          <span className="text-xs font-bold text-red-600 bg-red-50 border border-red-200 px-2 py-0.5 rounded-full">
            {data.alert_count}{" "}
            {data.alert_count === 1 ? "alert" : "alerts"}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {sorted.map((alert) => (
          <AlertCard key={alert.category} alert={alert} currency={currency} />
        ))}
      </div>
    </div>
  );
}
