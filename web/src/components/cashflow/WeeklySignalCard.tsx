import type { WeeklySignalsResponse, SignalValue } from "../../types";

type WeeklySignalCardProps = {
  data: WeeklySignalsResponse;
  currency: string;
};

const SIGNAL_CONFIG: Record<
  SignalValue,
  { label: string; color: string; bg: string; border: string; icon: string }
> = {
  INVEST: {
    label: "Invest Surplus",
    color: "text-emerald-700",
    bg: "bg-emerald-50",
    border: "border-emerald-300",
    icon: "\u2197",
  },
  HOLD: {
    label: "Hold Steady",
    color: "text-blue-700",
    bg: "bg-blue-50",
    border: "border-blue-300",
    icon: "\u2192",
  },
  ALERT: {
    label: "Spending Alert",
    color: "text-red-700",
    bg: "bg-red-50",
    border: "border-red-300",
    icon: "\u2198",
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

function formatChange(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toLocaleString("en-HK", { maximumFractionDigits: 0 })}`;
}

function changeColor(value: number): string {
  if (value > 0) return "text-emerald-600";
  if (value < 0) return "text-red-600";
  return "text-gray-500";
}

export function WeeklySignalCard({ data, currency }: WeeklySignalCardProps) {
  const cfg = SIGNAL_CONFIG[data.current_signal];
  const comp = data.comparison;

  return (
    <div className={`rounded-xl border-2 ${cfg.border} ${cfg.bg} p-5`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Weekly Signal
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-2xl">{cfg.icon}</span>
            <span className={`text-xl font-bold ${cfg.color}`}>
              {cfg.label}
            </span>
          </div>
        </div>
        <div className="text-right space-y-1">
          <div className="text-xs text-gray-500">RSI (4w)</div>
          <div
            className={`text-lg font-bold ${
              data.rsi === null
                ? "text-gray-400"
                : data.rsi >= 70
                  ? "text-emerald-600"
                  : data.rsi <= 30
                    ? "text-red-600"
                    : "text-gray-700"
            }`}
          >
            {data.rsi !== null ? data.rsi.toFixed(0) : "N/A"}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <div>
          <p className="text-xs text-gray-500">This week</p>
          <p className="text-sm font-bold text-gray-800">
            {formatAmount(comp.this_week_net, currency)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">vs last week</p>
          <p className={`text-sm font-bold ${changeColor(comp.week_over_week_change)}`}>
            {formatChange(comp.week_over_week_change)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">vs 4 weeks ago</p>
          <p
            className={`text-sm font-bold ${
              comp.vs_4_weeks_ago !== null
                ? changeColor(comp.vs_4_weeks_ago)
                : "text-gray-400"
            }`}
          >
            {comp.vs_4_weeks_ago !== null
              ? formatChange(comp.vs_4_weeks_ago)
              : "N/A"}
          </p>
        </div>
      </div>

      <p className="text-sm text-gray-600 leading-relaxed">
        {data.recommendation}
      </p>
    </div>
  );
}
