import type { SavingsCapacityResponse, SignalValue } from "../../types";

type SignalCardProps = {
  data: SavingsCapacityResponse;
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

function formatRSI(rsi: number | null): string {
  if (rsi === null) return "N/A";
  return rsi.toFixed(0);
}

function rsiColor(rsi: number | null): string {
  if (rsi === null) return "text-gray-400";
  if (rsi >= 70) return "text-emerald-600";
  if (rsi <= 30) return "text-red-600";
  return "text-gray-700";
}

function bbLabel(position: string): string {
  switch (position) {
    case "above_upper":
      return "Above Upper Band";
    case "upper_half":
      return "Upper Half";
    case "lower_half":
      return "Lower Half";
    case "below_lower":
      return "Below Lower Band";
    default:
      return "Insufficient Data";
  }
}

export function SignalCard({ data }: SignalCardProps) {
  const cfg = SIGNAL_CONFIG[data.current_signal];

  return (
    <div className={`rounded-xl border-2 ${cfg.border} ${cfg.bg} p-5`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Savings Signal
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-2xl">{cfg.icon}</span>
            <span className={`text-xl font-bold ${cfg.color}`}>
              {cfg.label}
            </span>
          </div>
        </div>
        <div className="text-right space-y-1">
          <div className="text-xs text-gray-500">RSI</div>
          <div className={`text-lg font-bold ${rsiColor(data.rsi)}`}>
            {formatRSI(data.rsi)}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4 mb-3 text-xs text-gray-500">
        <span>
          Bollinger:{" "}
          <span className="font-medium text-gray-700">
            {bbLabel(data.bollinger.position)}
          </span>
        </span>
        {data.bollinger.pband !== null && (
          <span>
            %B:{" "}
            <span className="font-medium text-gray-700">
              {(data.bollinger.pband * 100).toFixed(0)}%
            </span>
          </span>
        )}
      </div>

      <p className="text-sm text-gray-600 leading-relaxed">
        {data.recommendation}
      </p>
    </div>
  );
}
