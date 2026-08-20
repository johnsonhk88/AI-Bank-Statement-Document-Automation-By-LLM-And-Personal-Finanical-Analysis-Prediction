import type { IncomeStabilityResponse, RiskProfile, IncomeTrend } from "../../types";

type IncomeStabilityCardProps = {
  data: IncomeStabilityResponse;
  currency: string;
};

const RISK_CONFIG: Record<
  RiskProfile,
  { label: string; color: string; bg: string; border: string; icon: string; meter: string }
> = {
  LOW: {
    label: "Low Risk",
    color: "text-emerald-700",
    bg: "bg-emerald-50",
    border: "border-emerald-300",
    icon: "\u2714",
    meter: "bg-emerald-500",
  },
  MEDIUM: {
    label: "Moderate Risk",
    color: "text-amber-700",
    bg: "bg-amber-50",
    border: "border-amber-300",
    icon: "\u26A0",
    meter: "bg-amber-500",
  },
  HIGH: {
    label: "High Risk",
    color: "text-red-700",
    bg: "bg-red-50",
    border: "border-red-300",
    icon: "\u26A1",
    meter: "bg-red-500",
  },
};

const TREND_LABEL: Record<IncomeTrend, { label: string; icon: string; color: string }> = {
  growing: { label: "Growing", icon: "\u2191", color: "text-emerald-600" },
  declining: { label: "Declining", icon: "\u2193", color: "text-red-600" },
  stable: { label: "Stable", icon: "\u2192", color: "text-gray-600" },
};

function formatAmount(value: number, currency: string): string {
  return new Intl.NumberFormat("en-HK", {
    style: "currency",
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function VolatilityMeter({ atrPercent, meterColor }: { atrPercent: number; meterColor: string }) {
  // Clamp to 0-50% range for visual display
  const width = Math.min(atrPercent / 50 * 100, 100);
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500">Volatility</span>
        <span className="text-xs font-bold text-gray-700">{atrPercent.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${meterColor}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <div className="flex justify-between mt-0.5">
        <span className="text-[10px] text-gray-400">0%</span>
        <span className="text-[10px] text-gray-400">10%</span>
        <span className="text-[10px] text-gray-400">25%</span>
        <span className="text-[10px] text-gray-400">50%</span>
      </div>
    </div>
  );
}

export function IncomeStabilityCard({ data, currency }: IncomeStabilityCardProps) {
  const cfg = RISK_CONFIG[data.risk_profile];
  const trend = TREND_LABEL[data.income_trend];

  return (
    <div className={`rounded-xl border-2 ${cfg.border} ${cfg.bg} p-5`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Income Stability
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-2xl">{cfg.icon}</span>
            <span className={`text-xl font-bold ${cfg.color}`}>
              {cfg.label}
            </span>
          </div>
        </div>
        <div className="text-right space-y-1">
          <div className="text-xs text-gray-500">Avg Income</div>
          <div className="text-lg font-bold text-gray-800">
            {formatAmount(data.mean_income, currency)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-3">
        <div>
          <VolatilityMeter atrPercent={data.atr_percent} meterColor={cfg.meter} />
        </div>
        <div className="flex items-center gap-2">
          <div>
            <p className="text-xs text-gray-500">Trend</p>
            <p className={`text-sm font-bold ${trend.color}`}>
              {trend.icon} {trend.label}
            </p>
          </div>
          <div className="ml-auto">
            <p className="text-xs text-gray-500">ATR</p>
            <p className="text-sm font-bold text-gray-700">
              {formatAmount(data.atr, currency)}
            </p>
          </div>
        </div>
      </div>

      <p className="text-sm text-gray-600 leading-relaxed">
        {data.recommendation}
      </p>
    </div>
  );
}
