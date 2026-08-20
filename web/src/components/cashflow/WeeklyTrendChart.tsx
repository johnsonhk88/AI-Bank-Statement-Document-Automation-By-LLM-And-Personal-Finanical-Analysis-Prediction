import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import type { WeeklyTrend } from "../../types";

type WeeklyTrendChartProps = {
  weeks: WeeklyTrend[];
  currency: string;
};

export function WeeklyTrendChart({ weeks, currency }: WeeklyTrendChartProps) {
  const formatCurrency = (v: number) =>
    new Intl.NumberFormat("en", {
      notation: "compact",
      style: "currency",
      currency,
    }).format(v);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">
        Weekly Trends
      </h3>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart
          data={weeks}
          margin={{ left: 10, right: 10, top: 5, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="week" tick={{ fontSize: 10 }} />
          <YAxis tickFormatter={formatCurrency} tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(value: number, name: string) => [
              new Intl.NumberFormat("en-HK", {
                style: "currency",
                currency,
              }).format(value),
              name.charAt(0).toUpperCase() + name.slice(1),
            ]}
          />
          <Legend />
          <Area
            type="monotone"
            dataKey="income"
            stroke="#059669"
            fill="#059669"
            fillOpacity={0.15}
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="expenses"
            stroke="#dc2626"
            fill="#dc2626"
            fillOpacity={0.15}
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="net"
            stroke="#2563eb"
            fill="#2563eb"
            fillOpacity={0.1}
            strokeWidth={2}
            strokeDasharray="5 5"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
