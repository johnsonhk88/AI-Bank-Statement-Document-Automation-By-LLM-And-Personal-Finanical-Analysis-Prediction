import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Area,
  ComposedChart,
  Legend,
} from "recharts";
import type { CashFlowForecast } from "../../types";

type ForecastChartProps = {
  forecast: CashFlowForecast;
};

export function ForecastChart({ forecast }: ForecastChartProps) {
  // Merge income/expenses/net forecast into a single data array
  const data = forecast.net.map((netPoint, i) => ({
    month: netPoint.month,
    netPredicted: netPoint.predicted,
    netLower: netPoint.lower,
    netUpper: netPoint.upper,
    incomePredicted: forecast.income[i]?.predicted ?? 0,
    expensesPredicted: forecast.expenses[i]?.predicted ?? 0,
  }));

  const formatCurrency = (v: number) =>
    new Intl.NumberFormat("en", {
      notation: "compact",
      style: "currency",
      currency: forecast.currency,
    }).format(v);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-700 mb-1">
        Forecast ({forecast.horizon_months} months)
      </h3>
      <p className="text-xs text-gray-400 mb-4">
        Shaded area shows confidence band for net cashflow
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="month" tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={formatCurrency} tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(value: number, name: string) => [
              formatCurrency(value),
              name.replace(/([A-Z])/g, " $1").trim(),
            ]}
          />
          <Legend />
          {/* Confidence band for net */}
          <Area
            type="monotone"
            dataKey="netUpper"
            stroke="none"
            fill="#2563eb"
            fillOpacity={0.08}
          />
          <Area
            type="monotone"
            dataKey="netLower"
            stroke="none"
            fill="#ffffff"
            fillOpacity={1}
          />
          <Line
            type="monotone"
            dataKey="incomePredicted"
            stroke="#059669"
            strokeWidth={2}
            dot={false}
            name="Income"
          />
          <Line
            type="monotone"
            dataKey="expensesPredicted"
            stroke="#dc2626"
            strokeWidth={2}
            dot={false}
            name="Expenses"
          />
          <Line
            type="monotone"
            dataKey="netPredicted"
            stroke="#2563eb"
            strokeWidth={2.5}
            dot={{ r: 3 }}
            name="Net"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
