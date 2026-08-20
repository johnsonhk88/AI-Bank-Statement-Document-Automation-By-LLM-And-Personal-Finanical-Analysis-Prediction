import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { CategoryAmount } from "../../types";

type CategoryChartProps = {
  categories: CategoryAmount[];
  currency: string;
  onCategoryClick: (category: string) => void;
};

const COLORS = [
  "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
  "#db2777", "#0891b2", "#65a30d", "#ea580c", "#4f46e5",
];

export function CategoryChart({
  categories,
  currency,
  onCategoryClick,
}: CategoryChartProps) {
  const data = categories.map((c, i) => ({
    ...c,
    fill: COLORS[i % COLORS.length],
  }));

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">
        Expense Categories
      </h3>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 80, right: 20, top: 5, bottom: 5 }}
        >
          <XAxis
            type="number"
            tickFormatter={(v) =>
              new Intl.NumberFormat("en", {
                notation: "compact",
                style: "currency",
                currency,
              }).format(v)
            }
          />
          <YAxis
            type="category"
            dataKey="category"
            width={75}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            formatter={(value: number) =>
              new Intl.NumberFormat("en-HK", {
                style: "currency",
                currency,
              }).format(value)
            }
          />
          <Bar
            dataKey="amount"
            radius={[0, 4, 4, 0]}
            cursor="pointer"
            onClick={(entry) => onCategoryClick(entry.category)}
          >
            {data.map((entry, i) => (
              <Cell key={entry.category} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
