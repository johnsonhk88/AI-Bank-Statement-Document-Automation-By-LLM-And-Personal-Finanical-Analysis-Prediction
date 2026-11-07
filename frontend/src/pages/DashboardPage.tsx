import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";

interface MonthBucket {
  year: number;
  month: number;
  credits: number;
  debits: number;
  net: number;
  count: number;
}

interface TopItem {
  description: string;
  credits: number;
  debits: number;
  count: number;
}

interface AnalyticsSummary {
  year: number;
  quarter: number | null;
  credits: number;
  debits: number;
  net: number;
  count: number;
  by_month: MonthBucket[];
  top: TopItem[];
}

function Bar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const width = max > 0 ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  return (
    <div className="mb-1 flex items-center gap-2 text-xs">
      <span className="w-16 shrink-0 text-gray-500">{label}</span>
      <div className="h-4 flex-1 rounded bg-gray-100">
        <div className={`h-4 rounded ${color}`} style={{ width: `${width}%` }} />
      </div>
      <span className="w-24 shrink-0 text-right tabular-nums">{value.toFixed(2)}</span>
    </div>
  );
}

export default function DashboardPage() {
  const thisYear = new Date().getFullYear();
  const [year, setYear] = useState(thisYear);
  const [quarter, setQuarter] = useState<string>("");
  const [data, setData] = useState<AnalyticsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const q = quarter ? `&quarter=${quarter}` : "";
    api
      .get<AnalyticsSummary>(`/analytics/summary?year=${year}${q}`)
      .then(setData)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, [year, quarter]);

  const maxBar = useMemo(() => {
    if (!data) return 1;
    return Math.max(1, ...data.by_month.flatMap((b) => [b.credits, b.debits]));
  }, [data]);

  const downloadMarkdown = async () => {
    const token = localStorage.getItem("token");
    const q = quarter ? `&quarter=${quarter}` : "";
    const res = await fetch(`/api/analytics/export.md?year=${year}${q}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bankai-${year}${quarter ? `-Q${quarter}` : ""}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const printPdf = () => window.print();

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">History dashboard</h1>
          <p className="text-sm text-gray-500">
            Aggregated from succeeded runs stored in Postgres. No extra LLM call.
          </p>
        </div>
        <div className="flex gap-2">
          <select
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            className="rounded-md border border-gray-300 px-2 py-1 text-sm"
          >
            {[thisYear, thisYear - 1, thisYear - 2].map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
          <select
            value={quarter}
            onChange={(e) => setQuarter(e.target.value)}
            className="rounded-md border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="">Full year</option>
            <option value="1">Q1</option>
            <option value="2">Q2</option>
            <option value="3">Q3</option>
            <option value="4">Q4</option>
          </select>
          <button
            type="button"
            onClick={downloadMarkdown}
            className="rounded-md border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50"
          >
            Download Markdown
          </button>
          <button
            type="button"
            onClick={printPdf}
            className="rounded-md bg-indigo-600 px-3 py-1 text-sm text-white hover:bg-indigo-500"
          >
            Print / PDF
          </button>
        </div>
      </div>

      {loading && <p className="text-sm text-gray-500">Loading…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}
      {data && !loading && (
        <>
          <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
            {[
              ["Transactions", String(data.count)],
              ["Credits", data.credits.toFixed(2)],
              ["Debits", data.debits.toFixed(2)],
              ["Net", data.net.toFixed(2)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg border bg-white p-4">
                <div className="text-xs text-gray-500">{label}</div>
                <div className="text-lg font-semibold tabular-nums">{value}</div>
              </div>
            ))}
          </div>

          <div className="mb-8 rounded-lg border bg-white p-4">
            <h2 className="mb-3 text-sm font-semibold">Credits vs debits by month</h2>
            {data.by_month.length === 0 ? (
              <p className="text-sm text-gray-500">No dated transactions in this period.</p>
            ) : (
              data.by_month.map((b) => (
                <div key={`${b.year}-${b.month}`} className="mb-3">
                  <Bar
                    label={`${b.year}-${String(b.month).padStart(2, "0")}`}
                    value={b.credits}
                    max={maxBar}
                    color="bg-emerald-500"
                  />
                  <Bar label="" value={b.debits} max={maxBar} color="bg-rose-500" />
                </div>
              ))
            )}
          </div>

          <div className="rounded-lg border bg-white p-4">
            <h2 className="mb-3 text-sm font-semibold">Top descriptions</h2>
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-gray-500">
                  <th className="py-1">Description</th>
                  <th className="py-1 text-right">Credits</th>
                  <th className="py-1 text-right">Debits</th>
                  <th className="py-1 text-right">Count</th>
                </tr>
              </thead>
              <tbody>
                {data.top.map((t) => (
                  <tr key={t.description} className="border-t">
                    <td className="py-1">{t.description}</td>
                    <td className="py-1 text-right tabular-nums">{t.credits.toFixed(2)}</td>
                    <td className="py-1 text-right tabular-nums">{t.debits.toFixed(2)}</td>
                    <td className="py-1 text-right tabular-nums">{t.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
