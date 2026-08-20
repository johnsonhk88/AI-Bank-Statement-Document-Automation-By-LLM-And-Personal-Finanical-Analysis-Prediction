import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DateRangeFilter } from "../components/cashflow/DateRangeFilter";
import { SummaryCards } from "../components/cashflow/SummaryCards";
import { CategoryChart } from "../components/cashflow/CategoryChart";
import { TrendChart } from "../components/cashflow/TrendChart";
import { ForecastChart } from "../components/cashflow/ForecastChart";
import { TransactionTable } from "../components/cashflow/TransactionTable";
import type {
  CashFlowSummary,
  CashFlowCategories,
  CashFlowTrends,
  CashFlowForecast,
  TransactionList,
} from "../types";

function defaultDateFrom(): string {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 1);
  return d.toISOString().slice(0, 10);
}

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function CashFlowPage() {
  const [dateFrom, setDateFrom] = useState(defaultDateFrom);
  const [dateTo, setDateTo] = useState(todayStr);
  const [currency, setCurrency] = useState("HKD");
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [txOffset, setTxOffset] = useState(0);
  const txLimit = 25;

  const queryParams = `date_from=${dateFrom}&date_to=${dateTo}&currency=${currency}`;

  const summaryQ = useQuery({
    queryKey: ["cashflow-summary", dateFrom, dateTo, currency],
    queryFn: () => api.get<CashFlowSummary>(`/cashflow/summary?${queryParams}`),
  });

  const categoriesQ = useQuery({
    queryKey: ["cashflow-categories", dateFrom, dateTo, currency],
    queryFn: () =>
      api.get<CashFlowCategories>(`/cashflow/categories?${queryParams}`),
  });

  const trendsQ = useQuery({
    queryKey: ["cashflow-trends", dateFrom, dateTo, currency],
    queryFn: () => api.get<CashFlowTrends>(`/cashflow/trends?${queryParams}`),
  });

  const forecastQ = useQuery({
    queryKey: ["cashflow-forecast", currency],
    queryFn: () =>
      api.get<CashFlowForecast>(
        `/cashflow/forecast?horizon_months=6&currency=${currency}`
      ),
    retry: false,
  });

  const txParams = new URLSearchParams({
    date_from: dateFrom,
    date_to: dateTo,
    currency,
    limit: String(txLimit),
    offset: String(txOffset),
  });
  if (categoryFilter) txParams.set("category", categoryFilter);

  const transactionsQ = useQuery({
    queryKey: [
      "cashflow-transactions",
      dateFrom,
      dateTo,
      currency,
      categoryFilter,
      txOffset,
    ],
    queryFn: () =>
      api.get<TransactionList>(`/cashflow/transactions?${txParams.toString()}`),
  });

  const handleCategoryClick = (category: string) => {
    setCategoryFilter(category);
    setTxOffset(0);
  };

  const isLoading =
    summaryQ.isLoading || categoriesQ.isLoading || trendsQ.isLoading;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Cash Flow</h1>
        <DateRangeFilter
          dateFrom={dateFrom}
          dateTo={dateTo}
          currency={currency}
          onDateFromChange={setDateFrom}
          onDateToChange={setDateTo}
          onCurrencyChange={setCurrency}
        />
      </div>

      {isLoading && (
        <div className="text-center py-12 text-gray-400">Loading...</div>
      )}

      {summaryQ.data && <SummaryCards data={summaryQ.data} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {categoriesQ.data && (
          <CategoryChart
            categories={categoriesQ.data.categories}
            currency={currency}
            onCategoryClick={handleCategoryClick}
          />
        )}
        {trendsQ.data && (
          <TrendChart months={trendsQ.data.months} currency={currency} />
        )}
      </div>

      {forecastQ.data && <ForecastChart forecast={forecastQ.data} />}

      {forecastQ.isError && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-700">
          Forecast unavailable — need at least 3 months of data.
        </div>
      )}

      {transactionsQ.data && (
        <TransactionTable
          transactions={transactionsQ.data.items}
          total={transactionsQ.data.total}
          limit={transactionsQ.data.limit}
          offset={transactionsQ.data.offset}
          currency={currency}
          onPageChange={setTxOffset}
          onCategoryFilter={setCategoryFilter}
          activeCategory={categoryFilter}
        />
      )}
    </div>
  );
}
