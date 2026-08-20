import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { DateRangeFilter } from "../components/cashflow/DateRangeFilter";
import { SummaryCards } from "../components/cashflow/SummaryCards";
import { SignalCard } from "../components/cashflow/SignalCard";
import { SpendingAlerts } from "../components/cashflow/SpendingAlerts";
import { CategoryChart } from "../components/cashflow/CategoryChart";
import { TrendChart } from "../components/cashflow/TrendChart";
import { ForecastChart } from "../components/cashflow/ForecastChart";
import { WeeklySignalCard } from "../components/cashflow/WeeklySignalCard";
import { WeeklyTrendChart } from "../components/cashflow/WeeklyTrendChart";
import { WeeklyForecastChart } from "../components/cashflow/WeeklyForecastChart";
import { TransactionTable } from "../components/cashflow/TransactionTable";
import type {
  CashFlowSummary,
  CashFlowCategories,
  CashFlowTrends,
  CashFlowForecast,
  SavingsCapacityResponse,
  SpendingAlertsResponse,
  WeeklyTrends,
  WeeklySignalsResponse,
  WeeklyForecast,
  TransactionList,
} from "../types";

type Period = "monthly" | "weekly";

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
  const [period, setPeriod] = useState<Period>("monthly");
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [txOffset, setTxOffset] = useState(0);
  const txLimit = 25;

  const queryParams = `date_from=${dateFrom}&date_to=${dateTo}&currency=${currency}`;
  const isWeekly = period === "weekly";

  const summaryQ = useQuery({
    queryKey: ["cashflow-summary", dateFrom, dateTo, currency],
    queryFn: () => api.get<CashFlowSummary>(`/cashflow/summary?${queryParams}`),
  });

  const categoriesQ = useQuery({
    queryKey: ["cashflow-categories", dateFrom, dateTo, currency],
    queryFn: () =>
      api.get<CashFlowCategories>(`/cashflow/categories?${queryParams}`),
  });

  // --- Monthly queries ---

  const trendsQ = useQuery({
    queryKey: ["cashflow-trends", dateFrom, dateTo, currency],
    queryFn: () => api.get<CashFlowTrends>(`/cashflow/trends?${queryParams}`),
    enabled: !isWeekly,
  });

  const forecastQ = useQuery({
    queryKey: ["cashflow-forecast", currency],
    queryFn: () =>
      api.get<CashFlowForecast>(
        `/cashflow/forecast?horizon_months=6&currency=${currency}`
      ),
    retry: false,
    enabled: !isWeekly,
  });

  const signalsQ = useQuery({
    queryKey: ["cashflow-signals", currency],
    queryFn: () =>
      api.get<SavingsCapacityResponse>(
        `/cashflow/signals?currency=${currency}`
      ),
    retry: false,
    enabled: !isWeekly,
  });

  const spendingAlertsQ = useQuery({
    queryKey: ["cashflow-spending-alerts", currency],
    queryFn: () =>
      api.get<SpendingAlertsResponse>(
        `/cashflow/signals/spending?currency=${currency}`
      ),
    retry: false,
    enabled: !isWeekly,
  });

  // --- Weekly queries ---

  const weeklyTrendsQ = useQuery({
    queryKey: ["cashflow-weekly-trends", dateFrom, dateTo, currency],
    queryFn: () =>
      api.get<WeeklyTrends>(`/cashflow/weekly/trends?${queryParams}`),
    enabled: isWeekly,
  });

  const weeklySignalsQ = useQuery({
    queryKey: ["cashflow-weekly-signals", currency],
    queryFn: () =>
      api.get<WeeklySignalsResponse>(
        `/cashflow/weekly/signals?currency=${currency}`
      ),
    retry: false,
    enabled: isWeekly,
  });

  const weeklyForecastQ = useQuery({
    queryKey: ["cashflow-weekly-forecast", currency],
    queryFn: () =>
      api.get<WeeklyForecast>(
        `/cashflow/weekly/forecast?horizon_weeks=4&currency=${currency}`
      ),
    retry: false,
    enabled: isWeekly,
  });

  // --- Transactions (always active) ---

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
    summaryQ.isLoading || categoriesQ.isLoading || (isWeekly ? weeklyTrendsQ.isLoading : trendsQ.isLoading);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Cash Flow</h1>
        <div className="flex items-center gap-4">
          <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-0.5">
            <button
              type="button"
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                !isWeekly
                  ? "bg-white text-gray-900 shadow-sm border border-gray-200"
                  : "text-gray-500 hover:text-gray-700"
              }`}
              onClick={() => setPeriod("monthly")}
            >
              Monthly
            </button>
            <button
              type="button"
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                isWeekly
                  ? "bg-white text-gray-900 shadow-sm border border-gray-200"
                  : "text-gray-500 hover:text-gray-700"
              }`}
              onClick={() => setPeriod("weekly")}
            >
              Weekly
            </button>
          </div>
          <DateRangeFilter
            dateFrom={dateFrom}
            dateTo={dateTo}
            currency={currency}
            onDateFromChange={setDateFrom}
            onDateToChange={setDateTo}
            onCurrencyChange={setCurrency}
          />
        </div>
      </div>

      {isLoading && (
        <div className="text-center py-12 text-gray-400">Loading...</div>
      )}

      {summaryQ.data && <SummaryCards data={summaryQ.data} />}

      {/* --- Monthly view --- */}
      {!isWeekly && (
        <>
          {signalsQ.data && <SignalCard data={signalsQ.data} />}

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

          {spendingAlertsQ.data && (
            <SpendingAlerts data={spendingAlertsQ.data} currency={currency} />
          )}

          {forecastQ.data && <ForecastChart forecast={forecastQ.data} />}

          {forecastQ.isError && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-700">
              Forecast unavailable — need at least 3 months of data.
            </div>
          )}
        </>
      )}

      {/* --- Weekly view --- */}
      {isWeekly && (
        <>
          {weeklySignalsQ.data && (
            <WeeklySignalCard data={weeklySignalsQ.data} currency={currency} />
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {categoriesQ.data && (
              <CategoryChart
                categories={categoriesQ.data.categories}
                currency={currency}
                onCategoryClick={handleCategoryClick}
              />
            )}
            {weeklyTrendsQ.data && (
              <WeeklyTrendChart
                weeks={weeklyTrendsQ.data.weeks}
                currency={currency}
              />
            )}
          </div>

          {weeklyForecastQ.data && (
            <WeeklyForecastChart forecast={weeklyForecastQ.data} />
          )}

          {weeklyForecastQ.isError && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-700">
              Weekly forecast unavailable — need at least 4 weeks of data.
            </div>
          )}
        </>
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
