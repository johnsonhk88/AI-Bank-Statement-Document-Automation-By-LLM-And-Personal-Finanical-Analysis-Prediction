export type AgentStatus = "pending" | "running" | "succeeded" | "partial" | "failed";

export interface UserOut {
  id: string;
  email: string;
  is_admin: boolean;
}

export interface AgentInfo {
  name: string;
  display_name: string;
  enabled: boolean;
  description: string;
}

export interface LLMModel {
  id: string;
  display_name: string;
}

export interface LLMProvider {
  id: string;
  display_name: string;
  kind: "local" | "cloud";
  available: boolean;
  unavailable_reason: string | null;
  models: LLMModel[];
}

export interface DocumentOut {
  id: string;
  original_filename: string;
  mime_type: string;
  size_bytes: number | null;
  page_count: number | null;
  deduplicated: boolean;
  created_at: string;
}

export interface Transaction {
  date: string;
  description: string;
  credit: number | null;
  debit: number | null;
  balance: number | null;
  currency: string;
}

export interface AgentRunItemOut {
  id: string;
  document_id: string;
  status: string;
  error: string | null;
  markdown_report: string | null;
  transactions: Transaction[] | null;
  started_at: string | null;
  finished_at: string | null;
}

// --- Cashflow types ---

export interface CategoryAmount {
  category: string;
  amount: number;
  percentage: number;
  transaction_count: number;
}

export interface CashFlowSummary {
  date_from: string;
  date_to: string;
  currency: string;
  total_income: number;
  total_expenses: number;
  net: number;
  transaction_count: number;
  top_expense_categories: CategoryAmount[];
}

export interface CashFlowCategories {
  date_from: string;
  date_to: string;
  currency: string;
  categories: CategoryAmount[];
}

export interface MonthlyTrend {
  month: string;
  income: number;
  expenses: number;
  net: number;
}

export interface CashFlowTrends {
  date_from: string;
  date_to: string;
  currency: string;
  months: MonthlyTrend[];
}

export interface ForecastPoint {
  month: string;
  predicted: number;
  lower: number;
  upper: number;
}

export interface CashFlowForecast {
  currency: string;
  horizon_months: number;
  income: ForecastPoint[];
  expenses: ForecastPoint[];
  net: ForecastPoint[];
}

export interface LineItemOut {
  id: string;
  item_name: string;
  quantity: number;
  unit_price: number;
  total: number;
  sub_category: string;
}

export interface CashFlowTransaction {
  id: string;
  date: string;
  description: string;
  merchant: string;
  amount: number;
  balance: number | null;
  currency: string;
  category: string;
  receipt_id: string | null;
  match_confidence: number | null;
  line_items: LineItemOut[];
}

export interface TransactionList {
  items: CashFlowTransaction[];
  total: number;
  limit: number;
  offset: number;
}

// --- Savings-capacity signals (Phase 1) ---

export interface SignalMonthDetail {
  month: string;
  net: number;
  bb_upper: number | null;
  bb_middle: number | null;
  bb_lower: number | null;
  rsi: number | null;
}

export interface BollingerPosition {
  pband: number | null;
  position:
    | "above_upper"
    | "upper_half"
    | "lower_half"
    | "below_lower"
    | "insufficient_data";
}

export type SignalValue = "INVEST" | "HOLD" | "ALERT";

export interface SavingsCapacityResponse {
  currency: string;
  current_signal: SignalValue;
  rsi: number | null;
  bollinger: BollingerPosition;
  recommendation: string;
  months: SignalMonthDetail[];
}

// --- Spending-trend alerts (Phase 2) ---

export type SpendingTrend = "accelerating" | "decelerating" | "stable";

export interface SpendingMonthDetail {
  month: string;
  amount: number;
  macd: number | null;
  signal: number | null;
  short_sma: number | null;
  long_sma: number | null;
}

export interface CategorySpendingAlert {
  category: string;
  trend: SpendingTrend;
  alert: boolean;
  macd_histogram: number | null;
  sma_crossover: "golden_cross" | "death_cross" | "none";
  recommendation: string;
  months: SpendingMonthDetail[];
}

export interface SpendingAlertsResponse {
  currency: string;
  categories: CategorySpendingAlert[];
  alert_count: number;
  total_categories: number;
}

// --- Weekly signals (Phase 2.7) ---

export interface WeeklyTrend {
  week: string;
  income: number;
  expenses: number;
  net: number;
}

export interface WeeklyTrends {
  date_from: string;
  date_to: string;
  currency: string;
  weeks: WeeklyTrend[];
}

export interface WeeklySignalDetail {
  week: string;
  net: number;
  bb_upper: number | null;
  bb_middle: number | null;
  bb_lower: number | null;
  rsi: number | null;
}

export interface WeekComparison {
  this_week_net: number;
  last_week_net: number;
  week_over_week_change: number;
  vs_4_weeks_ago: number | null;
}

export interface WeeklySignalsResponse {
  currency: string;
  current_signal: SignalValue;
  rsi: number | null;
  bollinger: BollingerPosition;
  recommendation: string;
  weeks: WeeklySignalDetail[];
  comparison: WeekComparison;
}

export interface WeeklyForecastPoint {
  week: string;
  predicted: number;
  lower: number;
  upper: number;
}

export interface WeeklyForecast {
  currency: string;
  horizon_weeks: number;
  income: WeeklyForecastPoint[];
  expenses: WeeklyForecastPoint[];
  net: WeeklyForecastPoint[];
}

// --- Income stability (Phase 3) ---

export type RiskProfile = "LOW" | "MEDIUM" | "HIGH";
export type IncomeTrend = "growing" | "declining" | "stable";

export interface IncomeMonthDetail {
  month: string;
  income: number;
  atr: number | null;
}

export interface IncomeStabilityResponse {
  currency: string;
  risk_profile: RiskProfile;
  atr: number;
  atr_percent: number;
  mean_income: number;
  income_trend: IncomeTrend;
  recommendation: string;
  months: IncomeMonthDetail[];
}

// --- Agent run types ---

export interface AgentRunOut {
  id: string;
  agent: string;
  question: string;
  status: string;
  llm_provider: string;
  llm_model: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  items: AgentRunItemOut[];
}
