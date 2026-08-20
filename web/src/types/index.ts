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
