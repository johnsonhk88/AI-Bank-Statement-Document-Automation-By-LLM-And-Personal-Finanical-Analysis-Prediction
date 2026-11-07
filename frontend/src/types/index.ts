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
