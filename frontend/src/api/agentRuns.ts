import { api } from "./client";
import type { AgentRunOut } from "../types";

interface AgentRunListResponse {
  items: AgentRunOut[];
  total: number;
}

interface CreateAgentRunBody {
  document_ids: string[];
  agent: string;
  question: string;
  llm_provider_id: string;
  llm_model_id: string;
}

interface RetryResponse {
  retried_count: number;
}

export function createAgentRun(body: CreateAgentRunBody): Promise<AgentRunOut> {
  return api.post<AgentRunOut>("/agent-runs", body);
}

export function listAgentRuns(
  limit: number = 50,
  offset: number = 0,
  status?: string,
): Promise<AgentRunListResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) {
    params.append("status", status);
  }
  return api.get<AgentRunListResponse>(`/agent-runs?${params.toString()}`);
}

export function getAgentRun(id: string): Promise<AgentRunOut> {
  return api.get<AgentRunOut>(`/agent-runs/${id}`);
}

export function retryAgentRunItems(
  id: string,
  item_ids?: string[],
): Promise<RetryResponse> {
  const params = new URLSearchParams();
  if (item_ids) {
    item_ids.forEach((iid) => params.append("item_ids", iid));
  }
  const qs = params.toString();
  return api.post<RetryResponse>(
    `/agent-runs/${id}/retry${qs ? `?${qs}` : ""}`,
  );
}

export function deleteAgentRun(id: string): Promise<void> {
  return api.delete<void>(`/agent-runs/${id}`);
}
