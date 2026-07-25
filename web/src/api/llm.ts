import { api } from "./client";
import type { LLMProvider } from "../types";

interface LLMCatalogResponse {
  providers: LLMProvider[];
}

export function listLLMModels(): Promise<LLMCatalogResponse> {
  return api.get<LLMCatalogResponse>("/llm-models");
}
