import { useQuery } from "@tanstack/react-query";
import { listLLMModels } from "../api/llm";

export function useLLMModels() {
  return useQuery({
    queryKey: ["llm-models"],
    queryFn: listLLMModels,
    refetchInterval: 60000,
  });
}
