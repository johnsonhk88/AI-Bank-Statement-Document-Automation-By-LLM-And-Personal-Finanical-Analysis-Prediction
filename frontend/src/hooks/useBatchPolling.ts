import { useQuery } from "@tanstack/react-query";
import { getAgentRun } from "../api/agentRuns";
import type { AgentRunOut } from "../types";

export function useBatchPolling(id: string | undefined) {
  return useQuery<AgentRunOut>({
    queryKey: ["agent-run", id],
    queryFn: () => getAgentRun(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "pending" || status === "running") {
        return 2000;
      }
      return false;
    },
  });
}
