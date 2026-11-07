import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ModelDropdown from "../components/ModelDropdown";
import type { LLMProvider } from "../types";

vi.mock("../hooks/useLLMModels", () => ({
  useLLMModels: vi.fn(),
}));

import { useLLMModels } from "../hooks/useLLMModels";

const fakeProviders: { providers: LLMProvider[] } = {
  providers: [
    {
      id: "openai",
      display_name: "OpenAI",
      kind: "cloud",
      available: true,
      unavailable_reason: null,
      models: [
        { id: "openai/gpt-4o-mini", display_name: "GPT-4o Mini" },
        { id: "openai/gpt-4o", display_name: "GPT-4o" },
      ],
    },
    {
      id: "lm-studio",
      display_name: "LM Studio",
      kind: "local",
      available: false,
      unavailable_reason: "Connection refused",
      models: [
        { id: "openai/qwen2.5-14b-instruct", display_name: "Qwen 2.5 14B" },
      ],
    },
  ],
};

function renderWithQuery(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

describe("ModelDropdown", () => {
  it("renders provider and model select options from mock data", () => {
    vi.mocked(useLLMModels).mockReturnValue({
      data: fakeProviders,
      isLoading: false,
      isError: false,
      isSuccess: true,
      status: "success",
      dataUpdatedAt: Date.now(),
      error: null,
      errorUpdateCount: 0,
      fetchStatus: "idle",
      isFetched: true,
      isFetchedAfterMount: true,
      isFetching: false,
      isPending: false,
      isLoadingError: false,
      isPaused: false,
      isPlaceholderData: false,
      isRefetchError: false,
      isRefetching: false,
      isStale: false,
      failureCount: 0,
      failureReason: null,
      isInitialLoading: false,
      refetch: vi.fn(),
      remove: vi.fn(),
      promise: Promise.resolve(fakeProviders),
    } as unknown as ReturnType<typeof useLLMModels>);

    renderWithQuery(
      <ModelDropdown
        providerId="openai"
        modelId="openai/gpt-4o-mini"
        onProviderChange={() => {}}
      />,
    );

    const selects = screen.getAllByRole("combobox");
    expect(selects).toHaveLength(2);

    const providerOptions = selects[0].querySelectorAll("option");
    expect(providerOptions).toHaveLength(2);

    const lmStudioOption = providerOptions[1] as HTMLOptionElement;
    expect(lmStudioOption.disabled).toBe(true);
  });
});
