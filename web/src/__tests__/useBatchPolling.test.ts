import { describe, it, expect } from "vitest";
import { useBatchPolling } from "../hooks/useBatchPolling";

describe("useBatchPolling", () => {
  it("exists as a module export", () => {
    expect(useBatchPolling).toBeDefined();
    expect(typeof useBatchPolling).toBe("function");
  });

  it("is a valid hook function", () => {
    expect(useBatchPolling).toBeInstanceOf(Function);
    expect(useBatchPolling.length).toBeGreaterThanOrEqual(1);
  });
});
