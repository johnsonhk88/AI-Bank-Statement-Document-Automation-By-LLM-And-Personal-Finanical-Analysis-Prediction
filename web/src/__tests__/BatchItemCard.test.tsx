import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import BatchItemCard from "../components/BatchItemCard";
import type { AgentRunItemOut } from "../types";

const baseItem: AgentRunItemOut = {
  id: "item-1",
  document_id: "doc-1",
  status: "pending",
  error: null,
  markdown_report: null,
  transactions: null,
  started_at: null,
  finished_at: null,
};

describe("BatchItemCard", () => {
  it("renders pending status correctly", () => {
    render(
      <BatchItemCard
        item={{ ...baseItem, status: "pending" }}
        documentFilename="statement.pdf"
      />,
    );
    expect(screen.getByText("statement.pdf")).toBeDefined();
    expect(screen.getByText("Pending")).toBeDefined();
  });

  it("renders succeeded status correctly", () => {
    const item: AgentRunItemOut = {
      ...baseItem,
      status: "succeeded",
      markdown_report: "# Report\n\nBalance: 1000",
      transactions: [
        { date: "2024-01-01", description: "Deposit", credit: 500, debit: null, balance: 1500, currency: "HKD" },
      ],
    };
    render(<BatchItemCard item={item} documentFilename="stmt.pdf" />);
    expect(screen.getByText("Succeeded")).toBeDefined();
    expect(screen.getByText("View details")).toBeDefined();
  });

  it("renders failed status with error and retry button", () => {
    const item: AgentRunItemOut = {
      ...baseItem,
      status: "failed",
      error: "Something went wrong",
    };
    render(<BatchItemCard item={item} />);
    expect(screen.getByText("Failed")).toBeDefined();
    expect(screen.getByText("Something went wrong")).toBeDefined();
    expect(screen.getByText("Retry")).toBeDefined();
  });

  it("shows document id as fallback when no filename provided", () => {
    render(<BatchItemCard item={{ ...baseItem, document_id: "doc-abc-123" }} />);
    expect(screen.getByText("doc-abc-123")).toBeDefined();
  });

  it("expands to show report and transactions when View details is clicked", () => {
    const item: AgentRunItemOut = {
      ...baseItem,
      status: "succeeded",
      markdown_report: "**Hello World**",
      transactions: [
        { date: "2024-01-01", description: "Paycheck", credit: 5000, debit: null, balance: 5000, currency: "HKD" },
      ],
    };
    render(<BatchItemCard item={item} documentFilename="st.pdf" />);

    expect(screen.queryByText("Hello World")).toBeNull();

    fireEvent.click(screen.getByText("View details"));

    expect(screen.getByText("Hello World")).toBeDefined();
    expect(screen.getByText("Paycheck")).toBeDefined();
  });

  it("calls onRetry when retry button is clicked", () => {
    let retriedId = "";
    const handleRetry = (id: string) => {
      retriedId = id;
    };
    const item: AgentRunItemOut = {
      ...baseItem,
      status: "failed",
      error: "Error",
    };
    render(<BatchItemCard item={item} onRetry={handleRetry} />);
    fireEvent.click(screen.getByText("Retry"));
    expect(retriedId).toBe("item-1");
  });
});
