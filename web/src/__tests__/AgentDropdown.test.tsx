import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import AgentDropdown from "../components/AgentDropdown";

describe("AgentDropdown", () => {
  it("renders 3 options with correct enabled/disabled states", () => {
    render(<AgentDropdown value="crewai" onChange={() => {}} />);

    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(3);

    const crewaiOption = options[0] as HTMLOptionElement;
    expect(crewaiOption.value).toBe("crewai");
    expect(crewaiOption.disabled).toBe(false);

    const deepAgentsOption = options[1] as HTMLOptionElement;
    expect(deepAgentsOption.value).toBe("deep-agents");
    expect(deepAgentsOption.disabled).toBe(true);

    const hermesOption = options[2] as HTMLOptionElement;
    expect(hermesOption.value).toBe("hermes");
    expect(hermesOption.disabled).toBe(true);
  });
});
