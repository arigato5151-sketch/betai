import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import HistoryTable from "./HistoryTable.jsx";

const defaultProps = {
  filters: {
    query: "",
    result: "all",
    value: "all",
    sort: "newest",
  },
  history: [
    {
      id: 1,
      home_team: "Fenerbahçe",
      away_team: "Galatasaray",
      actual_result: "DRAW",
      odd: 2.1,
      is_value_bet: 1,
    },
  ],
  historyError: "",
  historyLoading: false,
  meta: { total: 1, pages: 1 },
  onFilterChange: vi.fn(),
  onPageChange: vi.fn(),
  onSelectMatch: vi.fn(),
  page: 1,
};

describe("HistoryTable Türkçe arayüzü", () => {
  it("sonuçları Türkçe gösterirken filtre sözleşmesini korur", () => {
    const onFilterChange = vi.fn();
    render(
      <HistoryTable {...defaultProps} onFilterChange={onFilterChange} />,
    );

    expect(screen.getByText("Fenerbahçe – Galatasaray")).toBeInTheDocument();
    expect(screen.getByText("(Berabere Bitti)")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Sonuç filtresi"), {
      target: { value: "HOME_WIN" },
    });
    expect(onFilterChange).toHaveBeenCalledWith("result", "HOME_WIN");
  });
});
