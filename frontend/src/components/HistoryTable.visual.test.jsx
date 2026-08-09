import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import HistoryTable from "./HistoryTable.jsx";

const longHomeTeam =
  "SpVgg Greuther Fürth – Kampfgemeinschaft Wacker Nord 05".repeat(3).trim();
const longAwayTeam =
  "Turn- und Sportverein 1860 München Amateure e.V.".repeat(3).trim();

const buildProps = (history) => ({
  filters: { query: "", result: "all", value: "all", sort: "newest" },
  history,
  historyError: "",
  historyLoading: false,
  meta: { total: history.length, pages: 1 },
  onFilterChange: vi.fn(),
  onPageChange: vi.fn(),
  onSelectMatch: vi.fn(),
  page: 1,
});

describe("HistoryTable görsel regresyon sözleşmesi", () => {
  it("uzun takım adlarını kırpma ve tam değer sözleşmesiyle gösterir", () => {
    const record = {
      id: 7,
      home_team: longHomeTeam,
      away_team: longAwayTeam,
      prediction: "AWAY_WIN",
      actual_result: null,
      odd: 3.1,
      is_value_bet: 0,
    };
    render(<HistoryTable {...buildProps([record])} />);

    const title = `${longHomeTeam} – ${longAwayTeam}`;
    const nameCell = screen.getByTitle(title);
    expect(nameCell).toHaveClass("truncate");
    expect(nameCell).toHaveAttribute("title", title);
    expect(nameCell).toHaveTextContent(`${longHomeTeam} – ${longAwayTeam}`);

    const card = nameCell.closest("button");
    expect(card).toHaveClass("flex-col");
    expect(card).toHaveClass("sm:flex-row");

    expect(screen.getByRole("button")).toMatchSnapshot();
  });

  it("mobil daraltılmış düzeni bozmadan hizalanır", () => {
    const record = {
      id: 8,
      home_team: "PEC Zwolle",
      away_team: "Fortuna Sittard",
      prediction: "DRAW",
      actual_result: "DRAW",
      actual_score_home: 1,
      actual_score_away: 1,
      odd: 2.4,
      is_value_bet: 1,
    };
    render(<HistoryTable {...buildProps([record])} />);

    const nameCell = screen.getByTitle("PEC Zwolle – Fortuna Sittard");
    expect(nameCell.className).toContain("truncate");
    expect(screen.getByRole("button")).toMatchSnapshot();
  });
});