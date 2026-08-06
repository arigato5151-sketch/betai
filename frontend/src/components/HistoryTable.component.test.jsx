import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import HistoryTable, { getResultVerificationBadge } from "./HistoryTable.jsx";

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
      prediction: "DRAW",
      actual_result: "DRAW",
      actual_score_home: 1,
      actual_score_away: 1,
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
  it("sonuç doğrulama durumlarını güven seviyesine göre ayırır", () => {
    expect(
      getResultVerificationBadge({
        actual_result: "HOME_WIN",
        result_verification_status: "verified",
      }).text,
    ).toBe("Provider sonucu doğrulandı");
    expect(
      getResultVerificationBadge({
        actual_result: "DRAW",
        result_verification_status: "conflict",
      }).text,
    ).toContain("karantinada");
    expect(getResultVerificationBadge({ actual_result: null })).toBeNull();
  });
  it("sonuçları Türkçe gösterirken filtre sözleşmesini korur", () => {
    const onFilterChange = vi.fn();
    render(
      <HistoryTable {...defaultProps} onFilterChange={onFilterChange} />,
    );

    expect(screen.getByText("Fenerbahçe – Galatasaray")).toBeInTheDocument();
    expect(
      screen.getByText("Maç sonucu: 1 – 1 · Tahmin doğru"),
    ).toHaveClass("text-emerald-400");

    fireEvent.change(screen.getByLabelText("Sonuç filtresi"), {
      target: { value: "HOME_WIN" },
    });
    expect(onFilterChange).toHaveBeenCalledWith("result", "HOME_WIN");
  });

  it("yanlış tahmini maç skoruyla birlikte kırmızı gösterir", () => {
    render(
      <HistoryTable
        {...defaultProps}
        history={[
          {
            ...defaultProps.history[0],
            prediction: "HOME_WIN",
            actual_result: "AWAY_WIN",
            actual_score_home: 0,
            actual_score_away: 2,
          },
        ]}
      />,
    );

    expect(
      screen.getByText("Maç sonucu: 0 – 2 · Tahmin yanlış"),
    ).toHaveClass("text-red-400");
  });

  it("skor bulunmayan sonuçlanmış maçta sonuç etiketini kullanır", () => {
    render(
      <HistoryTable
        {...defaultProps}
        history={[
          {
            ...defaultProps.history[0],
            prediction: "AWAY_WIN",
            actual_result: "AWAY_WIN",
            actual_score_home: null,
            actual_score_away: null,
          },
        ]}
      />,
    );

    expect(
      screen.getByText("Maç sonucu: Deplasman Kazandı · Tahmin doğru"),
    ).toHaveClass("text-emerald-400");
  });

  it("senaryo ve eğitim dışı kayıtları açıkça işaretler", () => {
    render(
      <HistoryTable
        {...defaultProps}
        history={[
          {
            ...defaultProps.history[0],
            analysis_origin: "scenario",
            eligibility_status: "abstain",
            training_eligible: false,
          },
        ]}
      />,
    );

    expect(screen.getByText("Senaryo · eğitim dışı")).toBeInTheDocument();
  });
});
