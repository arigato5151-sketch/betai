import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import AnalysisReport from "./AnalysisReport.jsx";

vi.mock("react-chartjs-2", () => ({
  Doughnut: () => <div data-testid="olasılık-grafiği" />,
}));

const match = {
  match: "Fenerbahçe vs Galatasaray",
  analysis: {
    prediction: "HOME_WIN",
    probability: 62,
    all_probabilities: {
      HOME_WIN: 62,
      DRAW: 23,
      AWAY_WIN: 15,
    },
  },
  value_assessment: {
    value_bet: true,
    edge: 7.4,
  },
  ml_safety_trigger: "INSUFFICIENT_DATA",
  ml_samples: 80,
  ml_min_samples: 200,
  record_id: 42,
};

describe("AnalysisReport Türkçe gösterim katmanı", () => {
  it("teknik kodları Türkçe gösterir ve sonucu API koduyla gönderir", () => {
    const onSubmitActualResult = vi.fn();
    render(
      <AnalysisReport
        canUpdateResult
        match={match}
        onSubmitActualResult={onSubmitActualResult}
      />,
    );

    expect(screen.getByText("Fenerbahçe – Galatasaray")).toBeInTheDocument();
    expect(screen.getByText(/Ev Sahibi Kazanır/)).toBeInTheDocument();
    expect(screen.getByText("Yetersiz Veri")).toBeInTheDocument();
    expect(screen.getByText(/DEĞERLİ ORAN BULUNDU/)).toBeInTheDocument();
    expect(screen.queryByText("HOME_WIN")).not.toBeInTheDocument();
    expect(screen.queryByText("INSUFFICIENT_DATA")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Ev Sahibi Kazandı" }),
    );
    expect(onSubmitActualResult).toHaveBeenCalledWith(42, "HOME_WIN");
  });
});
