import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import AnalysisReport, { buildAlternativeResults } from "./AnalysisReport.jsx";

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
    expected_goals: { home: 1.8, away: 1.1, total: 2.9 },
    expected_score: { home: 1, away: 0, label: "1-0", probability: 14.2 },
    score_band: "0-2 Gol",
    secondary_markets: [
      { market: "OVER_2_5", pick: "UST", probability: 56.4 },
      { market: "BTTS", pick: "VAR", probability: 52.3 },
      { market: "OVER_1_5", pick: "UST", probability: 73.1 },
    ],
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
    expect(screen.getByText("ML Modeli Hazır Değil")).toBeInTheDocument();
    expect(screen.getByText("(80/200 model eğitim örneği)")).toBeInTheDocument();
    expect(screen.getByText(/Poisson \/ Dixon-Coles/)).toBeInTheDocument();
    expect(screen.getByText(/DEĞERLİ ORAN BULUNDU/)).toBeInTheDocument();
    expect(screen.queryByText("HOME_WIN")).not.toBeInTheDocument();
    expect(screen.queryByText("INSUFFICIENT_DATA")).not.toBeInTheDocument();
    expect(screen.getByText("Alternatif Analiz Sonuçları")).toBeInTheDocument();
    expect(screen.getByText("1-X · %85")).toBeInTheDocument();
    expect(screen.getByText("1-0 · %14.2")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Ev Sahibi Kazandı" }),
    );
    expect(onSubmitActualResult).toHaveBeenCalledWith(42, "HOME_WIN");
  });

  it("geçersiz alternatif olasılıkları güvenli biçimde eler", () => {
    expect(
      buildAlternativeResults({
        all_probabilities: {
          HOME_WIN: Number.NaN,
          DRAW: 30,
          AWAY_WIN: 30,
        },
        expected_score: { home: 1, away: 1, probability: 120 },
        secondary_markets: [{ market: "BTTS", pick: "VAR", probability: -1 }],
      }),
    ).toEqual([]);
  });

  it("ML güven yüzdesini ve olasılık farkını gösterir", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          ml_safety_trigger: "UPSET_CANDIDATE",
          ml_safety_details: {
            ml_confidence: 55,
            confidence_gap: 15,
          },
        }}
      />,
    );

    expect(screen.getByText("Sürpriz Adayı")).toBeInTheDocument();
    expect(screen.getByText(/ML %55 · olasılık farkı %15/)).toBeInTheDocument();
  });

  it("ABSTAIN çıktısında değerli bahis mesajını bastırır", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          data_quality: {
            prediction_eligibility: {
              status: "abstain",
              reasons: ["market_unavailable", "home_history_insufficient"],
            },
          },
        }}
      />,
    );

    expect(screen.getByText("Sınırlı veriyle istatistik analizi")).toBeInTheDocument();
    expect(
      screen.getByText(/Güncel 1X2 oranları bulunamadı/),
    ).toBeInTheDocument();
    expect(
      screen.getByText("VERİ YETERSİZ — DEĞER HESABI KULLANILMAMALI"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/DEĞERLİ ORAN BULUNDU/)).not.toBeInTheDocument();
  });

  it("manuel feature değişikliğini senaryo olarak işaretler", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          provenance: { analysis_origin: "scenario" },
        }}
      />,
    );

    expect(screen.getByText("Senaryo analizi")).toBeInTheDocument();
    expect(screen.getByText(/eğitim ve performans hesaplarına katılmaz/)).toBeInTheDocument();
  });
});
