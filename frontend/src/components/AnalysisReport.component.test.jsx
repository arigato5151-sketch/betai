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

  it("yüksek tahmin belirsizliğini karar ABSTAIN olarak gösterir", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          data_quality: {
            prediction_eligibility: { status: "eligible", reasons: [] },
            decision_recommendation: {
              status: "abstain",
              reasons: ["probability_margin_too_low"],
            },
          },
        }}
      />,
    );

    expect(screen.getByText("Yüksek tahmin belirsizliği (ABSTAIN)")).toBeInTheDocument();
    expect(
      screen.getByText(/En olası iki sonuç arasındaki fark yetersiz/),
    ).toBeInTheDocument();
    expect(screen.queryByText("probability_margin_too_low")).not.toBeInTheDocument();
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

  it("sonuçlanmış maçta skor ve doğruluk bilgisini gösterir", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          actual_result: "HOME_WIN",
          actual_score_home: 2,
          actual_score_away: 0,
          roi: 12.5,
          result_source: "api_football",
        }}
      />,
    );

    expect(screen.getByText("Maç Sonucu")).toBeInTheDocument();
    expect(screen.getByText("2 – 0")).toBeInTheDocument();
    expect(screen.getByText(/Tahmin doğru ✓/)).toBeInTheDocument();
    expect(screen.getByText(/ROI %12\.50/)).toBeInTheDocument();
    expect(screen.getByText("Sonuç kaynağı: api_football")).toBeInTheDocument();
  });

  it("yanlış tahminde kırmızı doğruluk etiketini gösterir", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          actual_result: "AWAY_WIN",
          actual_score_home: 0,
          actual_score_away: 1,
          roi: -35.0,
        }}
      />,
    );

    expect(screen.getByText(/Tahmin yanlış ✗/)).toBeInTheDocument();
    expect(screen.getByText(/ROI %-35\.00/)).toBeInTheDocument();
  });

  it("sonuç yokken sonuç panelini göstermez", () => {
    render(<AnalysisReport match={match} />);

    expect(screen.queryByText("Maç Sonucu")).not.toBeInTheDocument();
  });

  it("başlamamış maçta manuel sonuç girişini gizler", () => {
    render(
      <AnalysisReport
        canUpdateResult
        match={{
          ...match,
          provenance: { kickoff: "2999-08-09T18:00:00+00:00" },
        }}
        onSubmitActualResult={vi.fn()}
      />,
    );

    expect(screen.queryByText("Gerçek sonucu girin:")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Ev Sahibi Kazandı" }),
    ).not.toBeInTheDocument();
  });

  it("tahmin kaynağı etiketini veri yeterliliğine göre seçer", () => {
    render(<AnalysisReport match={match} />);
    expect(
      screen.getByText(/Sınırlı istatistik analizi/),
    ).toBeInTheDocument();
    expect(screen.getByText("Ev Sahibi Kazanır (%62)")).toBeInTheDocument();
  });

  it("aktif modelde doğrulanmış model tahmini etiketini gösterir", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          ml_ready: true,
          ml_safety_trigger: "HIGH_CONFIDENCE",
        }}
      />,
    );

    expect(
      screen.getByText(/Doğrulanmış model tahmini/),
    ).toBeInTheDocument();
    expect(screen.getByText("Ev Sahibi Kazanır (%62)")).toBeInTheDocument();
  });

  it("abstain kararında tahmin üretilmedi ifadesini gösterir", () => {
    render(
      <AnalysisReport
        match={{
          ...match,
          data_quality: {
            prediction_eligibility: { status: "abstain", reasons: [] },
          },
        }}
      />,
    );

    expect(screen.getByText("Tahmin verilmedi")).toBeInTheDocument();
    expect(
      screen.getByText(/tahmin üretilmedi/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Ev Sahibi Kazanır (%62)")).not.toBeInTheDocument();
  });

  it("sonuç güncelleme hatasını erişilebilir biçimde gösterir ve girişi kilitler", () => {
    render(
      <AnalysisReport
        canUpdateResult
        match={{
          ...match,
          provenance: { kickoff: "2020-01-01T00:00:00+00:00" },
        }}
        onSubmitActualResult={vi.fn()}
        resultError="Sonuç kaydedilemedi."
        resultUpdating
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Sonuç kaydedilemedi.");
    expect(screen.getByRole("button", { name: "Ev Sahibi Kazandı" })).toBeDisabled();
  });
});
