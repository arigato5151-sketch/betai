import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import AnalysisContainer from "./AnalysisContainer.jsx";

vi.mock("react-chartjs-2", () => ({
  Doughnut: () => <div data-testid="olasılık-grafiği" />,
  Line: () => <div data-testid="banka-grafiği" />,
}));

const actions = {
  analyze: true,
  readHistory: true,
  runBacktest: false,
  updateResult: false,
};

const analysisResponse = {
  match: "Fenerbahçe – Galatasaray",
  analysis: {
    prediction: "HOME_WIN",
    probability: 55,
    all_probabilities: {
      HOME_WIN: 55,
      DRAW: 25,
      AWAY_WIN: 20,
    },
  },
  value_assessment: {
    value_bet: false,
    edge: 0,
  },
  ml_safety_trigger: "INSUFFICIENT_DATA",
};

const featurePreviewResponse = {
  feature_schema_version: "ml_features_v8",
  features: [
    {
      name: "fatigue_index",
      label: "Yorgunluk ve Seyahat Endeksi",
      group: "Dinlenme ve seyahat",
      value: 0,
      calculated_value: 0,
      default_value: 0,
      availability: "missing",
      missing_reason: "Yorgunluk hesabı için takvim verisi eksik.",
      overridden: false,
      minimum: -1,
      maximum: 1,
      step: 0.01,
      source: "neutral_default",
      captured_at: null,
      confidence: 0,
      is_fallback: true,
    },
  ],
  derived: {
    expected_goals: { home: 1.4, away: 1.1, total: 2.5 },
  },
  data_quality: { score: 70 },
};

function renderContainer(request, fixtureSelection) {
  const onHistoryChanged = vi.fn();
  const onSelectMatch = vi.fn();
  render(
    <AnalysisContainer
      actions={actions}
      fixtureSelection={fixtureSelection}
      onHistoryChanged={onHistoryChanged}
      onSelectMatch={onSelectMatch}
      request={request}
      selectedMatch={null}
    />,
  );
  return { onHistoryChanged, onSelectMatch };
}

describe("AnalysisContainer lig entegrasyonu", () => {
  it("seçilen fikstürü analiz formuna taşır ve fixture bağlamıyla gönderir", async () => {
    const prefillResponse = {
      fixture: {
        fixture_id: 10,
        home_team_id: 101,
        away_team_id: 202,
        league_id: 848,
        season: 2030,
        kickoff: "2030-07-30T18:00:00+03:00",
      },
      home_team: "Erken Takım",
      away_team: "Rakip A",
      odd: 2.15,
      home_stats: { form: 72, attack: 68, defense: 66, xg: 1.6 },
      away_stats: { form: 61, attack: 64, defense: 60, xg: 1.2 },
      market_1x2: {
        raw_odds: { HOME_WIN: 2.15, DRAW: 3.2, AWAY_WIN: 3.4 },
      },
      opening_odds_1x2: {
        HOME_WIN: 2.4,
        DRAW: 3.1,
        AWAY_WIN: 3.2,
      },
      current_odds_1x2: {
        HOME_WIN: 2.15,
        DRAW: 3.2,
        AWAY_WIN: 3.4,
      },
      opening_odds_at: "2030-07-29T09:00:00+00:00",
      current_odds_at: "2030-07-30T09:00:00+00:00",
    };
    const request = vi.fn(async (path) => {
      if (path === "/leagues") {
        return {
          ok: true,
          json: async () => [
            { id: 848, name: "UEFA Europa Conference League" },
          ],
        };
      }
      if (path === "/fixtures/10/prefill") {
        return {
          ok: true,
          json: async () => prefillResponse,
        };
      }
      if (path === "/analyze/preview") {
        return {
          ok: true,
          json: async () => featurePreviewResponse,
        };
      }
      if (path === "/analyze") {
        return {
          ok: true,
          json: async () => analysisResponse,
        };
      }
      throw new Error(`Beklenmeyen istek: ${path}`);
    });
    renderContainer(request, {
      fixture: { fixture_id: 10 },
      revision: 1,
    });

    expect(await screen.findByDisplayValue("Erken Takım")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Rakip A")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Lig" })).toHaveValue("848");
    expect(screen.getByLabelText("Bahis oranı")).toHaveValue(2.15);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Erken Takım – Rakip A analiz formuna yüklendi.",
    );
    expect(screen.getByText(/Kaynak:/)).toHaveTextContent(
      "Kaynak: Nötr varsayılan · Güven %0 · fallback",
    );
    fireEvent.change(
      screen.getByLabelText("Yorgunluk ve Seyahat Endeksi"),
      { target: { value: "0.4" } },
    );

    fireEvent.click(screen.getByRole("button", { name: "Tahmin Oluştur" }));
    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(
        "/analyze",
        expect.objectContaining({ method: "POST" }),
      );
    });

    const analyzeCall = request.mock.calls.find(([path]) => path === "/analyze");
    expect(JSON.parse(analyzeCall[1].body)).toMatchObject({
      fixture_id: 10,
      home_team_id: 101,
      away_team_id: 202,
      league_id: 848,
      season: 2030,
      kickoff: "2030-07-30T18:00:00+03:00",
      market_1x2: prefillResponse.market_1x2,
      opening_odds_1x2: prefillResponse.opening_odds_1x2,
      current_odds_1x2: prefillResponse.current_odds_1x2,
      opening_odds_at: prefillResponse.opening_odds_at,
      current_odds_at: prefillResponse.current_odds_at,
      feature_overrides: { fatigue_index: 0.4 },
    });
  });

  it("ligleri yükler ve seçilen sayısal league_id değerini analize gönderir", async () => {
    const request = vi.fn(async (path) => {
      if (path === "/leagues") {
        return {
          ok: true,
          json: async () => [
            { id: 2, name: "UEFA Champions League" },
            { id: 39, name: "Premier League" },
          ],
        };
      }
      if (path === "/analyze") {
        return {
          ok: true,
          json: async () => analysisResponse,
        };
      }
      if (path === "/analyze/preview") {
        return {
          ok: true,
          json: async () => featurePreviewResponse,
        };
      }
      throw new Error(`Beklenmeyen istek: ${path}`);
    });
    const { onHistoryChanged, onSelectMatch } = renderContainer(request);

    const leagueSelect = await screen.findByRole("combobox", { name: "Lig" });
    await screen.findByRole("option", { name: "UEFA Şampiyonlar Ligi" });
    fireEvent.change(leagueSelect, { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Tahmin Oluştur" }));

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(
        "/analyze",
        expect.objectContaining({ method: "POST" }),
      );
    });

    const analyzeCall = request.mock.calls.find(([path]) => path === "/analyze");
    const payload = JSON.parse(analyzeCall[1].body);
    expect(payload.league_id).toBe(2);
    expect(typeof payload.league_id).toBe("number");
    expect(onHistoryChanged).toHaveBeenCalledOnce();
    expect(onSelectMatch).toHaveBeenCalledOnce();
  });

  it("lig isteği başarısız olduğunda manuel analizi kullanılabilir tutar", async () => {
    const request = vi.fn(async (path) => {
      if (path === "/leagues") {
        throw new Error("Ağ hatası");
      }
      if (path === "/analyze/preview") {
        return {
          ok: true,
          json: async () => featurePreviewResponse,
        };
      }
      if (path === "/analyze") {
        return {
          ok: true,
          json: async () => analysisResponse,
        };
      }
      throw new Error(`Beklenmeyen istek: ${path}`);
    });
    const { onHistoryChanged } = renderContainer(request);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Lig seçmeden manuel analize devam edebilirsiniz",
    );
    expect(screen.getByLabelText("Ev sahibi takım")).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Tahmin Oluştur" }));

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(
        "/analyze",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const analyzeCall = request.mock.calls.find(([path]) => path === "/analyze");
    expect(JSON.parse(analyzeCall[1].body).league_id).toBeNull();
    expect(onHistoryChanged).toHaveBeenCalledOnce();
  });
});
