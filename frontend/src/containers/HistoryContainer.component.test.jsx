import { describe, expect, it } from "vitest";

import { historyItemToSelectedMatch } from "./HistoryContainer.jsx";

describe("historyItemToSelectedMatch", () => {
  it("maps persisted probability and provenance fields into the analysis view", () => {
    const selected = historyItemToSelectedMatch({
      id: 42,
      home_team: "Home",
      away_team: "Away",
      probability: 55,
      prob_home: 55,
      prob_draw: 25,
      prob_away: 20,
      home_form: 70,
      away_form: 60,
      home_xg: 1.8,
      away_xg: 1.1,
      prediction: "HOME_WIN",
      is_value_bet: 1,
      edge: 6.5,
      ml_cluster: null,
      model_name: "Calibrated Model",
      model_artifact_version: "v1",
    });

    expect(selected.match).toBe("Home – Away");
    expect(selected.analysis.all_probabilities).toEqual({
      HOME_WIN: 55,
      AWAY_WIN: 20,
      DRAW: 25,
    });
    expect(selected.value_assessment).toEqual({
      value_bet: true,
      edge: 6.5,
    });
    expect(selected.ml_safety_trigger).toBe("INSUFFICIENT_DATA");
    expect(selected.provenance.model_artifact_version).toBe("v1");
  });

  it("uses safe defaults for legacy history rows", () => {
    const selected = historyItemToSelectedMatch({
      id: 1,
      home_team: "Home",
      away_team: "Away",
      probability: 40,
      prediction: "DRAW",
      is_value_bet: 0,
      ml_cluster: 1,
    });

    expect(selected.analysis.all_probabilities).toEqual({
      HOME_WIN: 40,
      AWAY_WIN: 33,
      DRAW: 34,
    });
    expect(selected.home_stats.attack).toBe(80);
    expect(selected.ml_safety_trigger).toBe("HIGH_CONFIDENCE");
  });

  it("uses the persisted market-aware ML assessment", () => {
    const assessment = {
      trigger: "UPSET_CANDIDATE",
      ml_confidence: 55,
      confidence_gap: 15,
    };
    const selected = historyItemToSelectedMatch({
      id: 2,
      home_team: "Home",
      away_team: "Away",
      probability: 50,
      prediction: "AWAY_WIN",
      is_value_bet: 0,
      ml_cluster: 0,
      ml_confidence: 55,
      data_quality: { ml_assessment: assessment },
    });

    expect(selected.ml_safety_trigger).toBe("UPSET_CANDIDATE");
    expect(selected.ml_safety_details).toEqual(assessment);
  });
});
