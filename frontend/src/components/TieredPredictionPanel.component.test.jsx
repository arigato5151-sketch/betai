import { describe, expect, it } from "vitest";

import { buildTieredFeatures } from "./TieredPredictionPanel.jsx";

describe("buildTieredFeatures", () => {
  it("builds the complete market-aware Tier 1 contract from pre-kickoff data", () => {
    const features = buildTieredFeatures(
      {
        home_gf_last5: 1.8,
        away_gf_last5: 1.1,
        home_form: 80,
        away_form_ema: 50,
        home_elo: 1610,
        away_elo: 1490,
      },
      39,
      "Arsenal",
      "Chelsea",
      { HOME_WIN: 1.9, DRAW: 3.5, AWAY_WIN: 4.2 },
    );

    expect(features).toEqual({
      home_team: "Arsenal",
      away_team: "Chelsea",
      home_avg_goals: 1.8,
      away_avg_goals: 1.1,
      home_form_last5: 2.4,
      away_form_last5: 1.5,
      home_elo: 1610,
      away_elo: 1490,
      opening_home_odd: 1.9,
      opening_draw_odd: 3.5,
      opening_away_odd: 4.2,
    });
    expect(Object.keys(features).some((name) => name.startsWith("closing_"))).toBe(
      false,
    );
  });

  it("omits incomplete market inputs so the backend safely selects Tier 2", () => {
    const features = buildTieredFeatures(
      { home_gf_last5: 1.2, away_gf_last5: 0.9, home_form: 60, away_form: 40 },
      39,
      "Home",
      "Away",
      null,
    );

    expect(features.opening_home_odd).toBeUndefined();
    expect(features.home_form_last5).toBeCloseTo(1.8);
    expect(features.away_form_last5).toBeCloseTo(1.2);
  });
});
