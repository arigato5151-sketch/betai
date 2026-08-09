import { useEffect, useState } from "react";

import { validatePrefillResponse } from "../apiSchema.js";
import { useRequestGate } from "./useRequestGate.js";

export const initialFormData = {
  home_team: "",
  away_team: "",
  league_id: null,
  odd: 2.3,
  home_stats: { form: 93, attack: 88, defense: 85, xg: 2.15 },
  away_stats: { form: 73, attack: 82, defense: 75, xg: 1.9 },
  feature_overrides: {},
};

const boundedNumber = (value, fallback, minimum, maximum) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(maximum, Math.max(minimum, numeric));
};

const optionalPositiveInteger = (value) =>
  Number.isInteger(value) && value > 0 ? value : null;

export const normalizeTeamStats = (stats) => ({
  form: boundedNumber(stats?.form, 50, 0, 100),
  attack: boundedNumber(stats?.attack, 50, 0, 100),
  defense: boundedNumber(stats?.defense, 50, 0, 100),
  xg: boundedNumber(stats?.xg, 1.2, 0, 5),
});

export function buildFixtureFormData(prefill) {
  validatePrefillResponse(prefill);

  const fixture =
    prefill.fixture && typeof prefill.fixture === "object"
      ? prefill.fixture
      : {};
  const homeTeam = String(prefill.home_team ?? fixture.home_team ?? "").trim();
  const awayTeam = String(prefill.away_team ?? fixture.away_team ?? "").trim();
  const odd = Number(prefill.odd);
  if (!homeTeam || !awayTeam || !Number.isFinite(odd) || odd <= 1) {
    throw new TypeError("Maç ön dolum yanıtında zorunlu alanlar eksik.");
  }

  const kickoff =
    typeof fixture.kickoff === "string" &&
    Number.isFinite(Date.parse(fixture.kickoff))
      ? fixture.kickoff
      : null;

  return {
    home_team: homeTeam,
    away_team: awayTeam,
    odd,
    home_stats: normalizeTeamStats(prefill.home_stats),
    away_stats: normalizeTeamStats(prefill.away_stats),
    market_1x2:
      prefill.market_1x2 && typeof prefill.market_1x2 === "object"
        ? prefill.market_1x2
        : null,
    opening_odds_1x2:
      prefill.opening_odds_1x2 &&
      typeof prefill.opening_odds_1x2 === "object"
        ? prefill.opening_odds_1x2
        : null,
    current_odds_1x2:
      prefill.current_odds_1x2 &&
      typeof prefill.current_odds_1x2 === "object"
        ? prefill.current_odds_1x2
        : null,
    opening_odds_at:
      typeof prefill.opening_odds_at === "string"
        ? prefill.opening_odds_at
        : null,
    current_odds_at:
      typeof prefill.current_odds_at === "string"
        ? prefill.current_odds_at
        : null,
    fixture_id: optionalPositiveInteger(fixture.fixture_id),
    home_team_id: optionalPositiveInteger(fixture.home_team_id),
    away_team_id: optionalPositiveInteger(fixture.away_team_id),
    league_id: optionalPositiveInteger(fixture.league_id),
    season: optionalPositiveInteger(fixture.season),
    kickoff,
    away_travel_distance_km:
      Number.isFinite(Number(fixture.away_travel_distance_km)) &&
      Number(fixture.away_travel_distance_km) >= 0
        ? Number(fixture.away_travel_distance_km)
        : null,
    feature_overrides: {},
  };
}

export function useFixturePrefill({
  actions,
  fixtureSelection,
  formData,
  onClearFixtureSelection,
  request,
  setFormData,
}) {
  const { stale, begin, settle } = useRequestGate();
  const [fixtureLoading, setFixtureLoading] = useState(false);
  const [fixtureError, setFixtureError] = useState("");
  const [fixtureMessage, setFixtureMessage] = useState("");

  useEffect(() => {
    const selectedFixture = fixtureSelection?.fixture;
    if (!actions.analyze || !selectedFixture?.fixture_id) return undefined;

    let disposed = false;
    const mine = begin();
    setFixtureLoading(true);
    setFixtureError("");
    setFixtureMessage("");

    const loadFixture = async () => {
      try {
        const response = await request(
          `/fixtures/${selectedFixture.fixture_id}/prefill`,
        );
        if (!response.ok) {
          const errorBody = await response.json().catch(() => ({}));
          throw new Error(errorBody.detail || "Maç verileri alınamadı.");
        }
        const nextFormData = buildFixtureFormData(await response.json());
        settle(mine, () => {
          if (disposed) return;
          setFormData(nextFormData);
          setFixtureMessage(
            `${nextFormData.home_team} – ${nextFormData.away_team} analiz formuna yüklendi.`,
          );
        });
      } catch (error) {
        settle(mine, () => {
          if (disposed) return;
          setFixtureError(error.message || "Maç verileri alınamadı.");
        });
      } finally {
        if (!disposed) setFixtureLoading(false);
      }
    };

    loadFixture();
    return () => {
      disposed = true;
    };
  }, [actions.analyze, fixtureSelection, request, setFormData]);

  const handleFormChange = (nextFormData) => {
    const fixtureIdentityChanged =
      formData.fixture_id &&
      (nextFormData.home_team !== formData.home_team ||
        nextFormData.away_team !== formData.away_team ||
        nextFormData.league_id !== formData.league_id);

    if (!fixtureIdentityChanged) {
      setFormData(nextFormData);
      return;
    }

    setFixtureMessage("");
    setFixtureError("");
    onClearFixtureSelection?.();
    setFormData({
      ...nextFormData,
      fixture_id: null,
      home_team_id: null,
      away_team_id: null,
      season: null,
      kickoff: null,
      market_1x2: null,
      opening_odds_1x2: null,
      current_odds_1x2: null,
      opening_odds_at: null,
      current_odds_at: null,
      away_travel_distance_km: null,
      feature_overrides: {},
    });
  };

  return {
    fixtureError,
    fixtureLoading,
    fixtureMessage,
    fixtureStale: stale,
    handleFormChange,
  };
}