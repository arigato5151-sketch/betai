import { useEffect, useMemo, useRef, useState } from "react";

import { buildBankrollSeries } from "../bankroll.js";
import AnalysisForm from "../components/AnalysisForm.jsx";
import AnalysisReport from "../components/AnalysisReport.jsx";
import BankrollChart from "../components/BankrollChart.jsx";
import TieredPredictionPanel from "../components/TieredPredictionPanel.jsx";

const initialFormData = {
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

const normalizeTeamStats = (stats) => ({
  form: boundedNumber(stats?.form, 50, 0, 100),
  attack: boundedNumber(stats?.attack, 50, 0, 100),
  defense: boundedNumber(stats?.defense, 50, 0, 100),
  xg: boundedNumber(stats?.xg, 1.2, 0, 5),
});

export function buildFixtureFormData(prefill) {
  if (!prefill || typeof prefill !== "object") {
    throw new TypeError("Maç ön dolum yanıtı geçerli değil.");
  }

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

function normalizeLeagues(payload) {
  if (!Array.isArray(payload)) {
    throw new TypeError("Lig listesi geçerli bir dizi değil.");
  }

  const uniqueLeagues = new Map();
  payload.forEach((league) => {
    if (
      Number.isInteger(league?.id) &&
      league.id > 0 &&
      typeof league.name === "string" &&
      league.name.trim()
    ) {
      uniqueLeagues.set(league.id, {
        ...league,
        name: league.name.trim(),
      });
    }
  });
  return [...uniqueLeagues.values()];
}

function AnalysisContainer({
  actions,
  children,
  fixtureSelection,
  onClearFixtureSelection,
  onHistoryChanged,
  onSelectMatch,
  request,
  selectedMatch,
}) {
  const [formData, setFormData] = useState(initialFormData);
  const [loading, setLoading] = useState(false);
  const [fixtureLoading, setFixtureLoading] = useState(false);
  const [fixtureError, setFixtureError] = useState("");
  const [fixtureMessage, setFixtureMessage] = useState("");
  const [leagues, setLeagues] = useState([]);
  const [leaguesLoading, setLeaguesLoading] = useState(true);
  const [leaguesError, setLeaguesError] = useState("");
  const [backtest, setBacktest] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState("");
  const analysisSectionRef = useRef(null);
  const bankrollSeries = useMemo(
    () => buildBankrollSeries(backtest?.bankroll_history),
    [backtest],
  );

  useEffect(() => {
    let isActive = true;

    const loadLeagues = async () => {
      setLeaguesLoading(true);
      setLeaguesError("");
      try {
        const response = await request("/leagues");
        if (!response.ok) {
          throw new Error("Lig listesi isteği başarısız oldu.");
        }
        const loadedLeagues = normalizeLeagues(await response.json());
        if (isActive) {
          setLeagues(loadedLeagues);
        }
      } catch {
        if (isActive) {
          setLeagues([]);
          setLeaguesError(
            "Desteklenen ligler alınamadı. Lig seçmeden manuel analize devam edebilirsiniz.",
          );
        }
      } finally {
        if (isActive) {
          setLeaguesLoading(false);
        }
      }
    };

    loadLeagues();
    return () => {
      isActive = false;
    };
  }, [request]);

  useEffect(() => {
    const selectedFixture = fixtureSelection?.fixture;
    if (!actions.analyze || !selectedFixture?.fixture_id) return undefined;

    let isActive = true;
    analysisSectionRef.current?.scrollIntoView?.({
      behavior: "smooth",
      block: "start",
    });
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
        if (isActive) {
          setFormData(nextFormData);
          setFixtureMessage(
            `${nextFormData.home_team} – ${nextFormData.away_team} analiz formuna yüklendi.`,
          );
          analysisSectionRef.current?.scrollIntoView?.({
            behavior: "smooth",
            block: "start",
          });
        }
      } catch (error) {
        if (isActive) {
          setFixtureError(error.message || "Maç verileri alınamadı.");
        }
      } finally {
        if (isActive) {
          setFixtureLoading(false);
        }
      }
    };

    loadFixture();
    return () => {
      isActive = false;
    };
  }, [
    actions.analyze,
    fixtureSelection,
    request,
  ]);

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

  const submitActualResult = async (recordId, result) => {
    if (!actions.updateResult) return;
    try {
      const response = await request(`/history/${recordId}/result`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actual_result: result }),
      });
      if (!response.ok) throw new Error("Sonuç kaydedilemedi.");
      await response.json();
      onHistoryChanged();
    } catch (error) {
      alert(error.message || "Sonuç kaydedilemedi.");
    }
  };

  const runBacktest = async () => {
    if (!actions.runBacktest) return;
    setBacktestLoading(true);
    setBacktestError("");
    try {
      const response = await request("/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_bankroll: 1000,
          strategy: "fractional_kelly",
          kelly_fraction: 0.25,
          min_edge_pct: 3,
          commission_pct: 2,
          max_stake_pct: 5,
          max_daily_exposure_pct: 15,
          require_closing_odds: false,
          exclude_post_kickoff: true,
        }),
      });
      if (!response.ok) throw new Error("Geriye dönük test çalıştırılamadı.");
      setBacktest(await response.json());
    } catch (error) {
      setBacktestError(error.message || "Geriye dönük test çalıştırılamadı.");
    } finally {
      setBacktestLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!actions.analyze) return;
    setLoading(true);
    try {
      const response = await request("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error("Analiz isteği başarısız oldu.");
      const data = await response.json();
      onHistoryChanged();
      onSelectMatch({
        match: data.match,
        odd: formData.odd,
        home_team: formData.home_team,
        away_team: formData.away_team,
        league_id: formData.league_id,
        feature_snapshot: data.feature_snapshot,
        home_stats: formData.home_stats,
        away_stats: formData.away_stats,
        analysis: data.analysis,
        value_assessment: data.value_assessment,
        ml_safety_trigger: data.ml_safety_trigger,
        ml_safety_details: data.ml_safety_details,
        ml_confidence: data.ml_confidence,
        ml_ready: data.ml_ready,
        ml_samples: data.ml_samples,
        ml_min_samples: data.ml_min_samples,
        data_quality: data.data_quality,
        provenance: data.provenance,
        insights: data.insights,
      });
    } catch (error) {
      alert(error.message || "Analiz isteği başarısız oldu.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <section
        id="match-analysis-section"
        ref={analysisSectionRef}
        className="scroll-mt-6 lg:col-span-3"
      >
        {fixtureError && (
          <p
            role="alert"
            className="mb-3 rounded-lg border border-red-900/70 bg-red-950/40 p-3 text-sm text-red-300"
          >
            {fixtureError}
          </p>
        )}
        {fixtureMessage && !fixtureError && (
          <p
            role="status"
            className="mb-3 rounded-lg border border-emerald-900/70 bg-emerald-950/40 p-3 text-sm text-emerald-300"
          >
            {fixtureMessage}
          </p>
        )}
        {actions.analyze ? (
          <AnalysisForm
            fixtureLoading={fixtureLoading}
            formData={formData}
            leagues={leagues}
            leaguesError={leaguesError}
            leaguesLoading={leaguesLoading}
            loading={loading}
            onChange={handleFormChange}
            onSubmit={handleSubmit}
          />
        ) : (
          <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400 shadow-xl">
            <h2 className="mb-2 font-bold text-slate-200">
              Salt Okunur Oturum
            </h2>
            <p>Bu rol yeni analiz oluşturma yetkisine sahip değil.</p>
          </div>
        )}
      </section>

      <div className="space-y-6 lg:col-span-3">
        {selectedMatch && (
          <AnalysisReport
            canUpdateResult={actions.updateResult}
            match={selectedMatch}
            onSubmitActualResult={submitActualResult}
          />
        )}

        {selectedMatch && actions.analyze && (
          <TieredPredictionPanel
            request={request}
            leagueId={selectedMatch.league_id}
            homeTeam={selectedMatch.home_team}
            awayTeam={selectedMatch.away_team}
            featureSnapshot={selectedMatch.feature_snapshot}
            enabled
          />
        )}

        {actions.runBacktest && (
          <BankrollChart
            backtest={backtest}
            bankrollSeries={bankrollSeries}
            error={backtestError}
            loading={backtestLoading}
            onRun={runBacktest}
          />
        )}

        {children}
      </div>
    </>
  );
}

export default AnalysisContainer;
