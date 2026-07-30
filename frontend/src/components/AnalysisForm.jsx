import { leagueLabel } from "../localization.js";
import AnalysisInputsPanel from "./AnalysisInputsPanel.jsx";

const TEAM_STAT_FIELDS = [
  { key: "form", label: "Form", maximum: 100, step: 1 },
  { key: "attack", label: "Hücum Gücü", maximum: 100, step: 1 },
  { key: "defense", label: "Savunma Gücü", maximum: 100, step: 1 },
  { key: "xg", label: "Gol Beklentisi (xG)", maximum: 5, step: 0.01 },
];

const MARKET_OUTCOMES = [
  { key: "HOME_WIN", label: "Ev sahibi oranı (1)" },
  { key: "DRAW", label: "Beraberlik oranı (X)" },
  { key: "AWAY_WIN", label: "Deplasman oranı (2)" },
];

function devigMarket(rawOdds) {
  const values = Object.values(rawOdds).map(Number);
  if (values.length !== 3 || values.some((value) => value <= 1)) {
    return { raw_odds: rawOdds };
  }
  const implied = Object.fromEntries(
    Object.entries(rawOdds).map(([key, value]) => [key, 1 / Number(value)]),
  );
  const total = Object.values(implied).reduce((sum, value) => sum + value, 0);
  const fairProbability = Object.fromEntries(
    Object.entries(implied).map(([key, value]) => [
      key,
      Number(((value / total) * 100).toFixed(2)),
    ]),
  );
  return {
    raw_odds: Object.fromEntries(
      Object.entries(rawOdds).map(([key, value]) => [key, Number(value)]),
    ),
    implied_probability: Object.fromEntries(
      Object.entries(implied).map(([key, value]) => [
        key,
        Number((value * 100).toFixed(2)),
      ]),
    ),
    fair_probability: fairProbability,
    fair_odds: Object.fromEntries(
      Object.entries(fairProbability).map(([key, value]) => [
        key,
        Number((100 / value).toFixed(2)),
      ]),
    ),
    overround_pct: Number((total * 100 - 100).toFixed(2)),
    method: "manual_proportional_devig",
  };
}

function TeamStatsEditor({ accentClass, label, onChange, side, stats }) {
  return (
    <fieldset className="rounded-lg border border-slate-800 p-3">
      <legend className={`px-1 text-xs font-black ${accentClass}`}>{label}</legend>
      <div className="grid gap-2 sm:grid-cols-2">
        {TEAM_STAT_FIELDS.map((field) => (
          <div key={field.key}>
            <label
              htmlFor={`${side}-${field.key}`}
              className="mb-1 block text-[10px] font-bold text-slate-400"
            >
              {field.label}
            </label>
            <input
              id={`${side}-${field.key}`}
              type="number"
              min="0"
              max={field.maximum}
              step={field.step}
              value={stats[field.key]}
              onChange={(event) =>
                onChange(field.key, Number(event.target.value))
              }
              className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2 text-sm"
            />
          </div>
        ))}
      </div>
    </fieldset>
  );
}

function AnalysisForm({
  featureError = "",
  formData,
  fixtureLoading = false,
  featureLoading = false,
  featurePreview = null,
  featurePreviewStale = false,
  leagues = [],
  leaguesError = "",
  leaguesLoading = false,
  loading,
  onChange,
  onFeatureOverride,
  onFeatureRefresh,
  onFeatureReset,
  onSubmit,
}) {
  const hasLeagues = leagues.length > 0;
  const leagueSelectDisabled = leaguesLoading || !hasLeagues;
  const leagueStatusId = "analysis-league-status";
  const hasLeagueStatus = leaguesLoading || Boolean(leaguesError) || !hasLeagues;

  const handleLeagueChange = (event) => {
    const parsedLeagueId = Number.parseInt(event.target.value, 10);
    onChange({
      ...formData,
      league_id:
        Number.isInteger(parsedLeagueId) && parsedLeagueId > 0
          ? parsedLeagueId
          : null,
    });
  };

  const updateTeamStat = (side, key, value) => {
    onChange({
      ...formData,
      [`${side}_stats`]: {
        ...formData[`${side}_stats`],
        [key]: value,
      },
    });
  };

  const updateMarketOdd = (outcome, rawValue) => {
    const currentRaw = {
      HOME_WIN:
        formData.market_1x2?.raw_odds?.HOME_WIN ?? formData.odd,
      DRAW: formData.market_1x2?.raw_odds?.DRAW ?? "",
      AWAY_WIN: formData.market_1x2?.raw_odds?.AWAY_WIN ?? "",
    };
    const nextRaw = {
      ...currentRaw,
      [outcome]: rawValue === "" ? "" : Number(rawValue),
    };
    const nextData = {
      ...formData,
      market_1x2: devigMarket(nextRaw),
      // Manuel piyasa değişikliği, otomatik snapshot zaman serisini geçersiz kılar.
      opening_odds_1x2: null,
      current_odds_1x2: null,
      opening_odds_at: null,
      current_odds_at: null,
    };
    if (outcome === "HOME_WIN" && Number(rawValue) > 1) {
      nextData.odd = Number(rawValue);
    }
    onChange(nextData);
  };

  return (
    <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
      <h2 className="mb-4 text-lg font-bold">Manuel Maç Analizi</h2>
      <form onSubmit={onSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="analysis-league"
            className="mb-1 block text-xs font-bold text-slate-300"
          >
            Lig
          </label>
          <select
            id="analysis-league"
            aria-busy={leaguesLoading}
            aria-describedby={hasLeagueStatus ? leagueStatusId : undefined}
            className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm disabled:cursor-not-allowed disabled:opacity-60"
            disabled={leagueSelectDisabled}
            value={formData.league_id ?? ""}
            onChange={handleLeagueChange}
          >
            <option value="">
              {leaguesLoading
                ? "Ligler yükleniyor…"
                : hasLeagues
                  ? "Lig seçilmedi (isteğe bağlı)"
                  : "Lig seçmeden devam edin"}
            </option>
            {leagues.map((league) => (
              <option key={league.id} value={league.id}>
                {leagueLabel(league)}
              </option>
            ))}
          </select>

          {leaguesLoading && (
            <p
              id={leagueStatusId}
              role="status"
              className="mt-1 text-xs text-slate-400"
            >
              Desteklenen ligler yükleniyor…
            </p>
          )}
          {!leaguesLoading && leaguesError && (
            <p
              id={leagueStatusId}
              role="alert"
              className="mt-1 text-xs text-amber-400"
            >
              {leaguesError}
            </p>
          )}
          {!leaguesLoading && !leaguesError && !hasLeagues && (
            <p
              id={leagueStatusId}
              role="status"
              className="mt-1 text-xs text-slate-400"
            >
              Listelenecek lig bulunamadı. Lig seçmeden manuel analize devam
              edebilirsiniz.
            </p>
          )}
        </div>

        <input
          type="text"
          aria-label="Ev sahibi takım"
          className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm"
          placeholder="Ev Sahibi Takım"
          value={formData.home_team}
          onChange={(event) =>
            onChange({ ...formData, home_team: event.target.value })
          }
        />
        <input
          type="text"
          aria-label="Deplasman takımı"
          className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm"
          placeholder="Deplasman Takımı"
          value={formData.away_team}
          onChange={(event) =>
            onChange({ ...formData, away_team: event.target.value })
          }
        />
        <div className="grid gap-3 xl:grid-cols-2">
          <TeamStatsEditor
            accentClass="text-amber-400"
            label="Ev Sahibi"
            side="home"
            stats={formData.home_stats}
            onChange={(key, value) => updateTeamStat("home", key, value)}
          />
          <TeamStatsEditor
            accentClass="text-red-400"
            label="Deplasman"
            side="away"
            stats={formData.away_stats}
            onChange={(key, value) => updateTeamStat("away", key, value)}
          />
        </div>

        <details className="rounded-lg border border-slate-800 bg-slate-950/40">
          <summary className="cursor-pointer px-3 py-2 text-xs font-black text-slate-200">
            Piyasa Oranları ve Maç Bağlamı
          </summary>
          <div className="space-y-4 border-t border-slate-800 p-3">
            <div className="grid gap-2 sm:grid-cols-3">
              {MARKET_OUTCOMES.map((outcome) => (
                <div key={outcome.key}>
                  <label
                    htmlFor={`market-${outcome.key}`}
                    className="mb-1 block text-[10px] font-bold text-slate-400"
                  >
                    {outcome.label}
                  </label>
                  <input
                    id={`market-${outcome.key}`}
                    aria-label={
                      outcome.key === "HOME_WIN" ? "Bahis oranı" : undefined
                    }
                    type="number"
                    min="1.01"
                    max="1000"
                    step="0.01"
                    value={
                      formData.market_1x2?.raw_odds?.[outcome.key] ??
                      (outcome.key === "HOME_WIN" ? formData.odd : "")
                    }
                    onChange={(event) =>
                      updateMarketOdd(outcome.key, event.target.value)
                    }
                    className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2 text-sm"
                  />
                </div>
              ))}
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {[
                ["fixture_id", "Fikstür ID"],
                ["home_team_id", "Ev sahibi takım ID"],
                ["away_team_id", "Deplasman takım ID"],
                ["season", "Sezon"],
              ].map(([key, label]) => (
                <div key={key}>
                  <label
                    htmlFor={`context-${key}`}
                    className="mb-1 block text-[10px] font-bold text-slate-400"
                  >
                    {label}
                  </label>
                  <input
                    id={`context-${key}`}
                    type="number"
                    min={key === "season" ? 2000 : 1}
                    max={key === "season" ? 2100 : undefined}
                    value={formData[key] ?? ""}
                    onChange={(event) =>
                      onChange({
                        ...formData,
                        [key]:
                          event.target.value === ""
                            ? null
                            : Number(event.target.value),
                      })
                    }
                    className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2 text-sm"
                  />
                </div>
              ))}
              <div>
                <label
                  htmlFor="context-kickoff"
                  className="mb-1 block text-[10px] font-bold text-slate-400"
                >
                  Maç başlangıcı (ISO 8601)
                </label>
                <input
                  id="context-kickoff"
                  type="text"
                  value={formData.kickoff ?? ""}
                  onChange={(event) =>
                    onChange({
                      ...formData,
                      kickoff: event.target.value || null,
                    })
                  }
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2 text-sm"
                />
              </div>
              <div>
                <label
                  htmlFor="context-travel"
                  className="mb-1 block text-[10px] font-bold text-slate-400"
                >
                  Deplasman seyahat mesafesi (km)
                </label>
                <input
                  id="context-travel"
                  type="number"
                  min="0"
                  max="20000"
                  step="1"
                  value={formData.away_travel_distance_km ?? ""}
                  onChange={(event) =>
                    onChange({
                      ...formData,
                      away_travel_distance_km:
                        event.target.value === ""
                          ? null
                          : Number(event.target.value),
                    })
                  }
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2 text-sm"
                />
              </div>
            </div>
          </div>
        </details>

        <AnalysisInputsPanel
          error={featureError}
          featureOverrides={formData.feature_overrides}
          loading={featureLoading}
          onOverride={onFeatureOverride}
          onRefresh={onFeatureRefresh}
          onReset={onFeatureReset}
          preview={featurePreview}
          stale={featurePreviewStale}
        />

        <button
          type="submit"
          disabled={loading || fixtureLoading || featureLoading}
          className="w-full rounded-lg bg-emerald-500 p-3 text-sm font-bold text-slate-950 transition hover:bg-emerald-600 disabled:opacity-50"
        >
          {fixtureLoading
            ? "Maç verileri yükleniyor…"
            : featureLoading
              ? "Model girdileri hesaplanıyor…"
            : loading
              ? "Analiz ediliyor…"
              : "Tahmin Oluştur"}
        </button>
      </form>
    </div>
  );
}

export default AnalysisForm;
