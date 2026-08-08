import { useEffect, useState } from "react";

import { tierLabel, tierOutcomeLabels } from "../localization.js";

const asFiniteNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const percentage = (value) => {
  const parsed = asFiniteNumber(value);
  return parsed === null ? null : Math.max(0, Math.min(100, parsed * 100));
};

const collect = (features, name, value) => {
  const numeric = asFiniteNumber(value);
  if (numeric !== null) features[name] = numeric;
  return features;
};

export function buildTieredFeatures(featureSnapshot, leagueId, homeTeam, awayTeam) {
  const snapshot = featureSnapshot && typeof featureSnapshot === "object" ? featureSnapshot : {};
  const features = {};
  if (asFiniteNumber(leagueId) !== null) features.league_id = leagueId;
  if (homeTeam) features.home_team = homeTeam;
  if (awayTeam) features.away_team = awayTeam;
  collect(features, "home_avg_goals", snapshot.home_gf_last5);
  collect(features, "away_avg_goals", snapshot.away_gf_last5);
  collect(
    features,
    "home_form_last5",
    snapshot.home_form_last5 ?? snapshot.home_form ?? snapshot.home_form_ema,
  );
  collect(
    features,
    "away_form_last5",
    snapshot.away_form_last5 ?? snapshot.away_form ?? snapshot.away_form_ema,
  );
  collect(features, "home_elo", snapshot.home_elo);
  collect(features, "away_elo", snapshot.away_elo);
  return features;
}

function TieredPredictionPanel({
  request,
  leagueId,
  homeTeam,
  awayTeam,
  featureSnapshot,
  enabled,
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!enabled) {
      setResult(null);
      setError("");
      return undefined;
    }
    if (!request || asFiniteNumber(leagueId) === null) return undefined;

    let isActive = true;
    setLoading(true);
    setError("");
    setResult(null);

    const run = async () => {
      try {
        const response = await request("/predict/tiered", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            league_id: leagueId,
            features: buildTieredFeatures(
              featureSnapshot,
              leagueId,
              homeTeam,
              awayTeam,
            ),
          }),
        });
        if (!response.ok) {
          const errorBody = await response.json().catch(() => ({}));
          throw new Error(
            errorBody.detail ||
              (response.status === 503
                ? "İmzalı tier modeli kullanıma hazır değil."
                : "Tier tahmini alınamadı."),
          );
        }
        const data = await response.json();
        if (isActive) {
          setResult(data);
        }
      } catch (error) {
        if (isActive) {
          setError(error.message || "Tier tahmini alınamadı.");
        }
      } finally {
        if (isActive) {
          setLoading(false);
        }
      }
    };

    run();
    return () => {
      isActive = false;
    };
  }, [enabled, request, leagueId, homeTeam, awayTeam, featureSnapshot]);

  const outcomes = tierOutcomeLabels();
  const scoreKeys = ["0", "1", "2"];
  const confidenceScores = result?.confidence_scores ?? {};

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5 shadow-xl">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-bold text-slate-200">Katmanlı Model Tahmini</h2>
        {result && (
          <span className="rounded-full border border-cyan-800 bg-cyan-950/50 px-3 py-1 text-xs font-semibold text-cyan-300">
            {tierLabel(result.used_tier)}
          </span>
        )}
      </div>

      {loading && <p className="text-sm text-slate-400">Katman tahmini hesaplanıyor…</p>}

      {!loading && error && (
        <p role="alert" className="text-sm text-amber-300">
          {error}
        </p>
      )}

      {!loading && result && (
        <>
          <ul className="space-y-3">
            {outcomes.map((label, index) => {
              const raw = confidenceScores[scoreKeys[index]];
              const percent = percentage(raw);
              return (
                <li key={label} className="space-y-1">
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>{label}</span>
                    <span>{percent === null ? "–" : `${percent.toFixed(1)}%`}</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-600 to-cyan-400"
                      style={{
                        width: `${percent === null ? 0 : percent}%`,
                        transition: "width 300ms ease",
                      }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>

          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-800 pt-3 text-xs text-slate-400">
            <span>
              En Yüksek Güven:{" "}
              <strong className="text-slate-200">
                {confidenceScores[scoreKeys[0]] === undefined
                  ? "–"
                  : `${percentage(result.confidence)?.toFixed(1)}%`}
              </strong>
            </span>
            <span>
              Model Sürümü:{" "}
              <strong className="text-slate-200">
                {result.artifact_version ?? "Doğrulanmamış / Yüklenmemiş"}
              </strong>
            </span>
          </div>
        </>
      )}

      {!loading && !error && !result && (
        <p className="text-sm text-slate-500">
          Tahmin göstermek için önce bir maç analizi yapın.
        </p>
      )}
    </div>
  );
}

export default TieredPredictionPanel;