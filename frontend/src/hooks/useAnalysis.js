import { useState } from "react";

import { validateAnalysisResponse } from "../apiSchema.js";
import { useRequestGate } from "./useRequestGate.js";

export function useAnalysis({
  actions,
  formData,
  onHistoryChanged,
  onSelectMatch,
  request,
}) {
  const { stale, begin, settle } = useRequestGate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!actions.analyze) return;
    const mine = begin();
    setLoading(true);
    setError("");
    try {
      const fixtureId = formData.fixture_id;
      const isFixtureAnalysis = Number.isInteger(fixtureId) && fixtureId > 0;
      const endpoint = isFixtureAnalysis
        ? `/analyze/fixture/${fixtureId}`
        : "/analyze";
      const response = await request(endpoint, {
        method: "POST",
        ...(isFixtureAnalysis
          ? {}
          : {
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(formData),
            }),
      });
      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || "Analiz isteği başarısız oldu.");
      }
      const data = validateAnalysisResponse(await response.json());
      settle(mine, () => {
        onHistoryChanged?.();
        onSelectMatch?.({
          match: data.match,
          odd: formData.odd,
          home_team: formData.home_team,
          away_team: formData.away_team,
          league_id: formData.league_id,
          opening_odds_1x2: formData.opening_odds_1x2,
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
      });
    } catch (submitError) {
      settle(mine, () => {
        setError(submitError.message || "Analiz isteği başarısız oldu.");
      });
    } finally {
      setLoading(false);
    }
  };

  return {
    analysisError: error,
    analysisLoading: loading,
    analysisStale: stale,
    handleSubmit,
  };
}
