import { useEffect, useRef, useState } from "react";

import HistoryTable from "../components/HistoryTable.jsx";
import { buildHistoryQuery } from "../history.js";

export function historyItemToSelectedMatch(rawDbItem) {
  const probHome = rawDbItem.prob_home ?? rawDbItem.probability ?? 33;
  const probAway = rawDbItem.prob_away ?? 33;
  const probDraw = rawDbItem.prob_draw ?? 34;
  const mlInsufficient =
    rawDbItem.ml_cluster === null || rawDbItem.ml_cluster === undefined;
  const storedAssessment = rawDbItem.data_quality?.ml_assessment;
  const storedAnalysisOutputs = rawDbItem.data_quality?.analysis_outputs ?? {};

  return {
    match: `${rawDbItem.home_team} – ${rawDbItem.away_team}`,
    odd: rawDbItem.odd,
    home_stats: {
      form: rawDbItem.home_form,
      attack: rawDbItem.home_attack ?? 80,
      defense: rawDbItem.home_defense ?? 80,
      xg: rawDbItem.home_xg,
    },
    away_stats: {
      form: rawDbItem.away_form,
      attack: rawDbItem.away_attack ?? 80,
      defense: rawDbItem.away_defense ?? 80,
      xg: rawDbItem.away_xg,
    },
    analysis: {
      prediction: rawDbItem.prediction,
      probability: rawDbItem.probability,
      all_probabilities: {
        HOME_WIN: probHome,
        AWAY_WIN: probAway,
        DRAW: probDraw,
      },
      ...storedAnalysisOutputs,
    },
    value_assessment: {
      value_bet: rawDbItem.is_value_bet === 1,
      edge: rawDbItem.edge,
    },
    ml_safety_trigger:
      storedAssessment?.trigger ??
      (mlInsufficient
        ? "INSUFFICIENT_DATA"
        : rawDbItem.ml_cluster === 1
          ? "HIGH_CONFIDENCE"
          : "RISKY_UNDERDOG"),
    ml_safety_details: storedAssessment ?? null,
    ml_confidence: rawDbItem.ml_confidence,
    ml_ready: !mlInsufficient,
    record_id: rawDbItem.id,
    actual_result: rawDbItem.actual_result,
    result_verification_status: rawDbItem.result_verification_status,
    result_source: rawDbItem.result_source,
    data_quality: rawDbItem.data_quality,
    provenance: {
      model_name: rawDbItem.model_name,
      model_artifact_version: rawDbItem.model_artifact_version,
      feature_schema_version: rawDbItem.feature_schema_version,
      ensemble_version: rawDbItem.ensemble_version,
      analysis_lead_minutes: rawDbItem.analysis_lead_minutes,
    },
  };
}

function HistoryContainer({
  canRead,
  hasSelectedMatch,
  onSelectMatch,
  refreshToken,
  request,
}) {
  const [history, setHistory] = useState([]);
  const [page, setPage] = useState(1);
  const [meta, setMeta] = useState({ total: 0, pages: 1 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(0);
  const [filters, setFilters] = useState({
    query: "",
    result: "all",
    value: "all",
    sort: "newest",
  });

  const fetchHistory = (requestedPage = page) => {
    const currentRequestId = ++requestId.current;
    setLoading(true);
    setError("");
    request(`/history${buildHistoryQuery(filters, requestedPage)}`)
      .then((response) => {
        if (!response.ok) throw new Error("Tahmin geçmişi alınamadı.");
        return response.json();
      })
      .then((data) => {
        if (currentRequestId !== requestId.current) return;
        const items = Array.isArray(data)
          ? data
          : Array.isArray(data.items)
            ? data.items
            : [];
        setHistory(items);
        setMeta({
          total: data.total ?? items.length,
          pages: data.pages ?? 1,
        });
        if (items.length > 0 && !hasSelectedMatch) {
          onSelectMatch(historyItemToSelectedMatch(items[0]));
        }
      })
      .catch((requestError) => {
        if (currentRequestId !== requestId.current) return;
        setHistory([]);
        setMeta({ total: 0, pages: 1 });
        setError(requestError.message || "Tahmin geçmişi alınamadı.");
      })
      .finally(() => {
        if (currentRequestId === requestId.current) setLoading(false);
      });
  };

  const updateFilter = (key, value) => {
    setFilters((current) => ({ ...current, [key]: value }));
    setPage(1);
  };

  useEffect(() => {
    if (!canRead) return undefined;
    const timeoutId = window.setTimeout(() => fetchHistory(), 250);
    return () => window.clearTimeout(timeoutId);
  }, [
    canRead,
    page,
    filters.query,
    filters.result,
    filters.value,
    filters.sort,
    refreshToken,
  ]);

  return (
    <HistoryTable
      filters={filters}
      history={history}
      historyError={error}
      historyLoading={loading}
      meta={meta}
      onFilterChange={updateFilter}
      onPageChange={setPage}
      onSelectMatch={(item) =>
        onSelectMatch(historyItemToSelectedMatch(item))
      }
      page={page}
    />
  );
}

export default HistoryContainer;
