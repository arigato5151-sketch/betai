import StatusMessage from "./StatusMessage.jsx";
import { resultLabel } from "../localization.js";

const hasScore = (value) =>
  Number.isInteger(value) && value >= 0;

export function getPredictionResult(item) {
  if (!item.actual_result) return null;

  const isCorrect = item.prediction === item.actual_result;
  const hasFullScore =
    hasScore(item.actual_score_home) && hasScore(item.actual_score_away);

  return {
    isCorrect,
    resultText: hasFullScore
      ? `${item.actual_score_home} – ${item.actual_score_away}`
      : resultLabel(item.actual_result),
    verdictText: isCorrect ? "Tahmin doğru" : "Tahmin yanlış",
  };
}

export function getResultVerificationBadge(item) {
  if (!item.actual_result) return null;

  const badges = {
    verified: {
      text: "Provider sonucu doğrulandı",
      className: "border-sky-800 bg-sky-950/50 text-sky-300",
    },
    manual: {
      text: "Manuel sonuç · eğitim dışı",
      className: "border-amber-800 bg-amber-950/50 text-amber-300",
    },
    conflict: {
      text: "Sonuç çelişkili · karantinada",
      className: "border-red-800 bg-red-950/50 text-red-300",
    },
    rejected: {
      text: "Sonuç doğrulanamadı · karantinada",
      className: "border-red-800 bg-red-950/50 text-red-300",
    },
  };

  return badges[item.result_verification_status] ?? {
    text: "Sonuç kaynağı doğrulanmamış",
    className: "border-slate-700 bg-slate-900 text-slate-400",
  };
}

export function getPredictionStatusBadge(item) {
  if (
    item.analysis_origin === "scenario" ||
    item.eligibility_status !== "abstain"
  ) {
    return null;
  }

  const dataEligibility = item.data_quality?.prediction_eligibility;
  if (dataEligibility?.status !== "eligible") {
    return {
      text: "Sınırlı veri",
      className: "border-amber-800 bg-amber-950/50 text-amber-300",
    };
  }

  const decision = item.data_quality?.decision_recommendation;
  if (decision?.status === "conditional") {
    return {
      text: "Koşullu tahmin",
      className: "border-violet-800 bg-violet-950/50 text-violet-300",
    };
  }

  const decisionReasons = decision?.reasons;
  const reasons = Array.isArray(decisionReasons) ? decisionReasons : [];
  if (reasons.includes("market_edge_insufficient")) {
    return {
      text: "Piyasa avantajı yok",
      className: "border-sky-800 bg-sky-950/50 text-sky-300",
    };
  }
  if (reasons.includes("market_unavailable")) {
    return {
      text: "Piyasa oranı yok",
      className: "border-amber-800 bg-amber-950/50 text-amber-300",
    };
  }
  if (reasons.includes("probability_margin_too_low")) {
    return {
      text: "Olasılıklar birbirine çok yakın",
      className: "border-slate-700 bg-slate-900 text-slate-400",
    };
  }
  if (reasons.includes("top_probability_too_low")) {
    return {
      text: "En yüksek olasılık düşük",
      className: "border-slate-700 bg-slate-900 text-slate-400",
    };
  }
  if (reasons.includes("prediction_sources_diverge")) {
    return {
      text: "Tahmin modelleri ayrışıyor",
      className: "border-slate-700 bg-slate-900 text-slate-400",
    };
  }
  return {
    text: "Tahmin eşiği geçilmedi",
    className: "border-slate-700 bg-slate-900 text-slate-400",
  };
}

function HistoryTable({
  filters,
  history,
  historyError,
  historyLoading,
  meta,
  onFilterChange,
  onPageChange,
  onSelectMatch,
  page,
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">Son Tahmin İstekleri</h3>
        <span className="text-xs text-slate-500">{history.length}/{meta.total} kayıt</span>
      </div>
      <div className="mb-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <label className="sm:col-span-2 lg:col-span-1">
          <span className="sr-only">Takım ara</span>
          <input
            type="search"
            placeholder="Takım ara…"
            value={filters.query}
            onChange={(event) => onFilterChange("query", event.target.value)}
            className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
          />
        </label>
        <label>
          <span className="sr-only">Sonuç filtresi</span>
          <select value={filters.result} onChange={(event) => onFilterChange("result", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
            <option value="all">Tüm sonuçlar</option>
            <option value="pending">Sonuç bekleyenler</option>
            <option value="HOME_WIN">Ev sahibi kazandı</option>
            <option value="DRAW">Beraberlik</option>
            <option value="AWAY_WIN">Deplasman kazandı</option>
          </select>
        </label>
        <label>
          <span className="sr-only">Değerli bahis filtresi</span>
          <select value={filters.value} onChange={(event) => onFilterChange("value", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
            <option value="all">Tüm bahisler</option>
            <option value="value">Yalnızca değerli bahisler</option>
            <option value="non_value">Değerli olmayan bahisler</option>
          </select>
        </label>
        <label>
          <span className="sr-only">Sıralama</span>
          <select value={filters.sort} onChange={(event) => onFilterChange("sort", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
            <option value="newest">En yeni</option>
            <option value="oldest">En eski</option>
            <option value="edge">En yüksek avantaj</option>
            <option value="odd">En yüksek oran</option>
          </select>
        </label>
      </div>
      <div className="space-y-2">
        {history.map((item) => {
          const predictionResult = getPredictionResult(item);
          const verificationBadge = getResultVerificationBadge(item);
          const predictionStatusBadge = getPredictionStatusBadge(item);
          const actionableValueBet =
            item.is_value_bet === 1 &&
            item.data_quality?.financial_recommendation?.status === "eligible";

          return (
            <button key={item.id} type="button" onClick={() => onSelectMatch(item)} className="flex w-full cursor-pointer flex-col gap-2 rounded-lg border border-slate-800 bg-slate-950 p-3 text-left transition hover:border-slate-600 sm:flex-row sm:items-center sm:justify-between">
              <span className="min-w-0 flex-1 text-sm font-medium text-slate-200">
                  <span
                    className="block truncate"
                    title={`${item.home_team} – ${item.away_team}`}
                  >
                    {item.home_team} – {item.away_team}
                  </span>
                </span>
              <div className="flex flex-wrap items-center gap-2">
                {item.analysis_origin === "scenario" && (
                  <span className="rounded border border-violet-800 bg-violet-950/50 px-2 py-1 text-xs font-semibold text-violet-300">
                    Senaryo · eğitim dışı
                  </span>
                )}
                {predictionStatusBadge && (
                  <span className={`rounded border px-2 py-1 text-xs font-semibold ${predictionStatusBadge.className}`}>
                    {predictionStatusBadge.text}
                  </span>
                )}
                {item.eligibility_status === "eligible" &&
                  item.training_eligible === false &&
                  item.analysis_origin !== "scenario" && (
                    <span className="rounded border border-slate-700 px-2 py-1 text-xs font-semibold text-slate-400">
                      Eğitim dışı
                    </span>
                  )}
                {item.eligibility_status === "unverified" && (
                  <span className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-400">
                    Eski kayıt · doğrulanmamış
                  </span>
                )}
                {verificationBadge && (
                  <span className={`rounded border px-2 py-1 text-xs font-semibold ${verificationBadge.className}`}>
                    {verificationBadge.text}
                  </span>
                )}
                {predictionResult && (
                  <span
                    className={`rounded border px-2 py-1 text-xs font-semibold ${
                      predictionResult.isCorrect
                        ? "border-emerald-800 bg-emerald-950/60 text-emerald-400"
                        : "border-red-800 bg-red-950/60 text-red-400"
                    }`}
                  >
                    Maç sonucu: {predictionResult.resultText} · {predictionResult.verdictText}
                  </span>
                )}
                <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-400">Oran: {item.odd}</span>
                <span className={`h-2 w-2 rounded-full ${actionableValueBet ? "animate-pulse bg-amber-400" : "bg-slate-700"}`}></span>
              </div>
            </button>
          );
        })}
        {historyLoading && (
          <StatusMessage tone="stale">Geçmiş yükleniyor…</StatusMessage>
        )}
        {historyError && (
          <StatusMessage tone="error" id="history-error">
            {historyError}
          </StatusMessage>
        )}
        {!historyLoading && !historyError && history.length === 0 && (
          <p className="rounded-lg border border-dashed border-slate-800 p-4 text-sm text-slate-500">
            Filtrelerle eşleşen kayıt bulunamadı.
          </p>
        )}
      </div>
      {meta.pages > 1 && (
        <nav className="mt-4 flex items-center justify-between" aria-label="Geçmiş sayfalama">
          <button type="button" disabled={page <= 1 || historyLoading} onClick={() => onPageChange(Math.max(1, page - 1))} className="rounded border border-slate-700 px-3 py-1.5 text-sm disabled:opacity-40">Önceki</button>
          <span className="text-xs text-slate-500">Sayfa {page} / {meta.pages}</span>
          <button type="button" disabled={page >= meta.pages || historyLoading} onClick={() => onPageChange(Math.min(meta.pages, page + 1))} className="rounded border border-slate-700 px-3 py-1.5 text-sm disabled:opacity-40">Sonraki</button>
        </nav>
      )}
    </div>
  );
}

export default HistoryTable;
