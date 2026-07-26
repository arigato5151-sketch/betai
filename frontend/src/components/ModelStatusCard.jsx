function Value({ label, children }) {
  return (
    <div className="rounded bg-slate-950 p-3">
      <span className="block text-xs text-slate-500">{label}</span>
      <strong className="text-sm text-slate-200">{children ?? "-"}</strong>
    </div>
  );
}

function ModelStatusCard({ status, error, loading, onRefresh }) {
  const metrics = status?.metrics ?? {};
  const trainingData = status?.training_data ?? {};

  return (
    <section
      className="mx-auto mb-8 max-w-7xl rounded-lg border border-slate-800 bg-slate-900 p-5 shadow-xl"
      aria-labelledby="model-status-title"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="model-status-title" className="text-sm font-bold uppercase tracking-wider text-slate-300">
            ML Model Durumu
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Walk-forward test, kalibrasyon ve champion karşılaştırması
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`rounded border px-3 py-1 text-xs font-bold ${
            status?.ready
              ? "border-emerald-700 bg-emerald-950/40 text-emerald-300"
              : "border-amber-700 bg-amber-950/40 text-amber-300"
          }`}>
            {status?.ready ? "AKTİF" : "EĞİTİM BEKLİYOR"}
          </span>
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="rounded border border-slate-700 px-3 py-1 text-xs hover:border-emerald-500 disabled:opacity-50"
          >
            {loading ? "Yenileniyor..." : "Yenile"}
          </button>
        </div>
      </div>
      {error && <p className="mt-3 rounded border border-red-900 bg-red-950/40 p-3 text-sm text-red-400">{error}</p>}
      {status && (
        <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4 lg:grid-cols-8">
          <Value label="Model">{status.model_name}</Value>
          <Value label="Eğitim Örneği">{metrics.samples ?? 0}</Value>
          <Value label="Tarihsel Fixture">{trainingData.historical_fixtures ?? 0}</Value>
          <Value label="Etiketli Tahmin">{trainingData.labeled_predictions ?? 0}</Value>
          <Value label="Brier">{metrics.brier_score?.toFixed?.(4)}</Value>
          <Value label="Baseline Brier">{metrics.baseline_brier_score?.toFixed?.(4)}</Value>
          <Value label="Kalibrasyon">{metrics.calibration_error?.toFixed?.(4)}</Value>
          <Value label="Accuracy">{metrics.accuracy !== undefined ? `%${(metrics.accuracy * 100).toFixed(1)}` : "-"}</Value>
          <Value label="Lig">{metrics.league_count ?? 0}</Value>
        </div>
      )}
    </section>
  );
}

export default ModelStatusCard;
