function Metric({ label, value, suffix = "" }) {
  return (
    <div className="rounded bg-slate-950 p-3">
      <span className="block text-xs text-slate-500">{label}</span>
      <strong className="text-sm text-slate-200">
        {value ?? "-"}
        {value !== null && value !== undefined ? suffix : ""}
      </strong>
    </div>
  );
}

function DataQualityCard({ data, error, loading, onRefresh }) {
  const statusColors = {
    healthy: "border-emerald-700 bg-emerald-950/40 text-emerald-300",
    warning: "border-amber-700 bg-amber-950/40 text-amber-300",
    critical: "border-red-800 bg-red-950/40 text-red-300",
  };

  return (
    <section
      className="mx-auto mb-8 max-w-7xl rounded-lg border border-slate-800 bg-slate-900 p-5 shadow-xl"
      aria-labelledby="data-quality-title"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="data-quality-title" className="text-sm font-bold uppercase tracking-wider text-slate-300">
            Veri Kalitesi ve Operasyon Durumu
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Tarihsel veri, etiket, closing odds ve provenance kapsaması
          </p>
        </div>
        <div className="flex items-center gap-2">
          {data && (
            <span className={`rounded border px-3 py-1 text-xs font-bold ${statusColors[data.status] ?? statusColors.warning}`}>
              {data.status?.toUpperCase()} · {data.score}/100
            </span>
          )}
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
      {data && (
        <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4 lg:grid-cols-8">
          <Metric label="Fixture" value={data.historical?.fixtures} />
          <Metric label="Lig / Sezon" value={`${data.historical?.leagues ?? 0} / ${data.historical?.seasons ?? 0}`} />
          <Metric label="Güncellik" value={data.historical?.freshness_hours} suffix=" saat" />
          <Metric label="Kadro Kapsaması" value={data.historical?.lineup_coverage_pct} suffix="%" />
          <Metric label="Etiket Kapsaması" value={data.predictions?.labeled_coverage_pct} suffix="%" />
          <Metric label="Closing Odds" value={data.predictions?.closing_odds_coverage_pct} suffix="%" />
          <Metric label="Provenance" value={data.predictions?.provenance_coverage_pct} suffix="%" />
          <Metric label="Son Senkron" value={data.latest_sync?.status ?? "Yok"} />
        </div>
      )}
    </section>
  );
}

export default DataQualityCard;
