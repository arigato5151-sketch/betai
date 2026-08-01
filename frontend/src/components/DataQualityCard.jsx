import {
  dataQualityStatusLabel,
  syncStatusLabel,
} from "../localization.js";

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

function ProviderStatus({ name, provider }) {
  if (!provider) return null;
  const status = provider.status ?? "unknown";
  const tone = status === "ready" || status === "configured"
    ? "border-emerald-800 text-emerald-300"
    : status === "disabled" || status === "unknown"
      ? "border-slate-700 text-slate-400"
      : "border-amber-800 text-amber-300";

  return (
    <div className={`rounded border p-3 ${tone}`}>
      <span className="block text-xs text-slate-500">{name}</span>
      <strong className="text-sm">{status}</strong>
      {provider.daily_remaining !== null && provider.daily_remaining !== undefined && (
        <span className="mt-1 block text-xs text-slate-400">
          Günlük kota: {provider.daily_remaining}/{provider.daily_limit ?? "?"}
        </span>
      )}
      {provider.circuit_open_until && (
        <span className="mt-1 block text-xs text-amber-400">
          Devre tekrar deneme: {new Date(provider.circuit_open_until).toLocaleTimeString("tr-TR")}
        </span>
      )}
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
            Tarihsel veri, etiket, kapanış oranı ve veri kökeni kapsamı
          </p>
        </div>
        <div className="flex items-center gap-2">
          {data && (
            <span className={`rounded border px-3 py-1 text-xs font-bold ${statusColors[data.status] ?? statusColors.warning}`}>
              {dataQualityStatusLabel(data.status).toLocaleUpperCase("tr-TR")} ·{" "}
              {data.score}/100
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
        <>
          <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4 lg:grid-cols-8">
            <Metric label="Maç" value={data.historical?.fixtures} />
            <Metric label="Lig / Sezon" value={`${data.historical?.leagues ?? 0} / ${data.historical?.seasons ?? 0}`} />
            <Metric label="Güncellik" value={data.historical?.freshness_hours} suffix=" saat" />
            <Metric label="Kadro Kapsaması" value={data.historical?.lineup_coverage_pct} suffix="%" />
            <Metric label="Etiket Kapsaması" value={data.predictions?.labeled_coverage_pct} suffix="%" />
            <Metric label="Kapanış Oranı" value={data.predictions?.closing_odds_coverage_pct} suffix="%" />
            <Metric label="Veri Kökeni" value={data.predictions?.provenance_coverage_pct} suffix="%" />
            <Metric label="Son Senkron" value={syncStatusLabel(data.latest_sync?.status)} />
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            <ProviderStatus name="API-Football" provider={data.providers?.api_football} />
            <ProviderStatus name="Sportmonks" provider={data.providers?.sportmonks} />
          </div>
          <div className="mt-3 rounded border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400">
            <strong className="text-slate-200">
              {data.historical?.current_season} sezon kapsamı: {data.historical?.current_season_covered_leagues ?? 0}/
              {data.historical?.current_season_coverage?.length ?? 0} lig
            </strong>
            {(data.historical?.current_season_missing_league_ids?.length ?? 0) > 0 && (
              <span className="ml-2">
                Eksik lig kimlikleri: {data.historical.current_season_missing_league_ids.join(", ")}
              </span>
            )}
          </div>
        </>
      )}
    </section>
  );
}

export default DataQualityCard;
