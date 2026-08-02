const DIRECT_FEATURES = new Set([
  "home_form",
  "home_attack",
  "home_defense",
  "home_xg",
  "away_form",
  "away_attack",
  "away_defense",
  "away_xg",
]);

const IDENTITY_FEATURES = new Set([
  "league_id",
  "home_team_id",
  "away_team_id",
]);

const STATUS_STYLES = {
  available: "border-emerald-900/70 bg-emerald-950/40 text-emerald-300",
  partial: "border-amber-900/70 bg-amber-950/40 text-amber-300",
  missing: "border-red-900/70 bg-red-950/40 text-red-300",
  manual: "border-sky-900/70 bg-sky-950/40 text-sky-300",
};

const STATUS_LABELS = {
  available: "Hesaplandı",
  partial: "Kısmi veri",
  missing: "Eksik / nötr",
  manual: "Manuel",
};

const SOURCE_LABELS = {
  analysis_form: "Analiz formu",
  api_football_availability: "API-Football oyuncu durumu",
  api_football_lineups: "API-Football kadroları",
  api_football_odds: "API-Football oran geçmişi",
  clubelo: "ClubElo",
  fixture_metadata: "Maç bilgileri",
  head_to_head_history: "İkili rekabet geçmişi",
  historical_fixtures: "Yerel maç geçmişi",
  manual_override: "Manuel değişiklik",
  market_snapshot: "Oran anlık görüntüsü",
  neutral_default: "Nötr varsayılan",
  player_context: "Oyuncu geçmişi",
  schedule_context: "Takvim ve seyahat verisi",
  schedule_and_geonames_city: "Takvim + GeoNames şehir merkezi",
  schedule_and_curated_team_locations: "Takvim + doğrulanmış takım konumu",
  schedule_and_manual_override: "Takvim + manuel seyahat mesafesi",
  open_meteo_forecast: "Open-Meteo hava tahmini",
  open_meteo_historical_forecast: "Open-Meteo tarihsel tahmin",
  open_meteo_archive: "Open-Meteo tarihsel hava arşivi",
};

function sourceLabel(source) {
  return SOURCE_LABELS[source] ?? source ?? "Bilinmiyor";
}

function formatCapturedAt(value) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(date);
}

function groupFeatures(features) {
  const groups = new Map();
  features
    .filter((feature) => !DIRECT_FEATURES.has(feature.name))
    .forEach((feature) => {
      if (!groups.has(feature.group)) groups.set(feature.group, []);
      groups.get(feature.group).push(feature);
    });
  return [...groups.entries()];
}

function Metric({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
      <span className="block text-[10px] font-bold uppercase tracking-wider text-slate-500">
        {label}
      </span>
      <strong className="mt-1 block text-sm text-slate-100">{value}</strong>
    </div>
  );
}

function AnalysisInputsPanel({
  error,
  featureOverrides,
  loading,
  onOverride,
  onRefresh,
  onReset,
  preview,
  stale,
}) {
  const features = Array.isArray(preview?.features) ? preview.features : [];
  const groups = groupFeatures(features);
  const visibleFeatures = groups.flatMap(([, rows]) => rows);
  const missingCount = visibleFeatures.filter(
    (feature) => feature.availability === "missing",
  ).length;
  const partialCount = visibleFeatures.filter(
    (feature) => feature.availability === "partial",
  ).length;
  const overrideCount = Object.keys(featureOverrides ?? {}).length;
  const expectedGoals = preview?.derived?.expected_goals;
  const qualityScore = preview?.data_quality?.score;

  return (
    <section
      aria-labelledby="analysis-inputs-title"
      className="rounded-xl border border-slate-800 bg-slate-950/40 p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="analysis-inputs-title" className="font-black text-slate-100">
            Hesaplanan Tüm Model Girdileri
          </h3>
          <p className="mt-1 text-xs leading-5 text-slate-400">
            Eksik alanlar nötr varsayılanla hesaplanır. Buradaki manuel
            değişiklikler doğrudan ML feature vektörüne uygulanır.
          </p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="rounded-lg border border-slate-700 px-3 py-2 text-xs font-bold text-slate-200 hover:border-emerald-500 disabled:opacity-50"
        >
          {loading ? "Hesaplanıyor…" : "Değerleri Yeniden Hesapla"}
        </button>
      </div>

      {stale && (
        <p
          role="status"
          className="mt-3 rounded-lg border border-amber-900/70 bg-amber-950/40 p-3 text-xs text-amber-300"
        >
          Temel girdiler değişti. Otomatik değerleri güncellemek için yeniden
          hesaplayın.
        </p>
      )}
      {error && (
        <p
          role="alert"
          className="mt-3 rounded-lg border border-red-900/70 bg-red-950/40 p-3 text-xs text-red-300"
        >
          {error}
        </p>
      )}

      {preview && (
        <>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric
              label="Veri Kalitesi"
              value={
                Number.isFinite(Number(qualityScore))
                  ? `${qualityScore}/100`
                  : "Bilinmiyor"
              }
            />
            <Metric label="Eksik Alan" value={missingCount} />
            <Metric label="Kısmi Alan" value={partialCount} />
            <Metric label="Manuel Değişiklik" value={overrideCount} />
          </div>

          {expectedGoals && (
            <div className="mt-2 grid grid-cols-3 gap-2">
              <Metric label="Ev Beklenen Gol" value={expectedGoals.home} />
              <Metric label="Dep. Beklenen Gol" value={expectedGoals.away} />
              <Metric label="Toplam xG" value={expectedGoals.total} />
            </div>
          )}

          <div className="mt-4 space-y-3">
            {groups.map(([group, rows]) => (
              <details
                key={group}
                open={rows.some(
                  (feature) =>
                    feature.availability === "missing" ||
                    feature.availability === "partial",
                )}
                className="rounded-lg border border-slate-800 bg-slate-900/70"
              >
                <summary className="cursor-pointer px-3 py-2 text-xs font-black text-slate-200">
                  {group} · {rows.length} değer
                </summary>
                <div className="grid gap-3 border-t border-slate-800 p-3 md:grid-cols-2">
                  {rows.map((feature) => {
                    const manuallyOverridden = Object.hasOwn(
                      featureOverrides ?? {},
                      feature.name,
                    );
                    const readOnlyIdentity =
                      IDENTITY_FEATURES.has(feature.name) ||
                      feature.name.startsWith("league_");
                    const status = manuallyOverridden
                      ? "manual"
                      : feature.availability;
                    const capturedAt = formatCapturedAt(feature.captured_at);
                    const confidence = Number(feature.confidence);
                    return (
                      <div
                        key={feature.name}
                        className="rounded-lg border border-slate-800 bg-slate-950/70 p-3"
                      >
                        <div className="mb-2 flex items-start justify-between gap-2">
                          <label
                            htmlFor={`feature-${feature.name}`}
                            className="text-xs font-bold text-slate-200"
                          >
                            {feature.label}
                          </label>
                          <span
                            className={`shrink-0 rounded border px-1.5 py-0.5 text-[9px] font-black uppercase ${STATUS_STYLES[status]}`}
                          >
                            {STATUS_LABELS[status]}
                          </span>
                        </div>
                        <div className="flex gap-2">
                          <input
                            id={`feature-${feature.name}`}
                            type="number"
                            min={feature.minimum}
                            max={feature.maximum}
                            step={feature.step}
                            readOnly={readOnlyIdentity}
                            value={
                              manuallyOverridden
                                ? featureOverrides[feature.name]
                                : feature.value
                            }
                            onChange={(event) =>
                              onOverride(feature.name, Number(event.target.value))
                            }
                            className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-900 p-2 text-sm read-only:cursor-not-allowed read-only:opacity-60"
                          />
                          {manuallyOverridden && (
                            <button
                              type="button"
                              onClick={() => onReset(feature.name)}
                              className="rounded-lg border border-slate-700 px-2 text-[10px] font-bold text-slate-300 hover:border-sky-500"
                            >
                              Sıfırla
                            </button>
                          )}
                        </div>
                        <p className="mt-1 text-[10px] text-slate-500">
                          Anahtar: {feature.name} · Varsayılan:{" "}
                          {feature.default_value}
                        </p>
                        <p className="mt-1 text-[10px] text-slate-400">
                          Kaynak:{" "}
                          <span className="font-bold text-slate-300">
                            {sourceLabel(
                              manuallyOverridden
                                ? "manual_override"
                                : feature.source,
                            )}
                          </span>
                          {Number.isFinite(confidence) &&
                            ` · Güven %${Math.round(confidence * 100)}`}
                          {capturedAt && ` · Güncellendi ${capturedAt}`}
                          {feature.is_fallback && !manuallyOverridden && (
                            <span className="ml-1 text-amber-300">
                              {" · fallback"}
                            </span>
                          )}
                        </p>
                        {feature.missing_reason && !manuallyOverridden && (
                          <p className="mt-1 text-[10px] text-red-300">
                            {feature.missing_reason}
                          </p>
                        )}
                        {readOnlyIdentity && (
                          <p className="mt-1 text-[10px] text-slate-500">
                            Bu değer takım/lig seçiminden otomatik üretilir.
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </details>
            ))}
          </div>
        </>
      )}

      {!preview && !loading && !error && (
        <p className="mt-3 text-xs text-slate-500">
          Hesaplanan girdileri görmek için değerleri yeniden hesaplayın.
        </p>
      )}
    </section>
  );
}

export default AnalysisInputsPanel;
