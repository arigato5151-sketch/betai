const metric = (value, suffix = "") =>
  value === null || value === undefined ? "-" : `${value}${suffix}`;

function LeaguePerformanceCard({ data, error, loading, onRefresh }) {
  return (
    <section
      className="mx-auto mb-8 max-w-7xl rounded-lg border border-slate-800 bg-slate-900 p-5 shadow-xl"
      aria-labelledby="league-performance-title"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2
            id="league-performance-title"
            className="text-sm font-bold uppercase tracking-wider text-slate-300"
          >
            Lig Bazlı Tahmin Performansı
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Sonuçlanmış tahminlerde doğruluk, kalibrasyon ve oran değeri
          </p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="rounded border border-slate-700 px-3 py-1 text-xs hover:border-emerald-500 disabled:opacity-50"
        >
          {loading ? "Yenileniyor..." : "Yenile"}
        </button>
      </div>

      {error && (
        <p className="mt-3 rounded border border-red-900 bg-red-950/40 p-3 text-sm text-red-400">
          {error}
        </p>
      )}

      {data && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-xs">
            <thead className="border-b border-slate-700 text-slate-500">
              <tr>
                <th className="px-3 py-2">Lig</th>
                <th className="px-3 py-2">Örnek</th>
                <th className="px-3 py-2">Doğruluk</th>
                <th className="px-3 py-2">Brier</th>
                <th className="px-3 py-2">ROI</th>
                <th className="px-3 py-2">CLV</th>
                <th className="px-3 py-2">Güven</th>
              </tr>
            </thead>
            <tbody>
              {(data.leagues ?? []).map((league) => {
                const sufficient = league.sample_status
                  ? league.sample_status === "reliable"
                  : league.total_predictions >= 30;
                const accuracyInterval = league.win_rate_confidence_interval_95;
                return (
                  <tr
                    key={league.league_id}
                    className="border-b border-slate-800 text-slate-300"
                  >
                    <td className="px-3 py-3 font-semibold text-slate-100">
                      {league.league_name}
                    </td>
                    <td className="px-3 py-3">{league.total_predictions}</td>
                    <td className="px-3 py-3">
                      {metric(league.win_rate_pct, "%")}
                      {accuracyInterval && (
                        <span className="block text-[10px] text-slate-500">
                          %95 GA: {accuracyInterval.lower_pct}–{accuracyInterval.upper_pct}
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-3">
                      {metric(league.brier_score)}
                    </td>
                    <td className="px-3 py-3">
                      {metric(league.total_roi_pct, "%")}
                    </td>
                    <td className="px-3 py-3">
                      {metric(league.avg_clv_pct, "%")}
                    </td>
                    <td className="px-3 py-3">
                      <span
                        className={`rounded border px-2 py-1 ${
                          sufficient
                            ? "border-emerald-800 text-emerald-300"
                            : "border-amber-800 text-amber-300"
                        }`}
                      >
                        {sufficient ? "Yeterli" : "Düşük örnek"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {(data.leagues?.length ?? 0) === 0 && (
            <p className="py-6 text-center text-sm text-slate-500">
              Henüz lig bazında sonuçlanmış tahmin yok.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export default LeaguePerformanceCard;
