import { leagueLabel } from "../localization.js";

function AnalysisForm({
  formData,
  fixtureLoading = false,
  leagues = [],
  leaguesError = "",
  leaguesLoading = false,
  loading,
  onChange,
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

  return (
    <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
      <h2 className="mb-4 text-lg font-bold">Maç Analizi</h2>
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
        <button
          type="submit"
          disabled={loading || fixtureLoading}
          className="w-full rounded-lg bg-emerald-500 p-3 text-sm font-bold text-slate-950 transition hover:bg-emerald-600 disabled:opacity-50"
        >
          {fixtureLoading
            ? "Maç verileri yükleniyor…"
            : loading
              ? "Analiz ediliyor…"
              : "Tahmin Oluştur"}
        </button>
      </form>
    </div>
  );
}

export default AnalysisForm;
