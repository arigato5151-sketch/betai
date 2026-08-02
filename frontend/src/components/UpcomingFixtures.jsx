import { leagueLabel } from "../localization.js";

const ISTANBUL_TIME_ZONE = "Europe/Istanbul";
const DAY_FORMATTER = new Intl.DateTimeFormat("tr-TR", {
  timeZone: ISTANBUL_TIME_ZONE,
  weekday: "long",
  day: "numeric",
  month: "long",
});
const TIME_FORMATTER = new Intl.DateTimeFormat("tr-TR", {
  timeZone: ISTANBUL_TIME_ZONE,
  hour: "2-digit",
  minute: "2-digit",
});

function groupFixturesByDay(fixtures) {
  const groups = new Map();
  fixtures.forEach((fixture) => {
    const kickoff = new Date(fixture.kickoff);
    const label = DAY_FORMATTER.format(kickoff);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(fixture);
  });
  return [...groups.entries()];
}

function UpcomingFixtures({
  error,
  fixtures,
  loading,
  onRefresh,
  onSelectFixture,
  selectedFixtureId,
}) {
  const groupedFixtures = groupFixturesByDay(fixtures);

  return (
    <section
      aria-labelledby="upcoming-fixtures-title"
      className="mx-auto mb-8 max-w-7xl rounded-xl border border-slate-800 bg-slate-900 p-5 shadow-xl"
    >
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold uppercase tracking-widest text-emerald-400">
            Önümüzdeki 7 gün
          </p>
          <h2
            id="upcoming-fixtures-title"
            className="mt-1 text-xl font-black text-slate-100"
          >
            Haftalık Maç Fikstürü
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Maçlar Türkiye saatine göre kronolojik sıralanır. Analiz formuna
            aktarmak için bir maça dokunun.
          </p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="rounded-lg border border-slate-700 px-3 py-2 text-sm font-bold text-slate-200 transition hover:border-emerald-500 hover:text-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Yükleniyor…" : "Fikstürü Yenile"}
        </button>
      </div>

      {error && (
        <div
          role="alert"
          className="rounded-lg border border-red-900/70 bg-red-950/40 p-4 text-sm text-red-300"
        >
          {error}
        </div>
      )}

      {!error && loading && fixtures.length === 0 && (
        <div
          role="status"
          className="rounded-lg border border-slate-800 bg-slate-950/60 p-5 text-sm text-slate-400"
        >
          Bir haftalık fikstür yükleniyor…
        </div>
      )}

      {!error && !loading && fixtures.length === 0 && (
        <div
          role="status"
          className="rounded-lg border border-slate-800 bg-slate-950/60 p-5 text-sm text-slate-400"
        >
          Önümüzdeki 7 gün için desteklenen liglerde maç bulunamadı.
        </div>
      )}

      {!error && groupedFixtures.length > 0 && (
        <div className="space-y-5">
          {groupedFixtures.map(([dayLabel, dayFixtures]) => (
            <section key={dayLabel} aria-label={dayLabel}>
              <h3 className="mb-2 text-sm font-black capitalize text-amber-400">
                {dayLabel}
              </h3>
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                {dayFixtures.map((fixture) => (
                  <article
                    key={fixture.fixture_id}
                    className={`rounded-lg border bg-slate-950/70 transition ${
                      selectedFixtureId === fixture.fixture_id
                        ? "border-emerald-500 ring-1 ring-emerald-500/50"
                        : "border-slate-800 hover:border-slate-600"
                    }`}
                  >
                    <button
                      type="button"
                      aria-label={`${fixture.home_team} – ${fixture.away_team} maçını analiz formuna taşı`}
                      disabled={!onSelectFixture}
                      onClick={() => onSelectFixture?.(fixture)}
                      className="w-full rounded-lg p-4 text-left disabled:cursor-not-allowed"
                    >
                      <div className="mb-3 flex items-center justify-between gap-3 text-xs">
                        <span className="truncate font-bold text-emerald-400">
                          {leagueLabel({
                            id: fixture.league_id,
                            name: fixture.league,
                          })}
                        </span>
                        <time
                          dateTime={fixture.kickoff}
                          className="shrink-0 font-black text-slate-300"
                        >
                          {TIME_FORMATTER.format(new Date(fixture.kickoff))}
                        </time>
                      </div>
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <span className="min-w-0 flex-1 truncate text-right font-bold text-slate-100">
                          {fixture.home_team}
                        </span>
                        <span className="shrink-0 text-xs font-black text-slate-500">
                          –
                        </span>
                        <span className="min-w-0 flex-1 truncate font-bold text-slate-100">
                          {fixture.away_team}
                        </span>
                      </div>
                      {(fixture.is_demo ||
                        fixture.sources?.length > 1 ||
                        selectedFixtureId === fixture.fixture_id) && (
                        <div className="mt-3 flex items-center justify-center gap-2 text-[10px] font-bold uppercase tracking-widest">
                          {fixture.is_demo && (
                            <span className="text-amber-500">Demo veri</span>
                          )}
                          {fixture.sources?.length > 1 && (
                            <span className="text-sky-400">
                              {fixture.sources.length} kaynak doğruladı
                            </span>
                          )}
                          {selectedFixtureId === fixture.fixture_id && (
                            <span className="text-emerald-400">
                              Analize seçildi
                            </span>
                          )}
                        </div>
                      )}
                    </button>
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </section>
  );
}

export default UpcomingFixtures;
