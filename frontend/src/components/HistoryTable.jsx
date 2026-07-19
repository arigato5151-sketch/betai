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
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">Veritabanindan Son Istekler</h3>
        <span className="text-xs text-slate-500">{history.length}/{meta.total} kayit</span>
      </div>
      <div className="mb-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <label className="sm:col-span-2 lg:col-span-1">
          <span className="sr-only">Takim ara</span>
          <input
            type="search"
            placeholder="Takim ara..."
            value={filters.query}
            onChange={(event) => onFilterChange("query", event.target.value)}
            className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
          />
        </label>
        <label>
          <span className="sr-only">Sonuc filtresi</span>
          <select value={filters.result} onChange={(event) => onFilterChange("result", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
            <option value="all">Tum sonuclar</option>
            <option value="pending">Sonuc bekleyen</option>
            <option value="HOME_WIN">Ev kazandi</option>
            <option value="DRAW">Beraberlik</option>
            <option value="AWAY_WIN">Deplasman kazandi</option>
          </select>
        </label>
        <label>
          <span className="sr-only">Value bet filtresi</span>
          <select value={filters.value} onChange={(event) => onFilterChange("value", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
            <option value="all">Tum bahisler</option>
            <option value="value">Sadece value</option>
            <option value="non_value">Value olmayan</option>
          </select>
        </label>
        <label>
          <span className="sr-only">Siralama</span>
          <select value={filters.sort} onChange={(event) => onFilterChange("sort", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
            <option value="newest">En yeni</option>
            <option value="oldest">En eski</option>
            <option value="edge">En yuksek edge</option>
            <option value="odd">En yuksek oran</option>
          </select>
        </label>
      </div>
      <div className="space-y-2">
        {history.map((item) => (
          <button key={item.id} type="button" onClick={() => onSelectMatch(item)} className="flex w-full cursor-pointer items-center justify-between rounded-lg border border-slate-800 bg-slate-950 p-3 text-left transition hover:border-slate-600">
            <span className="text-sm font-medium text-slate-200">
              {item.home_team} vs {item.away_team}
              {item.actual_result && <span className="ml-2 text-xs text-emerald-500">({item.actual_result})</span>}
            </span>
            <div className="flex items-center gap-2">
              <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-400">Oran: {item.odd}</span>
              <span className={`h-2 w-2 rounded-full ${item.is_value_bet === 1 ? "animate-pulse bg-amber-400" : "bg-slate-700"}`}></span>
            </div>
          </button>
        ))}
        {historyLoading && <p className="p-3 text-sm text-slate-500">Gecmis yukleniyor...</p>}
        {historyError && <p className="rounded border border-red-900 bg-red-950/40 p-3 text-sm text-red-400">{historyError}</p>}
        {!historyLoading && !historyError && history.length === 0 && (
          <p className="rounded-lg border border-dashed border-slate-800 p-4 text-sm text-slate-500">Filtrelerle eslesen kayit bulunamadi.</p>
        )}
      </div>
      {meta.pages > 1 && (
        <nav className="mt-4 flex items-center justify-between" aria-label="Gecmis sayfalama">
          <button type="button" disabled={page <= 1 || historyLoading} onClick={() => onPageChange(Math.max(1, page - 1))} className="rounded border border-slate-700 px-3 py-1.5 text-sm disabled:opacity-40">Onceki</button>
          <span className="text-xs text-slate-500">Sayfa {page} / {meta.pages}</span>
          <button type="button" disabled={page >= meta.pages || historyLoading} onClick={() => onPageChange(Math.min(meta.pages, page + 1))} className="rounded border border-slate-700 px-3 py-1.5 text-sm disabled:opacity-40">Sonraki</button>
        </nav>
      )}
    </div>
  );
}

export default HistoryTable;
