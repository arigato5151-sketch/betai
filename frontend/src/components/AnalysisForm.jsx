function AnalysisForm({ formData, loading, onChange, onSubmit }) {
  return (
    <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
      <h2 className="mb-4 text-lg font-bold">Manuel Mac Girisi</h2>
      <form onSubmit={onSubmit} className="space-y-4">
        <input type="text" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm" placeholder="Ev Sahibi" value={formData.home_team} onChange={(event) => onChange({ ...formData, home_team: event.target.value })} />
        <input type="text" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm" placeholder="Deplasman" value={formData.away_team} onChange={(event) => onChange({ ...formData, away_team: event.target.value })} />
        <input type="number" step="0.01" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm" placeholder="Oran" value={formData.odd} onChange={(event) => onChange({ ...formData, odd: Number(event.target.value) })} />

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <label className="mb-1 block font-bold text-amber-400">Ev Form (0-100)</label>
            <input type="number" min="0" max="100" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2" value={formData.home_stats.form} onChange={(event) => onChange({ ...formData, home_stats: { ...formData.home_stats, form: Number(event.target.value) } })} />
          </div>
          <div>
            <label className="mb-1 block font-bold text-red-400">Dep Form (0-100)</label>
            <input type="number" min="0" max="100" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2" value={formData.away_stats.form} onChange={(event) => onChange({ ...formData, away_stats: { ...formData.away_stats, form: Number(event.target.value) } })} />
          </div>
        </div>

        <button type="submit" disabled={loading} className="w-full rounded-lg bg-emerald-500 p-3 text-sm font-bold text-slate-950 transition hover:bg-emerald-600 disabled:opacity-50">
          {loading ? "Analiz Ediliyor..." : "Yapay Zekayi Calistir"}
        </button>
      </form>
    </div>
  );
}

export default AnalysisForm;
