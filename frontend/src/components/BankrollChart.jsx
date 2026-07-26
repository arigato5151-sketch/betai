import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
);

function BankrollChart({ backtest, bankrollSeries, error, loading, onRun }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl" aria-labelledby="bankroll-title">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 id="bankroll-title" className="text-sm font-bold uppercase tracking-wider text-slate-400">Bankroll Gecmisi</h3>
          <p className="mt-1 text-xs text-slate-500">Fractional Kelly stratejisi, baslangic bankroll: 1.000</p>
        </div>
        <button type="button" onClick={onRun} disabled={loading} className="rounded bg-emerald-500 px-4 py-2 text-sm font-bold text-slate-950 disabled:opacity-50">
          {loading ? "Hesaplaniyor..." : "Backtest Calistir"}
        </button>
      </div>
      {error && <p className="rounded border border-red-900 bg-red-950/40 p-3 text-sm text-red-400">{error}</p>}
      {backtest && (
        <>
          <div className="mb-4 grid grid-cols-2 gap-2 text-center sm:grid-cols-4 lg:grid-cols-8">
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Son Bankroll</span><strong>{backtest.final_bankroll}</strong></div>
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Net Degisim</span><strong className={bankrollSeries.change >= 0 ? "text-emerald-400" : "text-red-400"}>{bankrollSeries.change}</strong></div>
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">ROI</span><strong>%{backtest.total_roi_pct}</strong></div>
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Toplam Bahis</span><strong>{backtest.total_bets}</strong></div>
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Max Drawdown</span><strong>%{backtest.max_drawdown_pct}</strong></div>
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Profit Factor</span><strong>{backtest.profit_factor}</strong></div>
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Ruin Riski</span><strong>%{backtest.risk_of_ruin_pct}</strong></div>
            <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Closing Odds</span><strong>%{backtest.closing_odds_coverage_pct}</strong></div>
          </div>
          {Object.keys(backtest.skipped_reasons ?? {}).length > 0 && (
            <p className="mb-4 text-xs text-slate-500">
              Atlanan kayıtlar: {Object.entries(backtest.skipped_reasons).map(([reason, count]) => `${reason}: ${count}`).join(" · ")}
            </p>
          )}
          <div className="h-64" data-testid="bankroll-chart">
            <Line
              data={{ labels: bankrollSeries.labels, datasets: [{ label: "Bankroll", data: bankrollSeries.values, borderColor: "#34d399", backgroundColor: "rgba(52, 211, 153, 0.12)", fill: true, tension: 0.25, pointRadius: 3 }] }}
              options={{ maintainAspectRatio: false, interaction: { intersect: false, mode: "index" }, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: "#64748b" }, grid: { display: false } }, y: { ticks: { color: "#64748b" }, grid: { color: "rgba(71, 85, 105, 0.25)" } } } }}
            />
          </div>
        </>
      )}
      {!backtest && !error && <p className="rounded-lg border border-dashed border-slate-800 p-4 text-sm text-slate-500">Bankroll egirisini ve risk metriklerini gormek icin backtest calistirin.</p>}
    </section>
  );
}

export default BankrollChart;
