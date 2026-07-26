import { useMemo } from "react";
import { ArcElement, Chart as ChartJS, Tooltip } from "chart.js";
import { Doughnut } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip);

function AnalysisReport({ canUpdateResult, match, onSubmitActualResult }) {
  const chartData = useMemo(
    () => ({
      labels: ["Ev Sahibi", "Deplasman", "Beraberlik"],
      datasets: [
        {
          data: [
            match.analysis.all_probabilities.HOME_WIN,
            match.analysis.all_probabilities.AWAY_WIN,
            match.analysis.all_probabilities.DRAW,
          ],
          backgroundColor: ["#fbbf24", "#ef4444", "#4b5563"],
          borderWidth: 0,
        },
      ],
    }),
    [match],
  );

  return (
    <div className="grid grid-cols-1 gap-6 rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl md:grid-cols-2">
      <div>
        <span className="text-xs font-bold uppercase tracking-widest text-emerald-400">
          Aktif Analiz Raporu
        </span>
        <h2 className="mb-4 mt-1 text-xl font-black text-white">{match.match}</h2>

        <div className="space-y-3">
          <div>
            <span className="text-xs text-slate-400">Yapay Zeka Tahmini</span>
            <p className="text-lg font-bold text-amber-400">
              {match.analysis.prediction} (%{match.analysis.probability})
            </p>
          </div>
          {match.data_quality && (
            <div className="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs">
              <span className="text-slate-500">Analiz veri skoru</span>
              <strong className="ml-2 text-emerald-400">
                {match.data_quality.score}/100
              </strong>
              {match.provenance?.model_name && (
                <p className="mt-2 text-slate-400">
                  Model: {match.provenance.model_name}
                  {match.provenance.model_artifact_version
                    ? ` Â· ${match.provenance.model_artifact_version}`
                    : ""}
                </p>
              )}
            </div>
          )}
          <div>
            <span className="text-xs text-slate-400">
              Finansal Deger / Value Bet
            </span>
            <p
              className={`text-sm font-bold ${
                match.value_assessment.value_bet
                  ? "text-emerald-400"
                  : "text-slate-400"
              }`}
            >
              {match.value_assessment.value_bet
                ? `VALUE FOUND (+%${match.value_assessment.edge})`
                : `Degersiz Oran (%${match.value_assessment.edge})`}
            </p>
          </div>
          <div>
            <span className="text-xs text-slate-400">
              ML Model Guven Durumu
            </span>
            <p className="mt-1 text-xs">
              <span
                className={`rounded px-2 py-1 font-bold ${
                  match.ml_safety_trigger === "HIGH_CONFIDENCE"
                    ? "bg-emerald-950 text-emerald-400"
                    : match.ml_safety_trigger === "YETERLI VERI YOK"
                      ? "bg-slate-800 text-slate-400"
                      : "bg-red-950 text-red-400"
                }`}
              >
                {match.ml_safety_trigger}
              </span>
              {match.ml_samples !== undefined && (
                <span className="ml-2 text-slate-500">
                  ({match.ml_samples}/{match.ml_min_samples ?? 200} ornek)
                </span>
              )}
            </p>
          </div>
          {canUpdateResult && match.record_id && !match.actual_result && (
            <div className="flex flex-wrap gap-2 pt-2">
              <span className="w-full text-xs text-slate-500">
                Gercek sonucu gir:
              </span>
              {["HOME_WIN", "DRAW", "AWAY_WIN"].map((result) => (
                <button
                  key={result}
                  type="button"
                  onClick={() => onSubmitActualResult(match.record_id, result)}
                  className="rounded border border-slate-700 px-2 py-1 text-xs hover:border-emerald-500"
                >
                  {result}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="flex flex-col items-center justify-center border-t border-slate-800 pt-4 md:border-l md:border-t-0 md:pt-0">
        <div className="h-40 w-40">
          <Doughnut
            data={chartData}
            options={{
              maintainAspectRatio: false,
              plugins: { legend: { display: false } },
            }}
          />
        </div>
        <div className="mt-4 flex gap-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-amber-400" /> Ev
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-red-400" /> Dep
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-gray-500" /> Beraberlik
          </span>
        </div>
      </div>
    </div>
  );
}

export default AnalysisReport;
