import { useMemo, useState } from "react";

import { buildBankrollSeries } from "../bankroll.js";
import AnalysisForm from "../components/AnalysisForm.jsx";
import AnalysisReport from "../components/AnalysisReport.jsx";
import BankrollChart from "../components/BankrollChart.jsx";

const initialFormData = {
  home_team: "Fenerbahce",
  away_team: "Galatasaray",
  odd: 2.3,
  home_stats: { form: 93, attack: 88, defense: 85, xg: 2.15 },
  away_stats: { form: 73, attack: 82, defense: 75, xg: 1.9 },
};

function AnalysisContainer({
  actions,
  children,
  onHistoryChanged,
  onSelectMatch,
  request,
  selectedMatch,
}) {
  const [formData, setFormData] = useState(initialFormData);
  const [loading, setLoading] = useState(false);
  const [backtest, setBacktest] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState("");
  const bankrollSeries = useMemo(
    () => buildBankrollSeries(backtest?.bankroll_history),
    [backtest],
  );

  const submitActualResult = async (recordId, result) => {
    if (!actions.updateResult) return;
    try {
      const response = await request(`/history/${recordId}/result`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actual_result: result }),
      });
      if (!response.ok) throw new Error("Sonuc kaydedilemedi.");
      await response.json();
      onHistoryChanged();
    } catch (error) {
      alert(error.message || "Sonuc kaydedilemedi.");
    }
  };

  const runBacktest = async () => {
    if (!actions.runBacktest) return;
    setBacktestLoading(true);
    setBacktestError("");
    try {
      const response = await request("/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_bankroll: 1000,
          strategy: "fractional_kelly",
          kelly_fraction: 0.25,
          min_edge_pct: 3,
          commission_pct: 2,
          max_stake_pct: 5,
          max_daily_exposure_pct: 15,
          require_closing_odds: false,
          exclude_post_kickoff: true,
        }),
      });
      if (!response.ok) throw new Error("Backtest calistirilamadi.");
      setBacktest(await response.json());
    } catch (error) {
      setBacktestError(error.message || "Backtest calistirilamadi.");
    } finally {
      setBacktestLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!actions.analyze) return;
    setLoading(true);
    try {
      const response = await request("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error("Analiz istegi basarisiz oldu.");
      const data = await response.json();
      onHistoryChanged();
      onSelectMatch({
        match: data.match,
        odd: formData.odd,
        home_stats: formData.home_stats,
        away_stats: formData.away_stats,
        analysis: data.analysis,
        value_assessment: data.value_assessment,
        ml_safety_trigger: data.ml_safety_trigger,
        ml_ready: data.ml_ready,
        ml_samples: data.ml_samples,
        ml_min_samples: data.ml_min_samples,
        data_quality: data.data_quality,
        provenance: data.provenance,
        insights: data.insights,
      });
    } catch (error) {
      alert(error.message || "Analiz istegi basarisiz oldu.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {actions.analyze ? (
        <AnalysisForm
          formData={formData}
          loading={loading}
          onChange={setFormData}
          onSubmit={handleSubmit}
        />
      ) : (
        <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400 shadow-xl">
          <h2 className="mb-2 font-bold text-slate-200">
            Salt Okunur Oturum
          </h2>
          <p>Bu rol yeni analiz oluÅŸturma yetkisine sahip deÄŸil.</p>
        </div>
      )}

      <div className="space-y-6 lg:col-span-2">
        {selectedMatch && (
          <AnalysisReport
            canUpdateResult={actions.updateResult}
            match={selectedMatch}
            onSubmitActualResult={submitActualResult}
          />
        )}

        {actions.runBacktest && (
          <BankrollChart
            backtest={backtest}
            bankrollSeries={bankrollSeries}
            error={backtestError}
            loading={backtestLoading}
            onRun={runBacktest}
          />
        )}

        {children}
      </div>
    </>
  );
}

export default AnalysisContainer;
