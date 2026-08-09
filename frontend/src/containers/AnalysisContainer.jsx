import { useEffect, useMemo, useRef, useState } from "react";

import { buildBankrollSeries } from "../bankroll.js";
import AnalysisForm from "../components/AnalysisForm.jsx";
import AnalysisReport from "../components/AnalysisReport.jsx";
import BankrollChart from "../components/BankrollChart.jsx";
import StatusMessage from "../components/StatusMessage.jsx";
import { useAnalysis } from "../hooks/useAnalysis.js";
import { useBacktest } from "../hooks/useBacktest.js";
import { useFixturePrefill, initialFormData } from "../hooks/useFixturePrefill.js";
import { useHistoryResultUpdate } from "../hooks/useHistoryResultUpdate.js";
import { useLeagues } from "../hooks/useLeagues.js";

function AnalysisContainer({
  actions,
  children,
  fixtureSelection,
  onClearFixtureSelection,
  onHistoryChanged,
  onSelectMatch,
  request,
  selectedMatch,
}) {
  const analysisSectionRef = useRef(null);
  const [formData, setFormData] = useState(initialFormData);

  const leaguesState = useLeagues(request);
  const fixtureState = useFixturePrefill({
    actions,
    fixtureSelection,
    formData,
    onClearFixtureSelection,
    request,
    setFormData,
  });
  const analysisState = useAnalysis({
    actions,
    formData,
    onHistoryChanged,
    onSelectMatch,
    request,
  });
  const backtestState = useBacktest({ actions, request });
  const resultState = useHistoryResultUpdate({
    actions,
    onHistoryChanged,
    request,
  });
  const bankrollSeries = useMemo(
    () => buildBankrollSeries(backtestState.backtest?.bankroll_history),
    [backtestState.backtest],
  );

  useEffect(() => {
    if (fixtureSelection?.fixture?.fixture_id) {
      analysisSectionRef.current?.scrollIntoView?.({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [fixtureSelection]);

  return (
    <>
      <section
        id="match-analysis-section"
        ref={analysisSectionRef}
        className="scroll-mt-6 lg:col-span-3"
      >
        <StatusMessage tone="error">{fixtureState.fixtureError}</StatusMessage>
        <StatusMessage tone="info">
          {!fixtureState.fixtureError ? fixtureState.fixtureMessage : null}
        </StatusMessage>
        <StatusMessage tone="stale">{fixtureState.fixtureStale ? "Maç verileri yeniden yükleniyor…" : null}</StatusMessage>
        {actions.analyze ? (
          <AnalysisForm
            fixtureLoading={fixtureState.fixtureLoading}
            formData={formData}
            leagues={leaguesState.leagues}
            leaguesError={leaguesState.leaguesError}
            leaguesLoading={leaguesState.leaguesLoading}
            loading={analysisState.analysisLoading}
            onChange={fixtureState.handleFormChange}
            onSubmit={analysisState.handleSubmit}
          />
        ) : (
          <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400 shadow-xl">
            <h2 className="mb-2 font-bold text-slate-200">
              Salt Okunur Oturum
            </h2>
            <p>Bu rol yeni analiz oluşturma yetkisine sahip değil.</p>
          </div>
        )}
        <StatusMessage tone="error">{analysisState.analysisError}</StatusMessage>
      </section>

      <div className="space-y-6 lg:col-span-3">
        {selectedMatch && (
          <AnalysisReport
            canUpdateResult={actions.updateResult}
            match={selectedMatch}
            onSubmitActualResult={resultState.submitActualResult}
            resultError={resultState.resultError}
            resultUpdating={resultState.resultUpdating}
          />
        )}

        {actions.runBacktest && (
          <BankrollChart
            backtest={backtestState.backtest}
            bankrollSeries={bankrollSeries}
            error={backtestState.backtestError}
            loading={backtestState.backtestLoading}
            onRun={backtestState.runBacktest}
          />
        )}

        {children}
      </div>
    </>
  );
}

export default AnalysisContainer;