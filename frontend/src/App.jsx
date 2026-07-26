import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart as ChartJS,
  ArcElement,
  CategoryScale,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Doughnut } from "react-chartjs-2";
import { buildHistoryQuery } from "./history.js";
import { buildBankrollSeries } from "./bankroll.js";
import { allowedActions } from "./permissions.js";
import { normalizeApiMode } from "./apiMode.js";
import AdminPanel from "./AdminPanel.jsx";
import AnalysisForm from "./components/AnalysisForm.jsx";
import BankrollChart from "./components/BankrollChart.jsx";
import DataQualityCard from "./components/DataQualityCard.jsx";
import HistoryTable from "./components/HistoryTable.jsx";
import LoginForm, { DemoModeBadge } from "./components/LoginForm.jsx";
import ModelStatusCard from "./components/ModelStatusCard.jsx";
import { apiFetch, useAuth } from "./hooks/useAuth.js";

ChartJS.register(
  ArcElement,
  CategoryScale,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
);

function App() {
  const {
    authenticated,
    credentials,
    login,
    loginError,
    loginLoading,
    logout,
    registerMode,
    sessionUser,
    setCredentials,
    toggleRegisterMode,
  } = useAuth();
  const [apiMode, setApiMode] = useState("unknown");
  const [registrationEnabled, setRegistrationEnabled] = useState(false);
  const [history, setHistory] = useState([]);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyMeta, setHistoryMeta] = useState({ total: 0, pages: 1 });
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const historyRequestId = useRef(0);
  const [historyFilters, setHistoryFilters] = useState({
    query: "",
    result: "all",
    value: "all",
    sort: "newest",
  });
  const [loading, setLoading] = useState(false);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState("");
  const [dataQuality, setDataQuality] = useState(null);
  const [dataQualityLoading, setDataQualityLoading] = useState(false);
  const [dataQualityError, setDataQualityError] = useState("");
  const [modelStatus, setModelStatus] = useState(null);
  const [modelStatusLoading, setModelStatusLoading] = useState(false);
  const [modelStatusError, setModelStatusError] = useState("");
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const bankrollSeries = useMemo(
    () => buildBankrollSeries(backtest?.bankroll_history),
    [backtest],
  );
  const actions = useMemo(() => allowedActions(sessionUser), [sessionUser]);

  const [formData, setFormData] = useState({
    home_team: "Fenerbahce",
    away_team: "Galatasaray",
    odd: 2.3,
    home_stats: { form: 93, attack: 88, defense: 85, xg: 2.15 },
    away_stats: { form: 73, attack: 82, defense: 75, xg: 1.9 },
  });

  const handleSelectMatch = (rawDbItem) => {
    const probHome = rawDbItem.prob_home ?? rawDbItem.probability ?? 33;
    const probAway = rawDbItem.prob_away ?? 33;
    const probDraw = rawDbItem.prob_draw ?? 34;

    const mlInsufficient = rawDbItem.ml_cluster === null || rawDbItem.ml_cluster === undefined;

    setSelectedMatch({
      match: `${rawDbItem.home_team} vs ${rawDbItem.away_team}`,
      odd: rawDbItem.odd,
      home_stats: {
        form: rawDbItem.home_form,
        attack: rawDbItem.home_attack ?? 80,
        defense: rawDbItem.home_defense ?? 80,
        xg: rawDbItem.home_xg,
      },
      away_stats: {
        form: rawDbItem.away_form,
        attack: rawDbItem.away_attack ?? 80,
        defense: rawDbItem.away_defense ?? 80,
        xg: rawDbItem.away_xg,
      },
      analysis: {
        prediction: rawDbItem.prediction,
        probability: rawDbItem.probability,
        all_probabilities: {
          HOME_WIN: probHome,
          AWAY_WIN: probAway,
          DRAW: probDraw,
        },
      },
      value_assessment: { value_bet: rawDbItem.is_value_bet === 1, edge: rawDbItem.edge },
      ml_safety_trigger: mlInsufficient
        ? "YETERLI VERI YOK"
        : rawDbItem.ml_cluster === 1
          ? "HIGH_CONFIDENCE"
          : "RISKY_UNDERDOG",
      ml_ready: !mlInsufficient,
      record_id: rawDbItem.id,
      actual_result: rawDbItem.actual_result,
      data_quality: rawDbItem.data_quality,
      provenance: {
        model_name: rawDbItem.model_name,
        model_artifact_version: rawDbItem.model_artifact_version,
        feature_schema_version: rawDbItem.feature_schema_version,
        ensemble_version: rawDbItem.ensemble_version,
        analysis_lead_minutes: rawDbItem.analysis_lead_minutes,
      },
    });
  };

  const fetchDataQuality = async () => {
    if (!actions.readAudit) return;
    setDataQualityLoading(true);
    setDataQualityError("");
    try {
      const response = await apiFetch("/operations/data-quality");
      if (!response.ok) throw new Error("Veri kalitesi durumu alınamadı.");
      setDataQuality(await response.json());
    } catch (error) {
      setDataQualityError(error.message || "Veri kalitesi durumu alınamadı.");
    } finally {
      setDataQualityLoading(false);
    }
  };

  const fetchModelStatus = async () => {
    if (!actions.readHistory) return;
    setModelStatusLoading(true);
    setModelStatusError("");
    try {
      const response = await apiFetch("/ml/status");
      if (!response.ok) throw new Error("ML model durumu alınamadı.");
      setModelStatus(await response.json());
    } catch (error) {
      setModelStatusError(error.message || "ML model durumu alınamadı.");
    } finally {
      setModelStatusLoading(false);
    }
  };

  const fetchHistory = (requestedPage = historyPage) => {
    const requestId = ++historyRequestId.current;
    setHistoryLoading(true);
    setHistoryError("");
    apiFetch(`/history${buildHistoryQuery(historyFilters, requestedPage)}`)
      .then((res) => {
        if (!res.ok) throw new Error("Tahmin geçmişi alınamadı.");
        return res.json();
      })
      .then((data) => {
        if (requestId !== historyRequestId.current) return;
        const items = Array.isArray(data)
          ? data
          : Array.isArray(data.items)
            ? data.items
            : [];
        setHistory(items);
        setHistoryMeta({ total: data.total ?? items.length, pages: data.pages ?? 1 });
        if (items.length > 0 && !selectedMatch) {
          handleSelectMatch(items[0]);
        }
      })
      .catch((error) => {
        if (requestId !== historyRequestId.current) return;
        setHistory([]);
        setHistoryMeta({ total: 0, pages: 1 });
        setHistoryError(error.message || "Tahmin gecmisi alinamadi.");
      })
      .finally(() => {
        if (requestId === historyRequestId.current) setHistoryLoading(false);
      });
  };

  const updateHistoryFilter = (key, value) => {
    setHistoryFilters((current) => ({ ...current, [key]: value }));
    setHistoryPage(1);
  };

  useEffect(() => {
    apiFetch("/status", {}, false)
      .then(async (response) => {
        if (!response.ok) throw new Error("Platform durumu alinamadi.");
        const status = await response.json();
        setApiMode(normalizeApiMode(status));
        setRegistrationEnabled(Boolean(status.registration_enabled));
      })
      .catch(() => setApiMode("unknown"));

  }, []);

  useEffect(() => {
    if (authenticated !== true || !actions.readHistory) return undefined;
    const timeoutId = window.setTimeout(() => fetchHistory(), 250);
    return () => window.clearTimeout(timeoutId);
  }, [
    authenticated,
    actions.readHistory,
    historyPage,
    historyFilters.query,
    historyFilters.result,
    historyFilters.value,
    historyFilters.sort,
  ]);

  useEffect(() => {
    if (authenticated === true && actions.readAudit) fetchDataQuality();
  }, [authenticated, actions.readAudit]);

  useEffect(() => {
    if (authenticated === true && actions.readHistory) fetchModelStatus();
  }, [authenticated, actions.readHistory]);

  const handleLogout = async () => {
    await logout();
    setHistory([]);
    setSelectedMatch(null);
    setAdminPanelOpen(false);
  };

  const submitActualResult = (recordId, result) => {
    if (!actions.updateResult) return;
    apiFetch(`/history/${recordId}/result`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actual_result: result }),
    })
      .then((res) => {
        if (!res.ok) throw new Error("Sonuc kaydedilemedi.");
        return res.json();
      })
      .then(() => fetchHistory())
      .catch((err) => alert(err.message));
  };

  const runBacktest = async () => {
    if (!actions.runBacktest) return;
    setBacktestLoading(true);
    setBacktestError("");
    try {
      const response = await apiFetch("/backtest", {
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

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!actions.analyze) return;
    setLoading(true);
    apiFetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error("Analiz istegi basarisiz oldu.");
        }
        return res.json();
      })
      .then((data) => {
        fetchHistory();
        setSelectedMatch({
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
      })
      .finally(() => setLoading(false));
  };

  const chartData = {
    labels: ["Ev Sahibi", "Deplasman", "Beraberlik"],
    datasets: [
      {
        data: selectedMatch
          ? [
              selectedMatch.analysis.all_probabilities.HOME_WIN,
              selectedMatch.analysis.all_probabilities.AWAY_WIN,
              selectedMatch.analysis.all_probabilities.DRAW,
            ]
          : [33, 33, 34],
        backgroundColor: ["#fbbf24", "#ef4444", "#4b5563"],
        borderWidth: 0,
      },
    ],
  };

  if (authenticated === null) {
    return <main className="min-h-screen bg-slate-950" />;
  }

  if (!authenticated) {
    return (
      <LoginForm
        apiMode={apiMode}
        credentials={credentials}
        loginError={loginError}
        loginLoading={loginLoading}
        onCredentialsChange={setCredentials}
        onSubmit={login}
        onToggleRegisterMode={toggleRegisterMode}
        registerMode={registerMode}
        registrationEnabled={registrationEnabled}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 p-4 font-sans text-slate-100 md:p-8">
      <header className="mx-auto mb-8 flex max-w-7xl items-center justify-between border-b border-slate-800 pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-black tracking-wider text-emerald-400">BET AI PLATFORM PRO</h1>
            <DemoModeBadge apiMode={apiMode} />
          </div>
          <p className="mt-1 text-xs text-slate-500">{sessionUser?.username} · {(sessionUser?.roles ?? []).join(", ")}</p>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          {actions.manageUsers && actions.manageRoles && (
            <button type="button" onClick={() => setAdminPanelOpen((open) => !open)} className="rounded border border-emerald-800 px-3 py-1 text-sm text-emerald-400 hover:border-emerald-500">
              Kullanıcı Yönetimi
            </button>
          )}
          <button type="button" onClick={handleLogout} className="rounded border border-slate-700 px-3 py-1 text-sm hover:border-emerald-500">Çıkış</button>
        </div>
      </header>

      {adminPanelOpen && actions.manageUsers && actions.manageRoles && (
        <AdminPanel request={apiFetch} currentUserId={sessionUser?.id} onClose={() => setAdminPanelOpen(false)} />
      )}

      {actions.readAudit && (
        <DataQualityCard
          data={dataQuality}
          error={dataQualityError}
          loading={dataQualityLoading}
          onRefresh={fetchDataQuality}
        />
      )}

      {actions.readHistory && (
        <ModelStatusCard
          status={modelStatus}
          error={modelStatusError}
          loading={modelStatusLoading}
          onRefresh={fetchModelStatus}
        />
      )}

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-8 lg:grid-cols-3">
        {actions.analyze ? (
          <AnalysisForm
            formData={formData}
            loading={loading}
            onChange={setFormData}
            onSubmit={handleSubmit}
          />
        ) : (
          <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 text-sm text-slate-400 shadow-xl">
            <h2 className="mb-2 font-bold text-slate-200">Salt Okunur Oturum</h2>
            <p>Bu rol yeni analiz oluşturma yetkisine sahip değil.</p>
          </div>
        )}

        <div className="space-y-6 lg:col-span-2">
          {selectedMatch && (
            <div className="grid grid-cols-1 gap-6 rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl md:grid-cols-2">
              <div>
                <span className="text-xs font-bold uppercase tracking-widest text-emerald-400">Aktif Analiz Raporu</span>
                <h2 className="mb-4 mt-1 text-xl font-black text-white">{selectedMatch.match}</h2>

                <div className="space-y-3">
                  <div>
                    <span className="text-xs text-slate-400">Yapay Zeka Tahmini</span>
                    <p className="text-lg font-bold text-amber-400">{selectedMatch.analysis.prediction} (%{selectedMatch.analysis.probability})</p>
                  </div>
                  {selectedMatch.data_quality && (
                    <div className="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs">
                      <span className="text-slate-500">Analiz veri skoru</span>
                      <strong className="ml-2 text-emerald-400">{selectedMatch.data_quality.score}/100</strong>
                      {selectedMatch.provenance?.model_name && (
                        <p className="mt-2 text-slate-400">
                          Model: {selectedMatch.provenance.model_name}
                          {selectedMatch.provenance.model_artifact_version
                            ? ` · ${selectedMatch.provenance.model_artifact_version}`
                            : ""}
                        </p>
                      )}
                    </div>
                  )}
                  <div>
                    <span className="text-xs text-slate-400">Finansal Deger / Value Bet</span>
                    <p className={`text-sm font-bold ${selectedMatch.value_assessment.value_bet ? "text-emerald-400" : "text-slate-400"}`}>
                      {selectedMatch.value_assessment.value_bet ? `VALUE FOUND (+%${selectedMatch.value_assessment.edge})` : `Degersiz Oran (%${selectedMatch.value_assessment.edge})`}
                    </p>
                  </div>
                  <div>
                    <span className="text-xs text-slate-400">ML Model Guven Durumu</span>
                    <p className="mt-1 text-xs">
                      <span
                        className={`rounded px-2 py-1 font-bold ${
                          selectedMatch.ml_safety_trigger === "HIGH_CONFIDENCE"
                            ? "bg-emerald-950 text-emerald-400"
                            : selectedMatch.ml_safety_trigger === "YETERLI VERI YOK"
                              ? "bg-slate-800 text-slate-400"
                              : "bg-red-950 text-red-400"
                        }`}
                      >
                        {selectedMatch.ml_safety_trigger}
                      </span>
                      {selectedMatch.ml_samples !== undefined && (
                        <span className="ml-2 text-slate-500">
                          ({selectedMatch.ml_samples}/{selectedMatch.ml_min_samples ?? 200} ornek)
                        </span>
                      )}
                    </p>
                  </div>
                  {actions.updateResult && selectedMatch.record_id && !selectedMatch.actual_result && (
                    <div className="flex flex-wrap gap-2 pt-2">
                      <span className="w-full text-xs text-slate-500">Gercek sonucu gir:</span>
                      {["HOME_WIN", "DRAW", "AWAY_WIN"].map((r) => (
                        <button
                          key={r}
                          type="button"
                          onClick={() => submitActualResult(selectedMatch.record_id, r)}
                          className="rounded border border-slate-700 px-2 py-1 text-xs hover:border-emerald-500"
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="flex flex-col items-center justify-center border-t border-slate-800 pt-4 md:border-l md:border-t-0 md:pt-0">
                <div className="h-40 w-40">
                  <Doughnut data={chartData} options={{ maintainAspectRatio: false, plugins: { legend: { display: false } } }} />
                </div>
                <div className="mt-4 flex gap-4 text-xs">
                  <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-amber-400"></span> Ev</span>
                  <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-red-400"></span> Dep</span>
                  <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-gray-500"></span> Beraberlik</span>
                </div>
              </div>
            </div>
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

          <HistoryTable
            filters={historyFilters}
            history={history}
            historyError={historyError}
            historyLoading={historyLoading}
            meta={historyMeta}
            onFilterChange={updateHistoryFilter}
            onPageChange={setHistoryPage}
            onSelectMatch={handleSelectMatch}
            page={historyPage}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
