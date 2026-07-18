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
import { Doughnut, Line } from "react-chartjs-2";
import { buildHistoryQuery } from "./history.js";
import { buildBankrollSeries } from "./bankroll.js";
import { allowedActions } from "./permissions.js";
import { normalizeApiMode } from "./apiMode.js";
import AdminPanel from "./AdminPanel.jsx";

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

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"
).replace(/\/$/, "");
const CSRF_COOKIE_NAME = import.meta.env.VITE_CSRF_COOKIE_NAME || "bet_ai_csrf";
const CSRF_HEADER_NAME = import.meta.env.VITE_CSRF_HEADER_NAME || "X-CSRF-Token";
const readCookie = (name) =>
  document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(`${name}=`))
    ?.slice(name.length + 1) || "";
const addCsrfHeader = (headers, method = "GET") => {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) return;
  const token = readCookie(CSRF_COOKIE_NAME);
  if (token) headers.set(CSRF_HEADER_NAME, decodeURIComponent(token));
};
const refreshAccessToken = async () => {
  const headers = new Headers({ "Content-Type": "application/json" });
  addCsrfHeader(headers, "POST");
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: "POST",
    credentials: "include",
    headers,
  });
  return response.ok;
};

const apiFetch = async (path, options = {}, allowRefresh = true) => {
  const headers = new Headers(options.headers || {});
  addCsrfHeader(headers, options.method || "GET");
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });
  if (response.status !== 401 || !allowRefresh) return response;

  const renewed = await refreshAccessToken();
  if (!renewed) {
    window.dispatchEvent(new Event("bet-ai:unauthorized"));
    return response;
  }

  // Refresh rotates the CSRF cookie together with the token pair.
  addCsrfHeader(headers, options.method || "GET");
  return fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });
};

const DemoModeBadge = ({ apiMode }) =>
  apiMode === "demo" ? (
    <span
      role="status"
      className="inline-flex rounded-full border border-amber-400/60 bg-amber-400/10 px-2.5 py-1 text-xs font-black uppercase tracking-wider text-amber-300"
    >
      Demo Modu
    </span>
  ) : null;

function App() {
  const [authenticated, setAuthenticated] = useState(null);
  const [apiMode, setApiMode] = useState("unknown");
  const [sessionUser, setSessionUser] = useState(null);
  const [credentials, setCredentials] = useState({ username: "", email: "", password: "" });
  const [registrationEnabled, setRegistrationEnabled] = useState(false);
  const [registerMode, setRegisterMode] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
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
    });
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

    apiFetch("/auth/session", {}, false)
      .then(async (response) => {
        if (!response.ok) {
          setAuthenticated(false);
          setSessionUser(null);
          return;
        }
        const session = await response.json();
        setSessionUser(session.user);
        setAuthenticated(true);
      })
      .catch(() => {
        setAuthenticated(false);
        setSessionUser(null);
      });
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
    const handleUnauthorized = () => {
      setAuthenticated(false);
      setSessionUser(null);
    };
    window.addEventListener("bet-ai:unauthorized", handleUnauthorized);
    return () => window.removeEventListener("bet-ai:unauthorized", handleUnauthorized);
  }, []);

  const handleLogin = async (event) => {
    event.preventDefault();
    setLoginLoading(true);
    setLoginError("");
    try {
      const endpoint = registerMode ? "/auth/register" : "/auth/login";
      const payload = registerMode ? credentials : {
        username: credentials.username,
        password: credentials.password,
      };
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || "Oturum işlemi başarısız.");
      }
      const session = await response.json();
      setSessionUser(session.user);
      setAuthenticated(true);
      setCredentials({ username: "", email: "", password: "" });
      setRegisterMode(false);
    } catch (error) {
      setLoginError(error.message || "Oturum açılamadı.");
    } finally {
      setLoginLoading(false);
    }
  };

  const handleLogout = async () => {
    await apiFetch("/auth/logout", {
      method: "POST",
    }, false);
    setAuthenticated(false);
    setSessionUser(null);
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
      <main className="flex min-h-screen items-center justify-center bg-slate-950 p-4 text-slate-100">
        <form onSubmit={handleLogin} className="w-full max-w-sm space-y-4 rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
          <div className="flex items-center justify-between gap-3">
            <h1 className="text-xl font-black text-emerald-400">BET AI PLATFORM PRO</h1>
            <DemoModeBadge apiMode={apiMode} />
          </div>
          <input
            required
            autoComplete="username"
            className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
            placeholder="Kullanıcı adı"
            value={credentials.username}
            onChange={(event) => setCredentials({ ...credentials, username: event.target.value })}
          />
          {registerMode && (
            <input
              required
              type="email"
              autoComplete="email"
              className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
              placeholder="E-posta"
              value={credentials.email}
              onChange={(event) => setCredentials({ ...credentials, email: event.target.value })}
            />
          )}
          <input
            required
            minLength={registerMode ? 12 : 8}
            type="password"
            autoComplete={registerMode ? "new-password" : "current-password"}
            className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
            placeholder="Parola"
            value={credentials.password}
            onChange={(event) => setCredentials({ ...credentials, password: event.target.value })}
          />
          {loginError && <p className="text-sm text-red-400">{loginError}</p>}
          <button disabled={loginLoading} className="w-full rounded-lg bg-emerald-500 p-3 font-bold text-slate-950 disabled:opacity-50">
            {loginLoading ? "İşleniyor..." : registerMode ? "Hesap Oluştur" : "Giriş Yap"}
          </button>
          {registrationEnabled && (
            <button
              type="button"
              onClick={() => {
                setRegisterMode((current) => !current);
                setLoginError("");
              }}
              className="w-full text-sm text-emerald-300 hover:text-emerald-200"
            >
              {registerMode ? "Mevcut hesapla giriş yap" : "Yeni hesap oluştur"}
            </button>
          )}
        </form>
      </main>
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

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-8 lg:grid-cols-3">
        {actions.analyze ? (
          <div className="h-fit rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
          <h2 className="mb-4 text-lg font-bold">Manuel Mac Girisi</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <input type="text" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm" placeholder="Ev Sahibi" value={formData.home_team} onChange={(e) => setFormData({ ...formData, home_team: e.target.value })} />
            <input type="text" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm" placeholder="Deplasman" value={formData.away_team} onChange={(e) => setFormData({ ...formData, away_team: e.target.value })} />
            <input type="number" step="0.01" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5 text-sm" placeholder="Oran" value={formData.odd} onChange={(e) => setFormData({ ...formData, odd: Number(e.target.value) })} />

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <label className="mb-1 block font-bold text-amber-400">Ev Form (0-100)</label>
                <input type="number" min="0" max="100" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2" value={formData.home_stats.form} onChange={(e) => setFormData({ ...formData, home_stats: { ...formData.home_stats, form: Number(e.target.value) } })} />
              </div>
              <div>
                <label className="mb-1 block font-bold text-red-400">Dep Form (0-100)</label>
                <input type="number" min="0" max="100" className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2" value={formData.away_stats.form} onChange={(e) => setFormData({ ...formData, away_stats: { ...formData.away_stats, form: Number(e.target.value) } })} />
              </div>
            </div>

            <button type="submit" disabled={loading} className="w-full rounded-lg bg-emerald-500 p-3 text-sm font-bold text-slate-950 transition hover:bg-emerald-600 disabled:opacity-50">
              {loading ? "Analiz Ediliyor..." : "Yapay Zekayi Calistir"}
            </button>
          </form>
          </div>
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

          {actions.runBacktest && <section className="rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl" aria-labelledby="bankroll-title">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 id="bankroll-title" className="text-sm font-bold uppercase tracking-wider text-slate-400">Bankroll Gecmisi</h3>
                <p className="mt-1 text-xs text-slate-500">Fractional Kelly stratejisi, baslangic bankroll: 1.000</p>
              </div>
              <button type="button" onClick={runBacktest} disabled={backtestLoading} className="rounded bg-emerald-500 px-4 py-2 text-sm font-bold text-slate-950 disabled:opacity-50">
                {backtestLoading ? "Hesaplaniyor..." : "Backtest Calistir"}
              </button>
            </div>
            {backtestError && <p className="rounded border border-red-900 bg-red-950/40 p-3 text-sm text-red-400">{backtestError}</p>}
            {backtest && (
              <>
                <div className="mb-4 grid grid-cols-2 gap-2 text-center sm:grid-cols-4">
                  <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Son Bankroll</span><strong>{backtest.final_bankroll}</strong></div>
                  <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Net Degisim</span><strong className={bankrollSeries.change >= 0 ? "text-emerald-400" : "text-red-400"}>{bankrollSeries.change}</strong></div>
                  <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">ROI</span><strong>%{backtest.total_roi_pct}</strong></div>
                  <div className="rounded bg-slate-950 p-2"><span className="block text-xs text-slate-500">Toplam Bahis</span><strong>{backtest.total_bets}</strong></div>
                </div>
                <div className="h-64" data-testid="bankroll-chart">
                  <Line
                    data={{ labels: bankrollSeries.labels, datasets: [{ label: "Bankroll", data: bankrollSeries.values, borderColor: "#34d399", backgroundColor: "rgba(52, 211, 153, 0.12)", fill: true, tension: 0.25, pointRadius: 3 }] }}
                    options={{ maintainAspectRatio: false, interaction: { intersect: false, mode: "index" }, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: "#64748b" }, grid: { display: false } }, y: { ticks: { color: "#64748b" }, grid: { color: "rgba(71, 85, 105, 0.25)" } } } }}
                  />
                </div>
              </>
            )}
            {!backtest && !backtestError && <p className="rounded-lg border border-dashed border-slate-800 p-4 text-sm text-slate-500">Bankroll egirisini ve risk metriklerini gormek icin backtest calistirin.</p>}
          </section>}

          <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400">Veritabanindan Son Istekler</h3>
              <span className="text-xs text-slate-500">{history.length}/{historyMeta.total} kayit</span>
            </div>
            <div className="mb-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <label className="sm:col-span-2 lg:col-span-1">
                <span className="sr-only">Takim ara</span>
                <input
                  type="search"
                  placeholder="Takim ara..."
                  value={historyFilters.query}
                  onChange={(event) => updateHistoryFilter("query", event.target.value)}
                  className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-emerald-500"
                />
              </label>
              <label>
                <span className="sr-only">Sonuc filtresi</span>
                <select value={historyFilters.result} onChange={(event) => updateHistoryFilter("result", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
                  <option value="all">Tum sonuclar</option>
                  <option value="pending">Sonuc bekleyen</option>
                  <option value="HOME_WIN">Ev kazandi</option>
                  <option value="DRAW">Beraberlik</option>
                  <option value="AWAY_WIN">Deplasman kazandi</option>
                </select>
              </label>
              <label>
                <span className="sr-only">Value bet filtresi</span>
                <select value={historyFilters.value} onChange={(event) => updateHistoryFilter("value", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
                  <option value="all">Tum bahisler</option>
                  <option value="value">Sadece value</option>
                  <option value="non_value">Value olmayan</option>
                </select>
              </label>
              <label>
                <span className="sr-only">Siralama</span>
                <select value={historyFilters.sort} onChange={(event) => updateHistoryFilter("sort", event.target.value)} className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
                  <option value="newest">En yeni</option>
                  <option value="oldest">En eski</option>
                  <option value="edge">En yuksek edge</option>
                  <option value="odd">En yuksek oran</option>
                </select>
              </label>
            </div>
            <div className="space-y-2">
              {history.map((item) => (
                <button key={item.id} type="button" onClick={() => handleSelectMatch(item)} className="flex w-full cursor-pointer items-center justify-between rounded-lg border border-slate-800 bg-slate-950 p-3 text-left transition hover:border-slate-600">
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
            {historyMeta.pages > 1 && (
              <nav className="mt-4 flex items-center justify-between" aria-label="Gecmis sayfalama">
                <button type="button" disabled={historyPage <= 1 || historyLoading} onClick={() => setHistoryPage((page) => Math.max(1, page - 1))} className="rounded border border-slate-700 px-3 py-1.5 text-sm disabled:opacity-40">Onceki</button>
                <span className="text-xs text-slate-500">Sayfa {historyPage} / {historyMeta.pages}</span>
                <button type="button" disabled={historyPage >= historyMeta.pages || historyLoading} onClick={() => setHistoryPage((page) => Math.min(historyMeta.pages, page + 1))} className="rounded border border-slate-700 px-3 py-1.5 text-sm disabled:opacity-40">Sonraki</button>
              </nav>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
