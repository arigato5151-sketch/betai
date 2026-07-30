import { useEffect, useMemo, useState } from "react";

import { normalizeApiMode } from "./apiMode.js";
import LoginForm, { DemoModeBadge } from "./components/LoginForm.jsx";
import AdminContainer from "./containers/AdminContainer.jsx";
import AnalysisContainer from "./containers/AnalysisContainer.jsx";
import HistoryContainer from "./containers/HistoryContainer.jsx";
import OperationsContainer from "./containers/OperationsContainer.jsx";
import UpcomingFixturesContainer from "./containers/UpcomingFixturesContainer.jsx";
import { apiFetch, useAuth } from "./hooks/useAuth.js";
import { roleLabel } from "./localization.js";
import { allowedActions } from "./permissions.js";

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
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [fixtureSelection, setFixtureSelection] = useState({
    fixture: null,
    revision: 0,
  });
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const actions = useMemo(() => allowedActions(sessionUser), [sessionUser]);
  const canManageAdmin = actions.manageUsers && actions.manageRoles;

  useEffect(() => {
    apiFetch("/status", {}, false)
      .then(async (response) => {
        if (!response.ok) throw new Error("Platform durumu alınamadı.");
        const status = await response.json();
        setApiMode(normalizeApiMode(status));
        setRegistrationEnabled(Boolean(status.registration_enabled));
      })
      .catch(() => setApiMode("unknown"));
  }, []);

  const handleLogout = async () => {
    await logout();
    setSelectedMatch(null);
    setAdminPanelOpen(false);
  };

  const handleFixtureSelection = (fixture) => {
    setSelectedMatch(null);
    setFixtureSelection((current) => ({
      fixture,
      revision: current.revision + 1,
    }));
  };

  const clearFixtureSelection = () => {
    setFixtureSelection((current) => ({
      fixture: null,
      revision: current.revision,
    }));
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
            <h1 className="text-2xl font-black tracking-wider text-emerald-400">
              BET AI TAHMİN PLATFORMU
            </h1>
            <DemoModeBadge apiMode={apiMode} />
          </div>
          <p className="mt-1 text-xs text-slate-500">
            {sessionUser?.username} ·{" "}
            {(sessionUser?.roles ?? []).map(roleLabel).join(", ")}
          </p>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          {canManageAdmin && (
            <button
              type="button"
              onClick={() => setAdminPanelOpen((open) => !open)}
              className="rounded border border-emerald-800 px-3 py-1 text-sm text-emerald-400 hover:border-emerald-500"
            >
              Kullanıcı Yönetimi
            </button>
          )}
          <button
            type="button"
            onClick={handleLogout}
            className="rounded border border-slate-700 px-3 py-1 text-sm hover:border-emerald-500"
          >
            Çıkış
          </button>
        </div>
      </header>

      <AdminContainer
        canManage={canManageAdmin}
        currentUserId={sessionUser?.id}
        onClose={() => setAdminPanelOpen(false)}
        open={adminPanelOpen}
        request={apiFetch}
      />

      <UpcomingFixturesContainer
        onSelectFixture={actions.analyze ? handleFixtureSelection : undefined}
        request={apiFetch}
        selectedFixtureId={fixtureSelection.fixture?.fixture_id}
      />

      <OperationsContainer actions={actions} request={apiFetch} />

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-8 lg:grid-cols-3">
        <AnalysisContainer
          actions={actions}
          fixtureSelection={fixtureSelection}
          onClearFixtureSelection={clearFixtureSelection}
          onHistoryChanged={() =>
            setHistoryRefreshToken((current) => current + 1)
          }
          onSelectMatch={setSelectedMatch}
          request={apiFetch}
          selectedMatch={selectedMatch}
        >
          <HistoryContainer
            canRead={actions.readHistory}
            hasSelectedMatch={Boolean(selectedMatch)}
            onSelectMatch={setSelectedMatch}
            refreshToken={historyRefreshToken}
            request={apiFetch}
          />
        </AnalysisContainer>
      </main>
    </div>
  );
}

export default App;
