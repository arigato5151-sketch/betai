export const DemoModeBadge = ({ apiMode }) =>
  apiMode === "demo" ? (
    <span
      role="status"
      className="inline-flex rounded-full border border-amber-400/60 bg-amber-400/10 px-2.5 py-1 text-xs font-black uppercase tracking-wider text-amber-300"
    >
      Demo Modu
    </span>
  ) : null;

function LoginForm({
  apiMode,
  credentials,
  loginError,
  loginLoading,
  onCredentialsChange,
  onSubmit,
  onToggleRegisterMode,
  registerMode,
  registrationEnabled,
}) {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 p-4 text-slate-100">
      <form
        aria-busy={loginLoading}
        onSubmit={onSubmit}
        className="w-full max-w-sm space-y-4 rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl"
      >
        <div className="flex items-center justify-between gap-3">
          <h1 className="text-xl font-black text-emerald-400">BET AI TAHMİN PLATFORMU</h1>
          <DemoModeBadge apiMode={apiMode} />
        </div>
        <label className="block space-y-1" htmlFor="auth-username">
          <span className="text-sm text-slate-300">Kullanıcı adı</span>
          <input
            id="auth-username"
            required
            autoComplete="username"
            aria-invalid={Boolean(loginError)}
            aria-describedby={loginError ? "auth-error" : undefined}
            className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
            placeholder="Kullanıcı adı"
            value={credentials.username}
            onChange={(event) => onCredentialsChange({ ...credentials, username: event.target.value })}
          />
        </label>
        {registerMode && (
          <label className="block space-y-1" htmlFor="auth-email">
            <span className="text-sm text-slate-300">E-posta</span>
            <input
              id="auth-email"
              required
              type="email"
              autoComplete="email"
              aria-invalid={Boolean(loginError)}
              aria-describedby={loginError ? "auth-error" : undefined}
              className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
              placeholder="E-posta"
              value={credentials.email}
              onChange={(event) => onCredentialsChange({ ...credentials, email: event.target.value })}
            />
          </label>
        )}
        <label className="block space-y-1" htmlFor="auth-password">
          <span className="text-sm text-slate-300">Parola</span>
          <input
            id="auth-password"
            required
            minLength={registerMode ? 12 : 8}
            type="password"
            autoComplete={registerMode ? "new-password" : "current-password"}
            aria-invalid={Boolean(loginError)}
            aria-describedby={loginError ? "auth-error" : undefined}
            className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
            placeholder="Parola"
            value={credentials.password}
            onChange={(event) => onCredentialsChange({ ...credentials, password: event.target.value })}
          />
        </label>
        {loginError && <p id="auth-error" role="alert" className="text-sm text-red-400">{loginError}</p>}
        <button type="submit" disabled={loginLoading} className="w-full rounded-lg bg-emerald-500 p-3 font-bold text-slate-950 disabled:opacity-50">
          {loginLoading ? "İşleniyor..." : registerMode ? "Hesap Oluştur" : "Giriş Yap"}
        </button>
        {registrationEnabled && (
          <button
            type="button"
            onClick={onToggleRegisterMode}
            className="w-full text-sm text-emerald-300 hover:text-emerald-200"
          >
            {registerMode ? "Mevcut hesapla giriş yap" : "Yeni hesap oluştur"}
          </button>
        )}
      </form>
    </main>
  );
}

export default LoginForm;
