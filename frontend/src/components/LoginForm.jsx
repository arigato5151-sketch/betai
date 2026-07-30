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
      <form onSubmit={onSubmit} className="w-full max-w-sm space-y-4 rounded-lg border border-slate-800 bg-slate-900 p-6 shadow-xl">
        <div className="flex items-center justify-between gap-3">
          <h1 className="text-xl font-black text-emerald-400">BET AI TAHMİN PLATFORMU</h1>
          <DemoModeBadge apiMode={apiMode} />
        </div>
        <input
          required
          autoComplete="username"
          className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
          placeholder="Kullanıcı adı"
          value={credentials.username}
          onChange={(event) => onCredentialsChange({ ...credentials, username: event.target.value })}
        />
        {registerMode && (
          <input
            required
            type="email"
            autoComplete="email"
            className="w-full rounded-lg border border-slate-800 bg-slate-950 p-2.5"
            placeholder="E-posta"
            value={credentials.email}
            onChange={(event) => onCredentialsChange({ ...credentials, email: event.target.value })}
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
          onChange={(event) => onCredentialsChange({ ...credentials, password: event.target.value })}
        />
        {loginError && <p className="text-sm text-red-400">{loginError}</p>}
        <button disabled={loginLoading} className="w-full rounded-lg bg-emerald-500 p-3 font-bold text-slate-950 disabled:opacity-50">
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
