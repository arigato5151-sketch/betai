const TONES = {
  error: "border-red-900/70 bg-red-950/40 text-red-300",
  info: "border-emerald-900/70 bg-emerald-950/40 text-emerald-300",
  warning: "border-amber-800 bg-amber-950/40 text-amber-300",
  stale: "border-slate-700 bg-slate-900/80 text-slate-400",
};

function StatusMessage({ children, id, tone = "info" }) {
  if (!children) return null;
  const isError = tone === "error";
  return (
    <p
      id={id}
      role={isError ? "alert" : "status"}
      aria-live={isError ? "assertive" : "polite"}
      className={`mb-3 rounded-lg border p-3 text-sm ${TONES[tone] ?? TONES.info}`}
    >
      {children}
    </p>
  );
}

export default StatusMessage;