const SUPPORTED_API_MODES = new Set(["demo", "live"]);

export const normalizeApiMode = (payload) =>
  SUPPORTED_API_MODES.has(payload?.api_mode) ? payload.api_mode : "unknown";
