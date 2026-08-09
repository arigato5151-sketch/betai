export class DataValidationError extends Error {
  constructor(message, field = null) {
    super(message);
    this.name = "DataValidationError";
    this.field = field;
  }
}

const isRecord = (value) =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const fail = (message, field) => {
  throw new DataValidationError(message, field);
};

const finiteNumber = (value, label) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${label} sayısal bir değer değil`, label);
  }
  return value;
};

const nonEmptyString = (value, label) => {
  if (typeof value !== "string" || value.trim() === "") {
    fail(`${label} boş olamaz`, label);
  }
  return value;
};

export function validateLeaguesResponse(payload) {
  if (!Array.isArray(payload)) {
    fail("Lig listesi geçerli bir dizi değil", "leagues");
  }
  const unique = new Map();
  for (const league of payload) {
    if (!isRecord(league)) {
      fail("Lig kaydı geçerli bir nesne değil", "leagues.item");
    }
    if (!Number.isInteger(league.id) || league.id <= 0) {
      fail("Lig kimliği geçersiz", "leagues.id");
    }
    nonEmptyString(league.name, "leagues.name");
    unique.set(league.id, league);
  }
  return [...unique.values()];
}

export function validatePrefillResponse(payload) {
  if (!isRecord(payload)) {
    fail("Maç ön dolum yanıtı geçerli bir nesne değil", "prefill");
  }
  nonEmptyString(payload.home_team, "prefill.home_team");
  nonEmptyString(payload.away_team, "prefill.away_team");
  if (typeof payload.odd !== "number" || payload.odd <= 1) {
    fail("prefill.odd 1'den büyük sayısal bir değer değil", "prefill.odd");
  }
  return payload;
}

export function validateAnalysisResponse(payload) {
  if (!isRecord(payload)) {
    fail("Analiz yanıtı geçerli bir nesne değil", "analysis");
  }
  nonEmptyString(payload.match, "analysis.match");
  if (!isRecord(payload.analysis)) {
    fail("analysis bloğu eksik", "analysis.analysis");
  }
  const allowedPredictions = ["HOME_WIN", "DRAW", "AWAY_WIN"];
  if (!allowedPredictions.includes(payload.analysis.prediction)) {
    fail("analysis.prediction geçersiz sonuç kodu", "analysis.analysis.prediction");
  }
  const probability = finiteNumber(payload.analysis.probability, "analysis.analysis.probability");
  if (probability < 0 || probability > 100) {
    fail("analysis.probability 0-100 aralığında olmalı", "analysis.analysis.probability");
  }
  for (const outcome of allowedPredictions) {
    finiteNumber(payload.analysis.all_probabilities?.[outcome], `analysis.all_probabilities.${outcome}`);
  }
  return payload;
}

export function validateBacktestResponse(payload) {
  if (!isRecord(payload)) {
    fail("Geriye dönük test yanıtı geçerli bir nesne değil", "backtest");
  }
  if (!Number.isInteger(payload.total_bets) || payload.total_bets < 0) {
    fail("backtest.total_bets negatif olmayan tamsayı olmalı", "backtest.total_bets");
  }
  if (!Array.isArray(payload.bankroll_history)) {
    fail("backtest.bankroll_history dizi olmalı", "backtest.bankroll_history");
  }
  for (const value of payload.bankroll_history) {
    finiteNumber(value, "backtest.bankroll_history.item");
  }
  return payload;
}