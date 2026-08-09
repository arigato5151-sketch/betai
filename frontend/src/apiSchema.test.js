import assert from "node:assert/strict";
import test from "node:test";

import {
  DataValidationError,
  validateAnalysisResponse,
  validateBacktestResponse,
  validateLeaguesResponse,
  validatePrefillResponse,
} from "./apiSchema.js";

const assertThrowsValidation = (callback) =>
  assert.throws(callback, (error) => error instanceof DataValidationError);

test("lig yanıtını doğrular ve aynı kimliğe sahip kopyaları eler", () => {
  assert.deepEqual(
    validateLeaguesResponse([
      { id: 2, name: "UEFA Champions League" },
      { id: 2, name: "Yinelenen Lig" },
    ]),
    [{ id: 2, name: "Yinelenen Lig" }],
  );
  assertThrowsValidation(() => validateLeaguesResponse({}));
  assertThrowsValidation(() =>
    validateLeaguesResponse([{ id: 0, name: "Geçersiz" }]),
  );
  assertThrowsValidation(() =>
    validateLeaguesResponse([{ id: 1, name: "  " }]),
  );
});

test("ön dolum yanıtını zorunlu alanlarla doğrular", () => {
  assert.doesNotThrow(() =>
    validatePrefillResponse({
      home_team: "A",
      away_team: "B",
      odd: 2.1,
    }),
  );
  assertThrowsValidation(() => validatePrefillResponse({ home_team: "A" }));
  assertThrowsValidation(() =>
    validatePrefillResponse({ home_team: "A", away_team: "B", odd: 1 }),
  );
  assertThrowsValidation(() => validatePrefillResponse([]));
});

test("analiz yanıtını sonuç ve olasılık şemasıyla doğrular", () => {
  assert.doesNotThrow(() =>
    validateAnalysisResponse({
      match: "A – B",
      analysis: {
        prediction: "HOME_WIN",
        probability: 55,
        all_probabilities: { HOME_WIN: 55, DRAW: 25, AWAY_WIN: 20 },
      },
    }),
  );
  assertThrowsValidation(() =>
    validateAnalysisResponse({ match: "A – B", analysis: {} }),
  );
  assertThrowsValidation(() =>
    validateAnalysisResponse({
      match: "A – B",
      analysis: {
        prediction: "BANKRUPT",
        probability: 55,
        all_probabilities: { HOME_WIN: 55, DRAW: 25, AWAY_WIN: 20 },
      },
    }),
  );
  assertThrowsValidation(() =>
    validateAnalysisResponse({
      match: "A – B",
      analysis: {
        prediction: "DRAW",
        probability: 120,
        all_probabilities: { HOME_WIN: 55, DRAW: 25, AWAY_WIN: 20 },
      },
    }),
  );
  assertThrowsValidation(() =>
    validateAnalysisResponse({
      match: "A – B",
      analysis: { prediction: "DRAW", probability: 30, all_probabilities: {} },
    }),
  );
});

test("geriye dönük test yanıtını kasa geçmişiyle doğrular", () => {
  assert.doesNotThrow(() =>
    validateBacktestResponse({ total_bets: 0, bankroll_history: [1000] }),
  );
  assertThrowsValidation(() =>
    validateBacktestResponse({ total_bets: -1, bankroll_history: [] }),
  );
  assertThrowsValidation(() =>
    validateBacktestResponse({ total_bets: 1, bankroll_history: [NaN] }),
  );
  assertThrowsValidation(() => validateBacktestResponse([1000]));
});