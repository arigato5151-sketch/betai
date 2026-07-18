import assert from "node:assert/strict";
import test from "node:test";

import { buildBankrollSeries } from "./bankroll.js";

test("builds labeled bankroll series and summary", () => {
  const result = buildBankrollSeries([1000, 1015.25, 990, 1042.75]);

  assert.deepEqual(result.labels, ["Baslangic", "Bahis 1", "Bahis 2", "Bahis 3"]);
  assert.deepEqual(result.values, [1000, 1015.25, 990, 1042.75]);
  assert.equal(result.change, 42.75);
  assert.equal(result.min, 990);
  assert.equal(result.max, 1042.75);
});

test("handles empty and invalid histories safely", () => {
  assert.deepEqual(buildBankrollSeries([]), {
    labels: [],
    values: [],
    change: 0,
    min: 0,
    max: 0,
  });
  assert.deepEqual(buildBankrollSeries(["invalid", null]).values, []);
});
