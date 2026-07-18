import assert from "node:assert/strict";
import test from "node:test";

import { buildHistoryQuery, filterAndSortHistory } from "./history.js";

const rows = [
  {
    id: 1,
    home_team: "Fenerbahçe",
    away_team: "Galatasaray",
    actual_result: null,
    is_value_bet: 1,
    edge: 8.2,
    odd: 2.1,
    created_at: "2026-01-01T10:00:00Z",
  },
  {
    id: 2,
    home_team: "Beşiktaş",
    away_team: "Trabzonspor",
    actual_result: "HOME_WIN",
    is_value_bet: 0,
    edge: 1.2,
    odd: 3.4,
    created_at: "2026-01-02T10:00:00Z",
  },
];

test("filters by Turkish team query without mutating input", () => {
  const originalOrder = rows.map((row) => row.id);
  const result = filterAndSortHistory(rows, { query: "FENERBAHÇE" });

  assert.deepEqual(result.map((row) => row.id), [1]);
  assert.deepEqual(rows.map((row) => row.id), originalOrder);
});

test("combines result and value filters", () => {
  assert.deepEqual(
    filterAndSortHistory(rows, { result: "pending", value: "value" }).map(
      (row) => row.id,
    ),
    [1],
  );
});

test("supports deterministic sorting modes", () => {
  assert.deepEqual(filterAndSortHistory(rows).map((row) => row.id), [2, 1]);
  assert.deepEqual(
    filterAndSortHistory(rows, { sort: "edge" }).map((row) => row.id),
    [1, 2],
  );
  assert.deepEqual(
    filterAndSortHistory(rows, { sort: "odd" }).map((row) => row.id),
    [2, 1],
  );
});

test("builds encoded paginated history query", () => {
  const query = buildHistoryQuery(
    { query: " Fener & Gala ", result: "pending", value: "value", sort: "edge" },
    2,
    25,
  );
  const params = new URLSearchParams(query);

  assert.equal(params.get("paginated"), "true");
  assert.equal(params.get("page"), "2");
  assert.equal(params.get("page_size"), "25");
  assert.equal(params.get("query"), "Fener & Gala");
  assert.equal(params.get("result"), "pending");
  assert.equal(params.get("value"), "value");
  assert.equal(params.get("sort"), "edge");
});
