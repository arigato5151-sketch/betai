import assert from "node:assert/strict";
import test from "node:test";

import { normalizeApiMode } from "./apiMode.js";

test("normalizes supported API modes", () => {
  assert.equal(normalizeApiMode({ api_mode: "demo" }), "demo");
  assert.equal(normalizeApiMode({ api_mode: "live" }), "live");
});

test("rejects missing or unexpected API modes", () => {
  assert.equal(normalizeApiMode(), "unknown");
  assert.equal(normalizeApiMode({ api_mode: "staging" }), "unknown");
});
