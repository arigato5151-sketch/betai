import assert from "node:assert/strict";
import test from "node:test";

import { responseErrorMessage, toggleRoleSelection } from "./admin.js";

test("toggles roles without allowing an empty role set", () => {
  assert.deepEqual(toggleRoleSelection(["viewer"], "admin", true), ["admin", "viewer"]);
  assert.deepEqual(toggleRoleSelection(["admin", "viewer"], "viewer", false), ["admin"]);
  assert.deepEqual(toggleRoleSelection(["viewer"], "viewer", false), ["viewer"]);
});

test("extracts API detail with a safe fallback", async () => {
  assert.equal(
    await responseErrorMessage({ json: async () => ({ detail: "Yetki yok" }) }, "Hata"),
    "Yetki yok",
  );
  assert.equal(
    await responseErrorMessage({ json: async () => { throw new Error("invalid"); } }, "Hata"),
    "Hata",
  );
});
