import assert from "node:assert/strict";
import test from "node:test";

import { allowedActions, hasPermission } from "./permissions.js";

test("maps analyst permissions to visible actions", () => {
  const user = {
    permissions: [
      "analysis:create",
      "history:read",
      "history:update_result",
      "backtest:run",
      "audit:read",
    ],
  };
  assert.equal(hasPermission(user, "analysis:create"), true);
  assert.deepEqual(allowedActions(user), {
    analyze: true,
    readHistory: true,
    updateResult: true,
    runBacktest: true,
    readAudit: true,
    manageUsers: false,
    manageRoles: false,
  });
});

test("viewer cannot see mutating or backtest actions", () => {
  const actions = allowedActions({ permissions: ["history:read", "audit:read"] });
  assert.equal(actions.readHistory, true);
  assert.equal(actions.readAudit, true);
  assert.equal(actions.analyze, false);
  assert.equal(actions.updateResult, false);
  assert.equal(actions.runBacktest, false);
  assert.equal(actions.manageUsers, false);
  assert.equal(actions.manageRoles, false);
});

test("admin management permissions expose the RBAC panel", () => {
  const actions = allowedActions({ permissions: ["users:manage", "roles:manage"] });
  assert.equal(actions.manageUsers, true);
  assert.equal(actions.manageRoles, true);
});

test("missing permission list is denied by default", () => {
  assert.equal(hasPermission(null, "analysis:create"), false);
  assert.equal(hasPermission({}, "analysis:create"), false);
});
