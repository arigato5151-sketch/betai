export const hasPermission = (user, permission) =>
  Array.isArray(user?.permissions) && user.permissions.includes(permission);

export const allowedActions = (user) => ({
  analyze: hasPermission(user, "analysis:create"),
  readHistory: hasPermission(user, "history:read"),
  updateResult: hasPermission(user, "history:update_result"),
  runBacktest: hasPermission(user, "backtest:run"),
  readAudit: hasPermission(user, "audit:read"),
  manageUsers: hasPermission(user, "users:manage"),
  manageRoles: hasPermission(user, "roles:manage"),
});
