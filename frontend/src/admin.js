export const toggleRoleSelection = (selectedRoles, roleName, checked) => {
  const normalized = [...new Set(selectedRoles)];
  if (checked) return [...new Set([...normalized, roleName])].sort();
  if (normalized.length <= 1) return normalized;
  return normalized.filter((role) => role !== roleName);
};

export const responseErrorMessage = async (response, fallback) => {
  try {
    const payload = await response.json();
    return typeof payload?.detail === "string" ? payload.detail : fallback;
  } catch {
    return fallback;
  }
};
