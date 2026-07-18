const normalizeText = (value) => String(value ?? "").toLocaleLowerCase("tr-TR");

const timestamp = (value) => {
  const parsed = Date.parse(value ?? "");
  return Number.isNaN(parsed) ? 0 : parsed;
};

export const filterAndSortHistory = (
  items,
  { query = "", result = "all", value = "all", sort = "newest" } = {},
) => {
  const normalizedQuery = normalizeText(query).trim();
  const filtered = items.filter((item) => {
    const teams = normalizeText(`${item.home_team ?? ""} ${item.away_team ?? ""}`);
    const matchesQuery = !normalizedQuery || teams.includes(normalizedQuery);
    const matchesResult =
      result === "all" ||
      (result === "pending" ? !item.actual_result : item.actual_result === result);
    const isValueBet = item.is_value_bet === 1 || item.is_value_bet === true;
    const matchesValue =
      value === "all" || (value === "value" ? isValueBet : !isValueBet);
    return matchesQuery && matchesResult && matchesValue;
  });

  return [...filtered].sort((left, right) => {
    if (sort === "oldest") return timestamp(left.created_at) - timestamp(right.created_at);
    if (sort === "edge") return Number(right.edge ?? 0) - Number(left.edge ?? 0);
    if (sort === "odd") return Number(right.odd ?? 0) - Number(left.odd ?? 0);
    const dateDifference = timestamp(right.created_at) - timestamp(left.created_at);
    return dateDifference || Number(right.id ?? 0) - Number(left.id ?? 0);
  });
};

export const buildHistoryQuery = (
  { query = "", result = "all", value = "all", sort = "newest" } = {},
  page = 1,
  pageSize = 15,
) => {
  const params = new URLSearchParams({
    paginated: "true",
    page: String(page),
    page_size: String(pageSize),
    result,
    value,
    sort,
  });
  const normalizedQuery = query.trim();
  if (normalizedQuery) params.set("query", normalizedQuery);
  return `?${params.toString()}`;
};
