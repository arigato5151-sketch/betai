export const buildBankrollSeries = (history = []) => {
  const values = history
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map((value) => Number(value))
    .filter(Number.isFinite);
  if (values.length === 0) {
    return { labels: [], values: [], change: 0, min: 0, max: 0 };
  }

  return {
    labels: values.map((_, index) =>
      index === 0 ? "Başlangıç" : `Bahis ${index}`,
    ),
    values,
    change: Number((values.at(-1) - values[0]).toFixed(2)),
    min: Math.min(...values),
    max: Math.max(...values),
  };
};
