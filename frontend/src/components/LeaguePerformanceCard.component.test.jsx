import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import LeaguePerformanceCard from "./LeaguePerformanceCard.jsx";

test("lig performans metriklerini ve örnek güvenini gösterir", () => {
  render(
    <LeaguePerformanceCard
      data={{
        leagues: [
          {
            league_id: 39,
            league_name: "Premier League",
            total_predictions: 42,
            win_rate_pct: 57.14,
            brier_score: 0.54,
            total_roi_pct: 8.2,
            avg_clv_pct: 2.1,
          },
          {
            league_id: 203,
            league_name: "Süper Lig",
            total_predictions: 4,
            win_rate_pct: 50,
            brier_score: null,
            total_roi_pct: -5,
            avg_clv_pct: null,
          },
        ],
      }}
      error=""
      loading={false}
      onRefresh={() => {}}
    />,
  );

  expect(screen.getByText("Premier League")).toBeInTheDocument();
  expect(screen.getByText("Süper Lig")).toBeInTheDocument();
  expect(screen.getByText("Yeterli")).toBeInTheDocument();
  expect(screen.getByText("Düşük örnek")).toBeInTheDocument();
});
