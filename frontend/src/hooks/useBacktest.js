import { useState } from "react";

import { validateBacktestResponse } from "../apiSchema.js";
import { useRequestGate } from "./useRequestGate.js";

export function useBacktest({ actions, request }) {
  const { stale, begin, settle } = useRequestGate();
  const [backtest, setBacktest] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runBacktest = async () => {
    if (!actions.runBacktest) return;
    const mine = begin();
    setLoading(true);
    setError("");
    try {
      const response = await request("/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_bankroll: 1000,
          strategy: "fractional_kelly",
          kelly_fraction: 0.25,
          min_edge_pct: 3,
          commission_pct: 2,
          max_stake_pct: 5,
          max_daily_exposure_pct: 15,
          require_closing_odds: true,
          exclude_post_kickoff: true,
        }),
      });
      if (!response.ok) throw new Error("Geriye dönük test çalıştırılamadı.");
      const data = validateBacktestResponse(await response.json());
      settle(mine, () => setBacktest(data));
    } catch (runError) {
      settle(mine, () => {
        setError(runError.message || "Geriye dönük test çalıştırılamadı.");
      });
    } finally {
      setLoading(false);
    }
  };

  return {
    backtest,
    backtestEmpty: backtest === null,
    backtestError: error,
    backtestLoading: loading,
    backtestStale: stale,
    runBacktest,
  };
}