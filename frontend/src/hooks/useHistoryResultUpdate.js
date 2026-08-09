import { useState } from "react";

import { useRequestGate } from "./useRequestGate.js";

export function useHistoryResultUpdate({ actions, onHistoryChanged, request }) {
  const { stale, begin, settle } = useRequestGate();
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState("");

  const submitActualResult = async (recordId, result) => {
    if (!actions.updateResult) return;
    const mine = begin();
    setUpdating(true);
    setError("");
    try {
      const response = await request(`/history/${recordId}/result`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actual_result: result }),
      });
      if (!response.ok) throw new Error("Sonuç kaydedilemedi.");
      await response.json();
      settle(mine, () => onHistoryChanged?.());
    } catch (submitError) {
      settle(mine, () => {
        setError(submitError.message || "Sonuç kaydedilemedi.");
      });
    } finally {
      setUpdating(false);
    }
  };

  return {
    resultError: error,
    resultStale: stale,
    resultUpdating: updating,
    submitActualResult,
  };
}