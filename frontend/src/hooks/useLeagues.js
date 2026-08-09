import { useEffect, useState } from "react";

import { validateLeaguesResponse } from "../apiSchema.js";
import { useRequestGate } from "./useRequestGate.js";

export function normalizeLeagues(payload) {
  const validated = validateLeaguesResponse(payload);
  const uniqueLeagues = new Map();
  validated.forEach((league) => {
    uniqueLeagues.set(league.id, {
      ...league,
      name: league.name.trim(),
    });
  });
  return [...uniqueLeagues.values()];
}

export function useLeagues(request) {
  const { stale, begin, settle } = useRequestGate();
  const [leagues, setLeagues] = useState([]);
  const [leaguesLoading, setLeaguesLoading] = useState(true);
  const [leaguesError, setLeaguesError] = useState("");

  useEffect(() => {
    let disposed = false;
    const mine = begin();
    setLeaguesLoading(true);
    setLeaguesError("");

    const loadLeagues = async () => {
      try {
        const response = await request("/leagues");
        if (!response.ok) {
          throw new Error("Lig listesi isteği başarısız oldu.");
        }
        const loadedLeagues = normalizeLeagues(await response.json());
        settle(mine, () => {
          if (!disposed) setLeagues(loadedLeagues);
        });
      } catch {
        settle(mine, () => {
          if (!disposed) {
            setLeagues([]);
            setLeaguesError(
              "Desteklenen ligler alınamadı. Lig seçmeden manuel analize devam edebilirsiniz.",
            );
          }
        });
      } finally {
        if (!disposed) setLeaguesLoading(false);
      }
    };

    loadLeagues();
    return () => {
      disposed = true;
    };
  }, [request]);

  return {
    leagues,
    leaguesEmpty: !leaguesLoading && !leaguesError && leagues.length === 0,
    leaguesError,
    leaguesLoading,
    leaguesStale: stale,
  };
}