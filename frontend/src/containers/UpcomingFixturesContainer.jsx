import { useCallback, useEffect, useState } from "react";

import UpcomingFixtures from "../components/UpcomingFixtures.jsx";

export function normalizeUpcomingFixtures(payload) {
  if (!Array.isArray(payload)) {
    throw new TypeError("Fikstür yanıtı geçerli bir dizi değil.");
  }

  return payload
    .filter((fixture) => {
      const kickoff = Date.parse(fixture?.kickoff);
      return (
        Number.isInteger(fixture?.fixture_id) &&
        fixture.fixture_id > 0 &&
        typeof fixture.home_team === "string" &&
        typeof fixture.away_team === "string" &&
        Number.isFinite(kickoff)
      );
    })
    .map((fixture) => ({
      ...fixture,
      home_team: fixture.home_team.trim() || "Ev Sahibi",
      away_team: fixture.away_team.trim() || "Deplasman",
    }))
    .sort((left, right) => {
      const kickoffDifference =
        Date.parse(left.kickoff) - Date.parse(right.kickoff);
      if (kickoffDifference !== 0) return kickoffDifference;
      return left.fixture_id - right.fixture_id;
    });
}

function UpcomingFixturesContainer({
  onSelectFixture,
  request,
  selectedFixtureId,
}) {
  const [fixtures, setFixtures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchFixtures = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await request("/fixtures/upcoming?days=7&limit=100");
      if (!response.ok) {
        throw new Error("Haftalık fikstür alınamadı.");
      }
      setFixtures(normalizeUpcomingFixtures(await response.json()));
    } catch (requestError) {
      setError(requestError.message || "Haftalık fikstür alınamadı.");
    } finally {
      setLoading(false);
    }
  }, [request]);

  useEffect(() => {
    fetchFixtures();
  }, [fetchFixtures]);

  return (
    <UpcomingFixtures
      error={error}
      fixtures={fixtures}
      loading={loading}
      onRefresh={fetchFixtures}
      onSelectFixture={onSelectFixture}
      selectedFixtureId={selectedFixtureId}
    />
  );
}

export default UpcomingFixturesContainer;
