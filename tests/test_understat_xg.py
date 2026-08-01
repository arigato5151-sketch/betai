from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.team_identity import normalize_team_name
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import Base, HistoricalFixture
from app.providers.understat import (
    UnderstatClient,
    UnderstatFixtureXG,
    UnderstatFormatError,
)
from app.services.understat_xg import match_understat_xg
from app.tasks.jobs import sync_understat_xg_task


def historical_fixture(
    fixture_id: int = 1,
    *,
    home_team: str = "Inter Milan",
    away_team: str = "Juventus",
    kickoff: datetime | None = None,
) -> HistoricalFixture:
    return HistoricalFixture(
        fixture_id=fixture_id,
        league_id=135,
        season=2025,
        kickoff=kickoff or datetime(2025, 10, 25, 17, 0, tzinfo=UTC),
        home_team_id=10,
        away_team_id=20,
        home_team=home_team,
        away_team=away_team,
        home_goals=2,
        away_goals=1,
        actual_result="HOME_WIN",
        status="FT",
        data_source="football_data_csv",
    )


def provider_fixture(
    *,
    match_id: str = "12345",
    home_team: str = "Inter",
    away_team: str = "Juventus",
    kickoff: datetime | None = None,
) -> UnderstatFixtureXG:
    return UnderstatFixtureXG(
        provider_match_id=match_id,
        kickoff=kickoff or datetime(2025, 10, 25, 17, 5, tzinfo=UTC),
        home_team=home_team,
        away_team=away_team,
        home_goals=2,
        away_goals=1,
        home_xg=1.75,
        away_xg=0.82,
    )


@pytest.mark.asyncio
async def test_understat_client_sends_xhr_headers_and_normalizes_xg() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/getLeagueData/EPL/2025"
        assert request.headers["x-requested-with"] == "XMLHttpRequest"
        assert request.headers["referer"].endswith("/league/EPL/2025")
        return httpx.Response(
            200,
            json={
                "dates": [
                    {
                        "id": "28778",
                        "isResult": True,
                        "h": {"id": "87", "title": "Liverpool"},
                        "a": {"id": "73", "title": "Bournemouth"},
                        "goals": {"h": "4", "a": "2"},
                        "xG": {"h": "2.33007", "a": "1.57303"},
                        "datetime": "2025-08-15 19:00:00",
                    },
                    {"id": "future", "isResult": False},
                ]
            },
        )

    rows = await UnderstatClient(
        base_url="https://understat.com",
        transport=httpx.MockTransport(handler),
    ).get_completed_fixture_xg(39, 2025)

    assert rows == [
        UnderstatFixtureXG(
            provider_match_id="28778",
            kickoff=datetime(2025, 8, 15, 19, 0, tzinfo=UTC),
            home_team="Liverpool",
            away_team="Bournemouth",
            home_goals=4,
            away_goals=2,
            home_xg=2.33007,
            away_xg=1.57303,
        )
    ]


@pytest.mark.asyncio
async def test_understat_client_rejects_invalid_xg() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "dates": [
                    {
                        "id": "1",
                        "isResult": True,
                        "h": {"title": "Home"},
                        "a": {"title": "Away"},
                        "goals": {"h": "1", "a": "0"},
                        "xG": {"h": "nan", "a": "0.5"},
                        "datetime": "2025-08-15 19:00:00",
                    }
                ]
            },
        )

    client = UnderstatClient(transport=httpx.MockTransport(handler))
    with pytest.raises(UnderstatFormatError, match="row 1"):
        await client.get_completed_fixture_xg(39, 2025)


@pytest.mark.asyncio
async def test_understat_client_rejects_invalid_json_without_retrying() -> None:
    requests = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, text="not-json")

    client = UnderstatClient(transport=httpx.MockTransport(handler))
    with pytest.raises(UnderstatFormatError, match="valid JSON"):
        await client.get_completed_fixture_xg(39, 2025)
    assert requests == 1


def test_xg_matching_requires_team_score_and_close_kickoff() -> None:
    historical = historical_fixture()
    matched = match_understat_xg(
        [historical],
        [provider_fixture()],
        tolerance_hours=12,
    )

    assert matched.unmatched_provider_ids == ()
    assert matched.ambiguous_provider_ids == ()
    assert matched.updates == (
        {
            "fixture_id": 1,
            "home_xg": 1.75,
            "away_xg": 0.82,
            "xg_source": "understat",
            "xg_provider_match_id": "12345",
        },
    )

    outside_window = provider_fixture(
        match_id="late",
        kickoff=historical.kickoff + timedelta(hours=13),
    )
    unmatched = match_understat_xg([historical], [outside_window], tolerance_hours=12)
    assert unmatched.unmatched_provider_ids == ("late",)


@pytest.mark.parametrize(
    ("provider_name", "football_data_name"),
    [
        ("Newcastle United", "Newcastle"),
        ("Borussia Dortmund", "Dortmund"),
        ("Eintracht Frankfurt", "Ein Frankfurt"),
        ("Borussia M.Gladbach", "M'gladbach"),
        ("RasenBallsport Leipzig", "RB Leipzig"),
        ("AC Milan", "Milan"),
        ("Parma Calcio 1913", "Parma"),
        ("Athletic Club", "Ath Bilbao"),
        ("Real Sociedad", "Sociedad"),
    ],
)
def test_verified_provider_team_aliases_match(
    provider_name: str, football_data_name: str
) -> None:
    assert normalize_team_name(provider_name) == normalize_team_name(football_data_name)


def test_xg_matching_rejects_equal_distance_candidates() -> None:
    kickoff = datetime(2025, 10, 25, 17, 0, tzinfo=UTC)
    first = historical_fixture(fixture_id=1, kickoff=kickoff - timedelta(hours=1))
    second = historical_fixture(fixture_id=2, kickoff=kickoff + timedelta(hours=1))
    observation = provider_fixture(kickoff=kickoff)

    result = match_understat_xg([first, second], [observation], tolerance_hours=48)

    assert result.updates == ()
    assert result.ambiguous_provider_ids == ("12345",)


def test_repository_persists_xg_with_provenance() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(historical_fixture())
        session.commit()
        repository = HistoricalFixtureRepository(session)

        assert (
            repository.update_xg_many(
                match_understat_xg(
                    repository.get_all(), [provider_fixture()], tolerance_hours=12
                ).updates
            )
            == 1
        )

        stored = repository.get_by_fixture_id(1)
        assert stored is not None
        assert stored.home_xg == 1.75
        assert stored.away_xg == 0.82
        assert stored.xg_source == "understat"
        assert stored.xg_provider_match_id == "12345"
        assert stored.xg_updated_at is not None


def test_understat_sync_task_updates_matching_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tasks import jobs

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    fixture = historical_fixture()
    fixture.league_id = 39
    with Session(engine) as session:
        session.add(fixture)
        session.commit()

    class FakeClient:
        supported_league_ids = frozenset({39})

        async def get_completed_fixture_xg(
            self, league_id: int, season: int
        ) -> list[UnderstatFixtureXG]:
            assert (league_id, season) == (39, 2025)
            return [provider_fixture()]

    monkeypatch.setattr(settings, "UNDERSTAT_ENABLED", True)
    monkeypatch.setattr(jobs, "UnderstatClient", FakeClient)
    monkeypatch.setattr(jobs, "SessionLocal", lambda: Session(engine))

    result = sync_understat_xg_task.run([2025])

    assert result["status"] == "completed"
    assert result["fixtures_fetched"] == 1
    assert result["fixtures_updated"] == 1
    assert result["unmatched_fixtures"] == 0
    assert result["failed_league_seasons"] == []
