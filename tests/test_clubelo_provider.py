from datetime import UTC, date, datetime

import httpx
import pytest

from app.providers.clubelo import ClubEloClient, ClubEloFormatError


def _csv(*rows: str) -> str:
    return "Rank,Club,Country,Level,Elo,From,To\n" + "\n".join(rows) + "\n"


@pytest.mark.asyncio
async def test_clubelo_resolves_only_one_exact_normalized_name() -> None:
    expected_date = min(date(2026, 7, 30), datetime.now(UTC).date())

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/{expected_date.isoformat()}"
        return httpx.Response(
            200,
            text=_csv(
                "1,Fenerbahce,TUR,1,1742.25,2026-07-29,2026-08-04",
                "2,Galatasaray,TUR,0,1731.10,2026-07-29,2026-08-04",
            ),
        )

    client = ClubEloClient(
        base_url="http://api.clubelo.com",
        confidence=0.8,
        transport=httpx.MockTransport(handler),
    )

    resolved = await client.resolve_elo(
        team_name="Fenerbahçe",
        as_of=datetime(2026, 7, 30, 18, tzinfo=UTC),
    )

    assert resolved is not None
    candidate, point = resolved
    assert candidate.provider_team_key == "Fenerbahce"
    assert point.value == pytest.approx(1742.25)
    assert point.confidence == pytest.approx(0.8)
    assert point.is_fallback is True
    assert point.details["effective_date"] == expected_date.isoformat()


@pytest.mark.asyncio
async def test_clubelo_rejects_ambiguous_normalized_identity() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_csv(
                "1,Team-A,ENG,1,1600,2026-07-29,2026-08-04",
                "2,Team A,ENG,2,1500,2026-07-29,2026-08-04",
            ),
        )

    client = ClubEloClient(
        base_url="http://api.clubelo.com",
        transport=httpx.MockTransport(handler),
    )

    assert (
        await client.resolve_elo(
            team_name="Team A",
            as_of=datetime(2026, 7, 30, tzinfo=UTC),
        )
        is None
    )


@pytest.mark.asyncio
async def test_clubelo_fails_closed_on_out_of_range_rating() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_csv("1,Unsafe FC,ENG,1,99999,2026-07-29,2026-08-04"),
        )

    client = ClubEloClient(
        base_url="http://api.clubelo.com",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ClubEloFormatError, match="out-of-range Elo"):
        await client.resolve_elo(
            team_name="Unsafe FC",
            as_of=datetime(2026, 7, 30, tzinfo=UTC),
        )
