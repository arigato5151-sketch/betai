from __future__ import annotations

import httpx
import pytest

from app.providers.wikidata import WikidataTeamLocationClient


def _coordinate_entity(latitude: float, longitude: float, label: str) -> dict:
    return {
        "labels": {"en": {"value": label}},
        "claims": {
            "P625": [
                {
                    "mainsnak": {
                        "datavalue": {
                            "value": {
                                "latitude": latitude,
                                "longitude": longitude,
                            }
                        }
                    }
                }
            ]
        },
    }


@pytest.mark.asyncio
async def test_resolves_exact_mens_club_home_venue_coordinate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        action = request.url.params["action"]
        if action == "wbsearchentities":
            return httpx.Response(
                200,
                json={
                    "search": [
                        {
                            "id": "Q1",
                            "label": "Liverpool F.C. Women",
                            "description": "women's association football club",
                        },
                        {
                            "id": "Q2",
                            "label": "Liverpool F.C.",
                            "description": (
                                "association football club in Liverpool, England"
                            ),
                        },
                    ]
                },
            )
        ids = request.url.params["ids"]
        if ids == "Q2":
            return httpx.Response(
                200,
                json={
                    "entities": {
                        "Q2": {
                            "claims": {
                                "P115": [
                                    {
                                        "rank": "preferred",
                                        "mainsnak": {
                                            "datavalue": {"value": {"id": "Q3"}}
                                        },
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={"entities": {"Q3": _coordinate_entity(53.4308, -2.9608, "Anfield")}},
        )

    client = WikidataTeamLocationClient(
        transport=httpx.MockTransport(handler),
        timeout_seconds=2,
        request_interval_seconds=0,
    )

    result = await client.resolve(team_name="Liverpool", country="England")

    assert result is not None
    assert result.club_qid == "Q2"
    assert result.location_qid == "Q3"
    assert result.location_name == "Anfield"
    assert result.method == "home_venue"
    assert result.latitude == pytest.approx(53.4308)
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_rejects_non_club_search_results_without_followup_request() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "search": [
                    {
                        "id": "Q1",
                        "label": "Galatasaray",
                        "description": "human settlement",
                    }
                ]
            },
        )

    client = WikidataTeamLocationClient(
        transport=httpx.MockTransport(handler),
        timeout_seconds=2,
        request_interval_seconds=0,
    )

    assert await client.resolve(team_name="Galatasaray") is None
    assert requests == 1
