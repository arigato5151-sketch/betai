import pytest

from app.providers.geonames_city import GeoNamesCityResolver


@pytest.fixture(scope="module")
def resolver() -> GeoNamesCityResolver:
    return GeoNamesCityResolver()


def test_resolver_matches_city_within_requested_country(
    resolver: GeoNamesCityResolver,
) -> None:
    amsterdam = resolver.resolve(city="Amsterdam", country="Netherlands")

    assert amsterdam is not None
    assert amsterdam.country_code == "NL"
    assert amsterdam.latitude == pytest.approx(52.37403)
    assert amsterdam.longitude == pytest.approx(4.88969)
    assert amsterdam.confidence == pytest.approx(0.75)


def test_resolver_does_not_mix_same_named_cities_across_countries(
    resolver: GeoNamesCityResolver,
) -> None:
    amsterdam = resolver.resolve(city="Amsterdam", country="USA")

    assert amsterdam is not None
    assert amsterdam.country_code == "US"
    assert amsterdam.latitude == pytest.approx(42.93869)
    assert amsterdam.longitude == pytest.approx(-74.18819)


def test_resolver_supports_audited_provider_city_aliases(
    resolver: GeoNamesCityResolver,
) -> None:
    moscow = resolver.resolve(city="Moskva", country="Russia")

    assert moscow is not None
    assert moscow.city == "Moscow"
    assert moscow.country_code == "RU"
    assert moscow.latitude == pytest.approx(55.75222)
    assert moscow.longitude == pytest.approx(37.61556)


@pytest.mark.parametrize(
    ("city", "country"),
    [
        ("", "Turkey"),
        ("Istanbul", ""),
        ("A city that cannot exist", "Turkey"),
        ("Istanbul", "A country that cannot exist"),
    ],
)
def test_resolver_returns_none_for_incomplete_or_unknown_inputs(
    resolver: GeoNamesCityResolver,
    city: str,
    country: str,
) -> None:
    assert resolver.resolve(city=city, country=country) is None
