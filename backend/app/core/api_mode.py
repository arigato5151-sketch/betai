from typing import Literal

ApiMode = Literal["demo", "live"]

_DEMO_API_KEYS = frozenset(
    {
        "",
        "DEMO_KEY",
        "DEMO_KEY_BURAYA",
        "your_api_key_here",
        "your_api_football_key",
    }
)


def get_api_mode(api_key: str | None) -> ApiMode:
    """Classify known empty/placeholder keys without exposing the key itself."""
    normalized_key = (api_key or "").strip()
    return "demo" if normalized_key in _DEMO_API_KEYS else "live"
