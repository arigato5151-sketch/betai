import bcrypt
import pytest

from app.core.passwords import hash_password, verify_password


LEGACY_PASSLIB_PBKDF2_HASH = (
    "$pbkdf2-sha256$29000$bGVnYWN5LXNhbHQtMTIzNA$"
    ".aJE6PUYheStNHYEMAiWFiaaInOgxwKFVU65YqyVkL0"
)


def test_new_password_hash_uses_direct_bcrypt() -> None:
    password_hash = hash_password("correct-horse-battery-staple")

    assert password_hash.startswith("$2b$")
    assert verify_password("correct-horse-battery-staple", password_hash) is True
    assert verify_password("wrong-password", password_hash) is False


def test_passlib_bcrypt_hash_remains_compatible() -> None:
    legacy_hash = bcrypt.hashpw(
        b"legacy-password",
        bcrypt.gensalt(rounds=4, prefix=b"2b"),
    ).decode("ascii")

    assert verify_password("legacy-password", legacy_hash) is True
    assert verify_password("wrong-password", legacy_hash) is False


def test_existing_passlib_pbkdf2_hash_remains_compatible() -> None:
    assert verify_password("legacy-password", LEGACY_PASSLIB_PBKDF2_HASH) is True
    assert verify_password("wrong-password", LEGACY_PASSLIB_PBKDF2_HASH) is False


@pytest.mark.parametrize(
    "password_hash",
    [
        "not-a-password-hash",
        "$2b$invalid",
        "$pbkdf2-sha256$invalid$salt$checksum",
        "$pbkdf2-sha256$999999999$salt$checksum",
    ],
)
def test_malformed_or_unsafe_hash_is_rejected(password_hash: str) -> None:
    assert verify_password("password", password_hash) is False


def test_passwords_over_bcrypt_byte_limit_are_rejected() -> None:
    with pytest.raises(ValueError, match="72-byte"):
        hash_password("x" * 73)

    assert verify_password("x" * 73, hash_password("short-password")) is False
