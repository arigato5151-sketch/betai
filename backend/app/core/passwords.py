from __future__ import annotations

import base64
import hashlib
import hmac

import bcrypt

BCRYPT_ROUNDS = 12
BCRYPT_MAX_PASSWORD_BYTES = 72
_LEGACY_PBKDF2_PREFIX = "$pbkdf2-sha256$"
_MAX_LEGACY_PBKDF2_ROUNDS = 1_000_000


def _password_bytes(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password exceeds bcrypt's {BCRYPT_MAX_PASSWORD_BYTES}-byte limit"
        )
    return encoded


def hash_password(password: str) -> str:
    password_bytes = _password_bytes(password)
    return bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt(rounds=BCRYPT_ROUNDS, prefix=b"2b"),
    ).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        password_bytes = _password_bytes(password)
    except (UnicodeEncodeError, ValueError):
        return False

    if password_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(password_bytes, password_hash.encode("ascii"))
        except (UnicodeEncodeError, ValueError):
            return False

    if password_hash.startswith(_LEGACY_PBKDF2_PREFIX):
        return _verify_legacy_passlib_pbkdf2(password_bytes, password_hash)
    return False


def _verify_legacy_passlib_pbkdf2(
    password_bytes: bytes,
    password_hash: str,
) -> bool:
    """Verify hashes emitted by the previous passlib pbkdf2_sha256 context."""
    try:
        _, scheme, rounds_value, salt_value, checksum_value = password_hash.split("$")
        if scheme != "pbkdf2-sha256":
            return False
        rounds = int(rounds_value)
        if rounds < 1 or rounds > _MAX_LEGACY_PBKDF2_ROUNDS:
            return False
        salt = _decode_passlib_base64(salt_value)
        expected = _decode_passlib_base64(checksum_value)
    except (TypeError, ValueError):
        return False

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password_bytes,
        salt,
        rounds,
        dklen=len(expected),
    )
    return hmac.compare_digest(derived, expected)


def _decode_passlib_base64(value: str) -> bytes:
    adapted = value.replace(".", "+")
    padding = "=" * (-len(adapted) % 4)
    return base64.b64decode(adapted + padding, validate=True)
