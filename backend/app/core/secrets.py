from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import hvac
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import dotenv_values

VAULT_SECRET_KEYS = frozenset(
    {
        "API_FOOTBALL_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "JWT_SECRET_KEY",
        "JWT_REFRESH_SECRET_KEY",
        "MODEL_SIGNING_KEY",
        "ADMIN_PASSWORD",
    }
)
REQUIRED_VAULT_SECRET_KEYS = frozenset(
    {
        "API_FOOTBALL_KEY",
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "JWT_REFRESH_SECRET_KEY",
        "MODEL_SIGNING_KEY",
        "ADMIN_PASSWORD",
    }
)


def _bootstrap_environment(env_file: Path) -> dict[str, str]:
    values = {
        key: value
        for key, value in dotenv_values(env_file).items()
        if value is not None
    }
    values.update(os.environ)
    return values


def load_external_secrets(
    env_file: Path,
    *,
    client_factory: Callable[..., Any] = hvac.Client,
    azure_credential_factory: Callable[..., Any] = DefaultAzureCredential,
    azure_client_factory: Callable[..., Any] = SecretClient,
) -> dict[str, Any]:
    """Load allowlisted secrets from Vault or Azure Key Vault."""
    config = _bootstrap_environment(env_file)
    provider = config.get("SECRET_PROVIDER", "env").strip().lower()
    if provider == "env":
        return {"provider": "env", "loaded_keys": 0}
    if provider == "azure_key_vault":
        vault_url = config.get("AZURE_KEY_VAULT_URL", "").strip()
        prefix = config.get("AZURE_KEY_VAULT_PREFIX", "bet-ai").strip().strip("-")
        if not vault_url.startswith("https://") or not prefix:
            raise RuntimeError(
                "Azure Key Vault HTTPS URL and secret prefix are required"
            )
        credential = azure_credential_factory()
        azure_client = azure_client_factory(vault_url=vault_url, credential=credential)
        loaded_keys = 0
        missing: list[str] = []
        for key in sorted(VAULT_SECRET_KEYS):
            secret_name = f"{prefix}-{key.lower().replace('_', '-')}"
            try:
                value = azure_client.get_secret(secret_name).value
            except Exception:
                if key in REQUIRED_VAULT_SECRET_KEYS:
                    missing.append(key)
                continue
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"Azure Key Vault secret {secret_name} is empty")
            os.environ[key] = value
            loaded_keys += 1
        if missing:
            raise RuntimeError(
                "Azure Key Vault is missing required keys: " + ", ".join(missing)
            )
        return {"provider": "azure_key_vault", "loaded_keys": loaded_keys}

    if provider != "vault":
        raise RuntimeError(f"Unsupported SECRET_PROVIDER: {provider!r}")

    address = config.get("VAULT_ADDR", "").strip()
    path = config.get("VAULT_SECRET_PATH", "bet-ai/production").strip()
    mount_point = config.get("VAULT_MOUNT_POINT", "secret").strip()
    if not address or not path or not mount_point:
        raise RuntimeError("Vault address, mount point and secret path are required")

    token = config.get("VAULT_TOKEN", "").strip()
    role_id = config.get("VAULT_ROLE_ID", "").strip()
    secret_id = config.get("VAULT_SECRET_ID", "").strip()
    if not token and not (role_id and secret_id):
        raise RuntimeError("Vault token or AppRole credentials are required")

    client = client_factory(url=address, token=token or None)
    if not token:
        client.auth.approle.login(role_id=role_id, secret_id=secret_id)
    if not client.is_authenticated():
        raise RuntimeError("Vault authentication failed")

    response = client.secrets.kv.v2.read_secret_version(
        path=path,
        mount_point=mount_point,
        raise_on_deleted_version=True,
    )
    payload: Mapping[str, object] = response.get("data", {}).get("data", {})
    missing = sorted(REQUIRED_VAULT_SECRET_KEYS - payload.keys())
    if missing:
        raise RuntimeError(
            f"Vault payload is missing required keys: {', '.join(missing)}"
        )

    loaded_keys = 0
    for key in VAULT_SECRET_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Vault secret {key} must be a non-empty string")
        os.environ[key] = value
        loaded_keys += 1

    return {"provider": "vault", "loaded_keys": loaded_keys}
