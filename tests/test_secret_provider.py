import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.secrets import (
    REQUIRED_VAULT_SECRET_KEYS,
    VAULT_SECRET_KEYS,
    load_external_secrets,
)


def configure_vault_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_PROVIDER", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
    monkeypatch.setenv("VAULT_TOKEN", "test-token")
    monkeypatch.setenv("VAULT_SECRET_PATH", "bet-ai/production")
    monkeypatch.setenv("VAULT_MOUNT_POINT", "secret")


def vault_client(payload: dict[str, str]) -> Mock:
    client = Mock()
    client.is_authenticated.return_value = True
    client.secrets.kv.v2.read_secret_version.return_value = {"data": {"data": payload}}
    return client


def test_env_provider_does_not_contact_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_PROVIDER", "env")
    factory = Mock()

    status = load_external_secrets(Path("missing.env"), client_factory=factory)

    assert status == {"provider": "env", "loaded_keys": 0}
    factory.assert_not_called()


def test_vault_provider_loads_only_allowlisted_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_vault_environment(monkeypatch)
    payload = {key: f"vault-{key.lower()}" for key in REQUIRED_VAULT_SECRET_KEYS}
    payload["UNSAFE_UNDECLARED_KEY"] = "must-not-load"
    client = vault_client(payload)

    status = load_external_secrets(
        Path("missing.env"), client_factory=Mock(return_value=client)
    )

    assert status == {"provider": "vault", "loaded_keys": 6}
    assert "UNSAFE_UNDECLARED_KEY" not in os.environ
    client.secrets.kv.v2.read_secret_version.assert_called_once_with(
        path="bet-ai/production",
        mount_point="secret",
        raise_on_deleted_version=True,
    )
    for key in REQUIRED_VAULT_SECRET_KEYS:
        monkeypatch.delenv(key)


def test_vault_provider_fails_when_required_secret_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_vault_environment(monkeypatch)
    payload = {key: "configured" for key in REQUIRED_VAULT_SECRET_KEYS}
    payload.pop("JWT_SECRET_KEY")

    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        load_external_secrets(
            Path("missing.env"),
            client_factory=Mock(return_value=vault_client(payload)),
        )


def test_vault_provider_supports_approle_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_vault_environment(monkeypatch)
    monkeypatch.delenv("VAULT_TOKEN")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-id")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-id")
    payload = {key: "configured" for key in REQUIRED_VAULT_SECRET_KEYS}
    client = vault_client(payload)

    load_external_secrets(Path("missing.env"), client_factory=Mock(return_value=client))

    client.auth.approle.login.assert_called_once_with(
        role_id="role-id", secret_id="secret-id"
    )
    for key in REQUIRED_VAULT_SECRET_KEYS:
        monkeypatch.delenv(key)


def test_azure_key_vault_uses_managed_identity_and_allowlisted_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SECRET_PROVIDER", "azure_key_vault")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://bets.vault.azure.net")
    monkeypatch.setenv("AZURE_KEY_VAULT_PREFIX", "bet-ai")
    credential = object()
    credential_factory = Mock(return_value=credential)
    client = Mock()
    client.get_secret.side_effect = lambda name: Mock(value=f"value-for-{name}")
    client_factory = Mock(return_value=client)

    status = load_external_secrets(
        Path("missing.env"),
        azure_credential_factory=credential_factory,
        azure_client_factory=client_factory,
    )

    assert status == {"provider": "azure_key_vault", "loaded_keys": 7}
    credential_factory.assert_called_once_with()
    client_factory.assert_called_once_with(
        vault_url="https://bets.vault.azure.net", credential=credential
    )
    requested_names = {call.args[0] for call in client.get_secret.call_args_list}
    assert "bet-ai-jwt-secret-key" in requested_names
    assert "bet-ai-model-signing-key" in requested_names
    assert "bet-ai-database-url" in requested_names
    for key in VAULT_SECRET_KEYS:
        monkeypatch.delenv(key)
