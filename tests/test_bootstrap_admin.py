import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cli import bootstrap_admin
from app.core.passwords import verify_password
from app.db.models import Base, Role
from app.db.user_repository import UserRepository

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


@pytest.fixture(autouse=True)
def sqlite_session(monkeypatch):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestingSessionLocal() as db:
        db.add(Role(name="admin", permissions=[]))
        db.commit()
    monkeypatch.setattr(bootstrap_admin, "SessionLocal", TestingSessionLocal)
    yield


def _pawn_settings(monkeypatch, *, password, secret):
    monkeypatch.setattr(bootstrap_admin.settings, "ADMIN_USERNAME", "rootadmin")
    monkeypatch.setattr(bootstrap_admin.settings, "ADMIN_PASSWORD", password)
    monkeypatch.setattr(bootstrap_admin.settings, "BOOTSTRAP_ADMIN_SECRET", secret)


def test_bootstrap_requires_bootstrap_secret(monkeypatch):
    _pawn_settings(
        monkeypatch,
        password="ignored-password-123",
        secret=None,
    )
    with pytest.raises(SystemExit, match="BOOTSTRAP_ADMIN_SECRET"):
        bootstrap_admin.main()


def test_bootstrap_rejects_short_bootstrap_secret(monkeypatch):
    _pawn_settings(monkeypatch, password="ignored-password-123", secret="short")
    with pytest.raises(SystemExit, match="at least"):
        bootstrap_admin.main()


def test_bootstrap_rejects_weak_or_placeholder_password(monkeypatch):
    _pawn_settings(
        monkeypatch,
        password="change-this-password",
        secret="bootstrap-secret-12345",
    )
    with pytest.raises(SystemExit, match="ADMIN_PASSWORD"):
        bootstrap_admin.main()


def test_bootstrap_creates_admin_with_supplied_password(monkeypatch):
    _pawn_settings(
        monkeypatch,
        password="strong-admin-password",
        secret="bootstrap-secret-12345",
    )
    bootstrap_admin.main()

    with TestingSessionLocal() as db:
        user = UserRepository(db).get_by_identifier("rootadmin")
        assert user is not None
        assert "admin" in UserRepository(db).role_names(user)
        assert verify_password("strong-admin-password", user.password_hash)


def test_bootstrap_generates_one_time_password_when_absent(monkeypatch, capsys):
    _pawn_settings(monkeypatch, password=None, secret="bootstrap-secret-12345")
    bootstrap_admin.main()

    output = capsys.readouterr().out
    assert "Generated one-time password:" in output
    imported = next(
        line.split(":", 1)[1].strip()
        for line in output.splitlines()
        if line.startswith("Generated one-time password:")
    )
    assert len(imported) >= 12
    with TestingSessionLocal() as db:
        user = UserRepository(db).get_by_identifier("rootadmin")
        assert user is not None
        assert verify_password(imported, user.password_hash)


def test_bootstrap_is_skipped_after_first_setup(monkeypatch, capsys):
    _pawn_settings(
        monkeypatch,
        password="strong-admin-password",
        secret="bootstrap-secret-12345",
    )
    bootstrap_admin.main()
    first_setup_output = capsys.readouterr().out

    bootstrap_admin.main()
    second_output = capsys.readouterr().out

    assert "Admin user created" in first_setup_output
    assert "skipped" in second_output
    with TestingSessionLocal() as db:
        users = UserRepository(db).get_by_identifier("rootadmin")
        assert users is not None
