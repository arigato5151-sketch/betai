#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILES = [ROOT_DIR / ".env", ROOT_DIR / "backend" / ".env"]
REQUIRED_ENV_VARS = [
    "API_FOOTBALL_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "JWT_SECRET_KEY",
    "JWT_REFRESH_SECRET_KEY",
    "BACKEND_CORS_ORIGINS",
]
DEPENDENCY_CHECKS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "sqlalchemy": "sqlalchemy",
    "pydantic_settings": "pydantic_settings",
    "python_dotenv": "dotenv",
    "httpx": "httpx",
    "redis": "redis",
    "celery": "celery",
    "joblib": "joblib",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit_learn": "sklearn",
    "shap": "shap",
    "passlib": "passlib",
    "python_jose": "jose",
    "psycopg2": "psycopg2",
    "scipy": "scipy",
}

COLOR_MAP = {
    "reset": "\u001b[0m",
    "green": "\u001b[32m",
    "yellow": "\u001b[33m",
    "red": "\u001b[31m",
    "blue": "\u001b[34m",
}


def color(text: str, name: str) -> str:
    return f"{COLOR_MAP.get(name, COLOR_MAP['reset'])}{text}{COLOR_MAP['reset']}"


def manual_load_dotenv(env_path: Path) -> None:
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_environment() -> None:
    for env_file in ENV_FILES:
        if env_file.is_file():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file, override=False)
            except ImportError:
                manual_load_dotenv(env_file)
            return


def check_python_version() -> bool:
    supported = [(3, 11), (3, 12)]
    current = sys.version_info[:2]
    if current not in supported:
        print(color(f"WARNING: Python {current[0]}.{current[1]} is not officially supported.", "yellow"))
        print(color("Recommended versions are Python 3.11 or 3.12.", "yellow"))
        return False
    print(color(f"Python {current[0]}.{current[1]} is supported.", "green"))
    return True


def check_required_env() -> list[str]:
    missing = []
    for key in REQUIRED_ENV_VARS:
        if not os.getenv(key):
            missing.append(key)
    return missing


def check_dependencies() -> list[str]:
    missing = []
    for name, module in DEPENDENCY_CHECKS.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def install_requirements(requirements_path: Path) -> None:
    print(color("Installing missing dependencies...", "blue"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)


def load_app() -> object:
    sys.path.insert(0, str(ROOT_DIR))
    try:
        from app import app

        return app
    except Exception as exc:
        print(color("Failed to import FastAPI app:", "red"), str(exc))
        raise


def run_server(host: str, port: int, reload: bool, log_level: str) -> None:
    app = load_app()
    try:
        import uvicorn
    except ImportError as exc:
        print(color("uvicorn is not installed. Install dependencies first.", "red"))
        raise exc

    uvicorn.run(app, host=host, port=port, reload=reload, log_level=log_level)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bet AI Platform runtime launcher")
    parser.add_argument("--check", action="store_true", help="Validate environment and dependencies")
    parser.add_argument("--install-missing", action="store_true", help="Install missing dependencies from requirements.txt")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the FastAPI server")
    parser.add_argument("--reload", action="store_true", help="Run server with reload enabled")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    load_environment()

    print(color("Bet AI Platform runner starting...", "blue"))
    check_python_version()

    missing_env = check_required_env()
    if missing_env:
        print(color("Missing environment variables:", "red"), ", ".join(missing_env))

    missing_deps = check_dependencies()
    if missing_deps:
        print(color("Missing Python dependencies:", "red"), ", ".join(sorted(missing_deps)))
        if args.install_missing:
            install_requirements(ROOT_DIR / "requirements.txt")
            missing_deps = check_dependencies()

    if args.check:
        if missing_env or missing_deps:
            print(color("Environment validation failed.", "red"))
            return 1
        print(color("Environment validation passed.", "green"))
        return 0

    if missing_env or missing_deps:
        print(color("Startup aborted due to missing configuration or dependencies.", "red"))
        print(color("Use --check to validate or --install-missing to auto-install.", "yellow"))
        return 2

    run_server(args.host, args.port, args.reload, args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
