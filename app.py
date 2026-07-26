from pathlib import Path
import importlib
import os
import sys

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
BACKEND_APP_DIR = BACKEND_DIR / "app"
ENV_FILES = [ROOT_DIR / ".env", ROOT_DIR / "backend" / ".env"]
for env_file in ENV_FILES:
    if env_file.exists():
        load_dotenv(env_file, override=False)
        break

# Make this compatibility module behave as the ``app`` package when imported
# by Uvicorn, while keeping direct ``python app.py`` execution working.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
__path__ = [str(BACKEND_APP_DIR)]

app = importlib.import_module("app.main").app

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEV", "1").lower() in {"1", "true", "yes"}
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
