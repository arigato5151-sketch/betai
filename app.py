from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILES = [ROOT_DIR / ".env", ROOT_DIR / "backend" / ".env"]
for env_file in ENV_FILES:
    if env_file.exists():
        load_dotenv(env_file, override=False)
        break

from backend.app.main import app

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
