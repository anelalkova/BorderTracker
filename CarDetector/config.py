"""
Shared runtime configuration for BorderTracker scripts.

Environment-specific settings live here so the repository can be reused on
different machines without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path


CAR_DETECTOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAR_DETECTOR_DIR.parent


def _load_dotenv() -> None:
    """
    Load the first .env file we find without overriding existing env vars.

    Search order:
      1. repo root (.env)
      2. CarDetector/.env
    """
    candidates = (
        REPO_ROOT / ".env",
        CAR_DETECTOR_DIR / ".env",
    )

    for path in candidates:
        if not path.exists():
            continue

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": _get_int("DB_PORT", 5433),
    "dbname": os.getenv("DB_NAME", "border_crossing"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

STREAM_BASE_URL = os.getenv(
    "STREAM_BASE_URL",
    "https://streaming1.neotel.net.mk/stream/{name}.m3u8",
)

BORDERALARM_BASE_URL = os.getenv(
    "BORDERALARM_BASE_URL",
    "https://borderalarm.com/bottlenecks/{slug}/",
)

DEFAULT_SNAPSHOT_INTERVAL_MIN = _get_int("SNAPSHOT_INTERVAL_MIN", 5)
DEFAULT_BORDERALARM_INTERVAL_MIN = _get_int("BORDERALARM_INTERVAL_MIN", 15)


def build_sqlalchemy_url() -> str:
    return (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )
