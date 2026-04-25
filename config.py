"""Load config.yaml + environment variables into a simple settings object."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    raw: dict[str, Any]
    api_key: str
    secret_key: str
    live_trading: bool

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load(path: Path | str = ROOT / "config.yaml") -> Settings:
    with open(path) as f:
        raw = yaml.safe_load(f)

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    live = os.getenv("LIVE_TRADING", "false").strip().lower() == "true"

    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env "
            "(copy from .env.example)."
        )

    return Settings(raw=raw, api_key=api_key, secret_key=secret_key, live_trading=live)
