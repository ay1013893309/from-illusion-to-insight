"""Configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model: str = os.getenv("CPDP_MODEL", "gpt-5-mini")
    default_backend: str = os.getenv("CPDP_BACKEND", "heuristic")


SETTINGS = Settings()

