from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


def _parse_bool(v: Any) -> bool:
    """Parse a value as boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1", "on")
    return bool(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # Unify
    UNIFY_API_KEY: SecretStr = SecretStr("")

    # OpenAI
    OPENAI_API_KEY: SecretStr = SecretStr("")

    # Anthropic
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")

    # Google Vertex AI
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr = SecretStr("")
    VERTEXAI_LOCATION: SecretStr = SecretStr("")
    VERTEXAI_PROJECT: SecretStr = SecretStr("")

    # ─────────────────────────────────────────────────────────────────────────
    # LLM I/O Logging
    # ─────────────────────────────────────────────────────────────────────────
    # When enabled, writes request/response payloads to files for debugging.
    # Directory: {UNILLM_LOG_DIR}/ (required when UNILLM_IO_LOG is True)
    UNILLM_IO_LOG: bool = False
    UNILLM_LOG_DIR: str = ""

    @field_validator("UNILLM_IO_LOG", mode="before")
    @classmethod
    def parse_io_log(cls, v: Any) -> bool:
        return _parse_bool(v)


SETTINGS = Settings()
