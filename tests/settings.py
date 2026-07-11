from typing import Any, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from unillm.types.cache import CACHE_MODES, CacheParam


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # Cache. Accepts true/false or a fine-grained mode (e.g. "read-only"),
    # matching unillm.settings so CI can run the suite read-only.
    UNILLM_CACHE: CacheParam = True
    UNILLM_CACHE_BACKEND: str = "local_separate"
    UNILLM_SERVICE_TIER: str = "priority"
    UNILLM_DEFAULT_MODEL: str = "gpt-5.6-sol@openai"

    @field_validator("UNILLM_CACHE", mode="before")
    @classmethod
    def parse_cache(cls, v: Any) -> Union[bool, str]:
        if v is None or v == "":
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            _lower = v.lower()
            if _lower in ("true", "yes", "1"):
                return True
            if _lower in ("false", "no", "0"):
                return False
            if _lower in CACHE_MODES:
                return _lower
            raise ValueError(
                f"Invalid UNILLM_CACHE value: {v!r}. "
                f"Expected true/false or one of: {', '.join(CACHE_MODES)}",
            )
        return v


SETTINGS = Settings()
