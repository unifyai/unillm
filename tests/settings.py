from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # Cache
    UNILLM_CACHE: bool = True
    UNILLM_CACHE_BACKEND: str = "local_separate"
    UNILLM_SERVICE_TIER: str = "priority"
    UNILLM_DEFAULT_MODEL: str = "gpt-5.5@openai"


SETTINGS = Settings()
