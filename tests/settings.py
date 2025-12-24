from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # Cache
    UNILLM_CACHE: bool = False
    UNILLM_SERVICE_TIER: str = "priority"
    UNILLM_DEFAULT_MODEL: str = "gpt-5.2@openai"


SETTINGS = Settings()
