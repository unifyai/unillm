from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    OPENAI_API_KEY: SecretStr = SecretStr("")
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr = SecretStr("")
    VERTEXAI_LOCATION: SecretStr = SecretStr("")
    VERTEXAI_PROJECT: SecretStr = SecretStr("")


SETTINGS = Settings()
