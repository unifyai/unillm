from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


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


SETTINGS = Settings()
