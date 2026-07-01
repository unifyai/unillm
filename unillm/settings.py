from typing import Any, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

from unillm.types.cache import CACHE_MODES, CacheParam


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

    # OpenRouter
    OPENROUTER_API_KEY: SecretStr = SecretStr("")

    # Anthropic
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")

    # Google Vertex AI
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr = SecretStr("")
    VERTEXAI_LOCATION: SecretStr = SecretStr("")
    VERTEXAI_PROJECT: SecretStr = SecretStr("")

    # ─────────────────────────────────────────────────────────────────────────
    # LLM I/O Logging — terminal and file are independent
    # ─────────────────────────────────────────────────────────────────────────
    # - UNILLM_TERMINAL_LOG: Controls console output (default: true)
    # - UNILLM_LOG_DIR: Directory for file-based traces (independent of terminal)
    UNILLM_TERMINAL_LOG: bool = True
    UNILLM_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # OpenTelemetry Tracing
    # ─────────────────────────────────────────────────────────────────────────
    # Master switch for OTel tracing.
    # - UNILLM_OTEL=false (default): OTel tracing disabled
    # - UNILLM_OTEL=true: OTel tracing enabled, uses parent TracerProvider if available
    # - UNILLM_OTEL_ENDPOINT: OTLP endpoint for trace export (optional)
    # - UNILLM_OTEL_LOG_DIR: Directory for file-based span export (optional)
    #
    # File-based span export:
    # When UNILLM_OTEL_LOG_DIR is set, spans are written to JSONL files keyed
    # by trace_id. This enables standalone trace logging without a parent
    # TracerProvider or external collector (Tempo/Jaeger).
    UNILLM_OTEL: bool = False
    UNILLM_OTEL_ENDPOINT: str = ""
    UNILLM_OTEL_LOG_DIR: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Response Caching
    # ─────────────────────────────────────────────────────────────────────────
    # Controls whether LLM responses are cached locally.
    # - UNILLM_CACHE=true / false: Enable or disable caching (default: false)
    # - UNILLM_CACHE=<mode>: Fine-grained cache mode
    #   Modes: both, write, read, read-only, read-closest
    UNILLM_CACHE: CacheParam = False

    # ─────────────────────────────────────────────────────────────────────────
    # Transient Error Retry Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Number of retries for transient errors that are incorrectly classified
    # as 400 BadRequest by upstream providers (especially OpenAI).
    #
    # Background: OpenAI occasionally returns HTTP 400 with messages like
    # "something went wrong reading your request" for valid requests. This is
    # a transient server-side processing error, but because it returns as 400
    # (not 5xx), neither the OpenAI SDK nor LiteLLM will retry it.
    #
    # References:
    # - LiteLLM issue: https://github.com/BerriAI/litellm/issues/12503
    #   (400 errors don't trigger fallback/retry even for transient messages)
    # - OpenAI community reports of intermittent "something went wrong" errors:
    #   https://community.openai.com/t/error-something-went-wrong-if-this-issue-persists/200411
    #
    # Set to 0 to disable this retry logic entirely.
    UNILLM_TRANSIENT_RETRY_COUNT: int = 3

    @field_validator("UNILLM_TERMINAL_LOG", "UNILLM_OTEL", mode="before")
    @classmethod
    def parse_bool_fields(cls, v: Any) -> bool:
        return _parse_bool(v)

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
