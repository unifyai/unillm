import json
import os
from pathlib import Path
from typing import Any, Union

from pydantic import SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from unillm.types.cache import CACHE_KEYINGS, CACHE_MODES, CacheParam

# Provider keys that Secret Manager supplies when the environment does not.
PROVIDER_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MANAGEMENT_API_KEY",
    "TOGETHER_API_KEY",
    "ANTHROPIC_API_KEY",
)
SERVICE_ACCOUNT_KEY = Path("~/.config/gcloud/automation.json").expanduser()


class SecretManagerSource(PydanticBaseSettingsSource):
    """Provider keys from Google Secret Manager, read as the team service
    account whose key sits at ``~/.config/gcloud/automation.json``, from the
    project named by that key's ``project_id``.

    Lowest priority: a key in the environment or a ``.env`` file wins. On a
    machine without that key file the source contributes nothing, so the
    library behaves as a plain env-configured package everywhere else.
    """

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        wanted = [name for name in PROVIDER_KEYS if not os.environ.get(name)]
        if not wanted or not SERVICE_ACCOUNT_KEY.is_file():
            return {}
        from google.api_core.exceptions import NotFound
        from google.cloud import secretmanager
        from google.oauth2 import service_account

        info = json.loads(SERVICE_ACCOUNT_KEY.read_text())
        credentials = service_account.Credentials.from_service_account_info(info)
        client = secretmanager.SecretManagerServiceClient(credentials=credentials)
        values: dict[str, Any] = {}
        for name in wanted:
            try:
                response = client.access_secret_version(
                    name=f"projects/{info['project_id']}/secrets/{name}/versions/latest",
                )
            except NotFound:
                continue
            values[name] = response.payload.data.decode("utf-8")
        return values


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            SecretManagerSource(settings_cls),
        )

    # OpenRouter
    OPENROUTER_API_KEY: SecretStr = SecretStr("")
    # OpenRouter Management API key (sk-or-…). Required only to create/update
    # account-level BYOK provider credentials via POST /api/v1/byok. Regular
    # completion keys cannot register BYOK.
    OPENROUTER_MANAGEMENT_API_KEY: SecretStr = SecretStr("")

    # Together AI — used as OpenRouter BYOK for Together-pinned open-weight
    # models (e.g. MiniMax-M3). Register on the OpenRouter workspace with
    # scripts/register_openrouter_together_byok.py or the Integrations UI.
    TOGETHER_API_KEY: SecretStr = SecretStr("")

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
    #   Modes: both, write, read, read-only
    UNILLM_CACHE: CacheParam = False

    # How cache lookups match stored entries.
    # - exact: byte-identical raw request keys only
    # - canonical: also accept entries whose canonical digest matches, so
    #   mundane prompt churn (block reordering, docstring rewording, volatile
    #   timestamps) keeps hitting recordings made before the change. Writes
    #   always store the exact raw key either way.
    UNILLM_CACHE_KEYING: str = "exact"

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
    # Default 6 → backoff sleeps of 1/2/4/8/16/32s between attempts.
    UNILLM_TRANSIENT_RETRY_COUNT: int = 6

    @field_validator("UNILLM_TERMINAL_LOG", "UNILLM_OTEL", mode="before")
    @classmethod
    def parse_bool_fields(cls, v: Any) -> bool:
        return _parse_bool(v)

    @field_validator("UNILLM_CACHE_KEYING", mode="before")
    @classmethod
    def parse_cache_keying(cls, v: Any) -> str:
        if v is None or v == "":
            return "exact"
        if isinstance(v, str):
            _lower = v.lower()
            if _lower in CACHE_KEYINGS:
                return _lower
        raise ValueError(
            f"Invalid UNILLM_CACHE_KEYING value: {v!r}. "
            f"Expected one of: {', '.join(CACHE_KEYINGS)}",
        )

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
