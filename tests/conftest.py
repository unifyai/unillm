# Load .env BEFORE importing unillm/unify - settings are evaluated at import time
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Look for .env in repo root (parent of tests/)
_repo_root = Path(__file__).resolve().parent.parent
load_dotenv(_repo_root / ".env", override=True)

# Set UNILLM_CACHE_DIR to repo root so cache location is consistent regardless of cwd
os.environ.setdefault("UNILLM_CACHE_DIR", str(_repo_root))

import atexit
import warnings

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

# ---------------------------------------------------------------------------
# Log directory configuration
# ---------------------------------------------------------------------------


def _get_log_subdir() -> str:
    """Generate a datetime-prefixed subdirectory name for log isolation."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # Use a simple identifier (PID) for this repo
    return f"{timestamp}_unillmpid{os.getpid()}"


def pytest_sessionstart(session):
    """Configure all file-based logging directories for trace correlation."""
    root_path = Path(session.config.rootpath)
    subdir = _get_log_subdir()

    # Unillm LLM I/O file logging (raw request/response traces)
    unillm_log_dir = root_path / "logs" / "unillm" / subdir
    unillm_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from unillm import configure_log_dir as configure_unillm_log_dir

        configure_unillm_log_dir(str(unillm_log_dir))
    except ImportError:
        os.environ["UNILLM_LOG_DIR"] = str(unillm_log_dir)

    # Unify SDK file logging (HTTP request traces)
    unify_log_dir = root_path / "logs" / "unify" / subdir
    unify_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from unify.utils.http import configure_log_dir as configure_unify_log_dir

        configure_unify_log_dir(str(unify_log_dir))
    except ImportError:
        os.environ["UNIFY_LOG_DIR"] = str(unify_log_dir)

    # Orchestra log directory (for local orchestra server, if running)
    # This sets the env var so that if a local orchestra is started, it knows where to log
    orchestra_log_dir = root_path / "logs" / "orchestra" / subdir
    orchestra_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ORCHESTRA_LOG_DIR"] = str(orchestra_log_dir)

    # Cross-repo OTEL traces (all services write to the same directory)
    otel_log_dir = root_path / "logs" / "all" / subdir
    otel_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UNILLM_OTEL_LOG_DIR"] = str(otel_log_dir)
    os.environ["UNIFY_OTEL_LOG_DIR"] = str(otel_log_dir)
    os.environ["ORCHESTRA_OTEL_LOG_DIR"] = str(otel_log_dir)


# ---------------------------------------------------------------------------
# Per-test cost tracking
# ---------------------------------------------------------------------------

import pytest

# Controlled via UNILLM_COST_REPORT env var (default: true)
_COST_REPORT_ENABLED = os.environ.get("UNILLM_COST_REPORT", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Module-level dict so pytest_report_teststatus can look up events by nodeid.
# Populated by pytest_runtest_protocol before the test runs (the list is mutable
# and gets filled during the test body).
_cost_events_by_nodeid: dict = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Hook wrapper that captures cost events during the full test protocol.

    Wraps setup + call + teardown so that any LLM calls made in fixtures
    or the test body itself are captured.  The events list is registered in
    ``_cost_events_by_nodeid`` *before* yielding so that
    ``pytest_report_teststatus`` can read it while the test is still in
    progress (the list is the same mutable object and is already populated
    by the time the "call" report is created).
    """
    if not _COST_REPORT_ENABLED:
        yield
        return

    from unillm.cost_tracker import capture_costs

    with capture_costs() as events:
        # Register before yield -- same mutable list gets populated during test
        _cost_events_by_nodeid[item.nodeid] = events
        yield


def _format_cost_label(events) -> str:
    """Build a compact cost label from a list of CostEvents."""
    from unillm.cost_tracker import summarize_costs

    summary = summarize_costs(events)
    parts = [f"${summary.total_provider_cost:.6f}"]
    cache_parts = []
    if summary.cache_hits:
        cache_parts.append(f"{summary.cache_hits}h")
    if summary.cache_misses:
        cache_parts.append(f"{summary.cache_misses}m")
    if cache_parts:
        parts.append(f"({'/'.join(cache_parts)})")
    return " ".join(parts)


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    """Append inline cost info to whatever the default status word is.

    For tests that made LLM calls, the verbose output changes from::

        tests/test_basics.py::test_foo PASSED

    to::

        tests/test_basics.py::test_foo PASSED [$0.001234 (1m)]

    Works for any outcome (PASSED, FAILED, SKIPPED, XFAIL, ERROR, etc.)
    by wrapping the default hook and appending to its verbose word.
    Tests without LLM calls keep the standard output with no suffix.
    """
    outcome = yield

    if not _COST_REPORT_ENABLED or report.when != "call":
        return

    events = _cost_events_by_nodeid.get(report.nodeid)
    if not events:
        return

    result = outcome.get_result()
    if result is None:
        return

    category, short, verbose = result
    cost_label = _format_cost_label(events)
    outcome.force_result((category, short, f"{verbose} [{cost_label}]"))


# TODO: Won't be needed once LiteLLM handles their type annotations correctly...

# Suppress Pydantic serialization warnings from litellm's logging system.
# Litellm registers an atexit handler that flushes logs after pytest finishes,
# which triggers warnings due to type annotation mismatches (Choices vs StreamingChoices).
# The warnings.filterwarnings approach doesn't work during interpreter shutdown,
# so we need to unregister the atexit handler and flush manually with warnings suppressed.

_flush_on_exit = GLOBAL_LOGGING_WORKER._flush_on_exit

# Unregister litellm's atexit handler
atexit.unregister(_flush_on_exit)


def _flush_with_suppressed_warnings():
    """Flush litellm logs with Pydantic serialization warnings suppressed."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings:",
            category=UserWarning,
        )
        _flush_on_exit()


# Register our wrapper instead
atexit.register(_flush_with_suppressed_warnings)
