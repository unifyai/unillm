# Load .env BEFORE importing unillm/unify - settings are evaluated at import time
import os
from datetime import datetime
from pathlib import Path

import pytest
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
        from unisdk.utils.http import configure_log_dir as configure_unify_log_dir

        configure_unify_log_dir(str(unify_log_dir))
    except ImportError:
        os.environ["UNISDK_LOG_DIR"] = str(unify_log_dir)

    # Orchestra log directory (for local orchestra server, if running)
    # This sets the env var so that if a local orchestra is started, it knows where to log
    orchestra_log_dir = root_path / "logs" / "orchestra" / subdir
    orchestra_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ORCHESTRA_LOG_DIR"] = str(orchestra_log_dir)

    # Cross-repo OTEL traces (all services write to the same directory)
    otel_log_dir = root_path / "logs" / "all" / subdir
    otel_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UNILLM_OTEL_LOG_DIR"] = str(otel_log_dir)
    os.environ["UNISDK_OTEL_LOG_DIR"] = str(otel_log_dir)
    os.environ["ORCHESTRA_OTEL_LOG_DIR"] = str(otel_log_dir)


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


# ---------------------------------------------------------------------------
# Per-test LLM cost tracking
# ---------------------------------------------------------------------------

from unillm.cost_tracker import capture_costs

_session_costs: list[tuple[str, float]] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    with capture_costs() as events:
        yield
    item._unillm_cost_events = events


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == "call":
        events = getattr(item, "_unillm_cost_events", [])
        total = sum(e.provider_cost for e in events)
        report._unillm_cost = total
        _session_costs.append((report.nodeid, total))


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    outcome = yield
    if report.when == "call":
        result = outcome.get_result()
        if result and len(result) >= 3:
            category, shortletter, verbose = result
            cost = getattr(report, "_unillm_cost", 0.0)
            if isinstance(verbose, str):
                verbose = f"{verbose} [${cost:.6g}]"
            outcome.force_result((category, shortletter, verbose))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    total = sum(cost for _, cost in _session_costs)
    terminalreporter.write_sep("=", f"UNILLM Provider Cost Summary: ${total:.6g}")
