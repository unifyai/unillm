"""
Canonical cache-key computation.

Byte-identical raw keys are the cache's ground truth: every stored entry keeps
the exact serialized request that produced its response. Canonical keying is a
derived, more lenient address for the same entries — a versioned digest over a
normalized form of the request — so that mundane prompt churn (block
reordering, docstring rewording, volatile timestamps) does not orphan an
otherwise-valid recording. Genuine contract changes (schema fields, parameter
types, message structure) still change the digest.

Both lookup sides derive the digest from the same representation: the raw key
string. Live requests serialize to their raw key first and digest that; index
builds digest the raw keys already on disk. This guarantees symmetry without a
second serialization path.

The pipeline is versioned. Any change to the transforms must bump
CANON_VERSION, which invalidates derived indexes (they rebuild from raw keys —
an offline scan, never an LLM spend) rather than corrupting lookups.
"""

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

CANON_VERSION = "canon-v1"

# Requests are canonically equal regardless of transport bookkeeping.
_DROPPED_KEYS = frozenset({"stream_options"})

# Assistant content is recorded model output: its exact bytes are the
# trajectory's identity, so it is never block-sorted or scrubbed.
_ORDERED_CONTENT_ROLES = frozenset({"assistant"})

# Volatile substrings that vary per run without changing what is being asked.
# Applied to non-assistant text only. Order matters: composite patterns
# (timestamps, UUIDs, paths) must run before the bare hex fallback.
_VOLATILE_PATTERNS = (
    # ISO-8601 datetimes, including the dashed-time form used in log dirs.
    (
        re.compile(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}[:\-]\d{2}"
            r"(?:[:\-]\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?",
        ),
        "<TS>",
    ),
    (
        re.compile(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
            r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        ),
        "<UUID>",
    ),
    (re.compile(r"(?:/private)?/tmp/[\w./-]+"), "<TMP>"),
    (re.compile(r"unitypid\d+"), "<PID>"),
    (re.compile(r"[0-9a-fA-F]{32,}"), "<HEX>"),
)


def parse_raw_key(raw_key: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Split a raw cache key back into (fn_name, kwargs), or None.

    Raw keys are ``f"{fn_name}_{json.dumps(kwargs)}"``; the payload always
    starts at the first ``_{``. Unparseable keys (foreign lines in a store)
    simply have no canonical address.
    """
    sep = raw_key.find("_{")
    if sep == -1:
        return None
    try:
        kw = json.loads(raw_key[sep + 1 :])
    except json.JSONDecodeError:
        return None
    if not isinstance(kw, dict):
        return None
    return raw_key[:sep], kw


def _normalize_text(text: str) -> str:
    """Whitespace- and unicode-normalize prose without reordering it."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _scrub_volatile(text: str) -> str:
    for pattern, placeholder in _VOLATILE_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def _canonical_prose(text: str) -> str:
    """Normalize, scrub, and block-sort instruction-side prose.

    Blocks are blank-line-delimited paragraphs. Sorting them makes the digest
    invariant to pure reorderings of independent prompt sections, which is the
    churn class that prompt-assembly refactors produce.
    """
    text = _scrub_volatile(_normalize_text(text))
    return "\n\n".join(sorted(text.split("\n\n")))


def _canonical_content_part(part: Any, role: Any) -> Any:
    if not isinstance(part, dict):
        return part
    part = dict(part)
    if part.get("type") == "text" and isinstance(part.get("text"), str):
        if role in _ORDERED_CONTENT_ROLES:
            part["text"] = _normalize_text(part["text"])
        else:
            part["text"] = _canonical_prose(part["text"])
    elif part.get("type") == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str) and url.startswith("data:"):
                digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
                part["image_url"] = {
                    **image_url,
                    "url": f"data:sha256:{digest}",
                }
    return part


def _canonical_message(msg: Any) -> Any:
    if not isinstance(msg, dict):
        return msg
    msg = dict(msg)
    role = msg.get("role")
    content = msg.get("content")
    if isinstance(content, str):
        if role in _ORDERED_CONTENT_ROLES:
            msg["content"] = _normalize_text(content)
        else:
            msg["content"] = _canonical_prose(content)
    elif isinstance(content, list):
        msg["content"] = [_canonical_content_part(p, role) for p in content]
    return msg


def _strip_descriptions(node: Any) -> Any:
    """Drop string-valued ``description`` fields from a tool/schema tree.

    Descriptions steer the model but describe an unchanged callable contract;
    they are this codebase's highest-frequency prompt churn. Schema structure
    (names, parameters, types, required) stays in the digest. The string-value
    guard keeps properties that merely *name* a field "description" intact,
    since those map to sub-schemas rather than prose.
    """
    if isinstance(node, dict):
        return {
            k: _strip_descriptions(v)
            for k, v in node.items()
            if not (k == "description" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_strip_descriptions(v) for v in node]
    return node


def _tool_sort_key(tool: Any) -> str:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return json.dumps(tool, sort_keys=True)


def _canonical_response_format(value: Any) -> Any:
    # BaseModel response formats serialize as a JSON-schema string inside the
    # raw key; parse so description-stripping reaches the schema either way.
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, (dict, list)):
            return _strip_descriptions(parsed)
        return value
    return _strip_descriptions(value)


def canonical_kw(kw: Dict[str, Any]) -> Dict[str, Any]:
    """The normalized request form the canonical digest is computed over."""
    out: Dict[str, Any] = {}
    for key, value in kw.items():
        if value is None or key in _DROPPED_KEYS:
            continue
        if key == "messages" and isinstance(value, list):
            out[key] = [_canonical_message(m) for m in value]
        elif key == "tools" and isinstance(value, list):
            out[key] = sorted(
                (_strip_descriptions(t) for t in value),
                key=_tool_sort_key,
            )
        elif key == "response_format":
            out[key] = _canonical_response_format(value)
        else:
            out[key] = value
    return out


def canonical_digest(fn_name: str, kw: Dict[str, Any]) -> str:
    canonical = json.dumps(
        canonical_kw(kw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(f"{fn_name}|{canonical}".encode("utf-8")).hexdigest()
    return f"{CANON_VERSION}:{digest}"


def canonical_digest_of_raw_key(raw_key: str) -> Optional[str]:
    """Digest a stored raw key, or None when the key is not a request key."""
    parsed = parse_raw_key(raw_key)
    if parsed is None:
        return None
    return canonical_digest(*parsed)
