"""Recover from OpenAI Responses invalid encrypted reasoning replay.

OpenAI (via OpenRouter Responses) occasionally mints ``encrypted_content`` on
a reasoning item that cannot be verified on the next turn, even when the
client echoes the bytes unchanged. Sibling items from the same conversation
can still be valid. Visible chat/tool history is sufficient to continue; the
encrypted payload is optional chain-of-thought continuity.

When a completion fails with ``invalid_encrypted_content`` /
``invalid_prompt`` naming a reasoning id, drop that item (or all encrypted
reasoning items if the id is absent / not found) from the request messages
in place and retry once.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

_REASONING_ID_RE = re.compile(r"\b(rs_[A-Za-z0-9]+)\b")

_INVALID_ENCRYPTED_CONTENT_MARKERS = (
    "invalid_encrypted_content",
    "encrypted content could not be decrypted",
    "encrypted content could not be verified",
)


def is_invalid_encrypted_content_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is OpenAI rejecting replayed encrypted reasoning."""
    msg = str(exc).lower()
    if any(marker in msg for marker in _INVALID_ENCRYPTED_CONTENT_MARKERS):
        return True
    # OpenRouter sometimes wraps the same failure as invalid_prompt.
    return "invalid_prompt" in msg and "encrypted content" in msg


def offending_reasoning_ids(exc: BaseException) -> list[str]:
    """Parse ``rs_…`` ids mentioned in an invalid-encrypted-content error."""
    if not is_invalid_encrypted_content_error(exc):
        return []
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for match in _REASONING_ID_RE.finditer(str(exc)):
        rid = match.group(1)
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def _item_id(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        raw = item.get("id")
    else:
        raw = getattr(item, "id", None)
    return str(raw) if raw else None


def _item_has_encrypted_content(item: Any) -> bool:
    if isinstance(item, dict):
        enc = item.get("encrypted_content")
    else:
        enc = getattr(item, "encrypted_content", None)
    return isinstance(enc, str) and bool(enc)


def _filter_reasoning_items(
    items: Sequence[Any],
    *,
    drop_ids: Optional[set[str]],
) -> list[Any]:
    """Drop targeted ids, else every item that carries encrypted_content."""
    if drop_ids:
        return [item for item in items if _item_id(item) not in drop_ids]
    return [item for item in items if not _item_has_encrypted_content(item)]


def strip_encrypted_reasoning_from_messages(
    messages: Iterable[Any],
    *,
    drop_ids: Optional[Iterable[str]] = None,
) -> bool:
    """Remove bad encrypted reasoning items from chat messages in place.

    If ``drop_ids`` is non-empty, only those reasoning item ids are removed.
    Otherwise every ``reasoning_items`` entry that has ``encrypted_content``
    is removed. Returns whether any message was mutated.
    """
    id_set = {str(x) for x in drop_ids} if drop_ids else None
    # If specific ids were requested but none match, fall back to stripping
    # all encrypted reasoning so the retry is still useful.
    targeted = bool(id_set)
    matched_targeted = False
    changed = False

    message_list = list(messages)
    if targeted:
        for msg in message_list:
            if not isinstance(msg, dict):
                continue
            items = msg.get("reasoning_items")
            if not items:
                continue
            for item in items:
                if _item_id(item) in id_set:
                    matched_targeted = True
                    break
            if matched_targeted:
                break
        if not matched_targeted:
            id_set = None

    for msg in message_list:
        if not isinstance(msg, dict):
            continue
        items = msg.get("reasoning_items")
        if not items:
            continue
        filtered = _filter_reasoning_items(items, drop_ids=id_set)
        if len(filtered) == len(items):
            continue
        changed = True
        if filtered:
            msg["reasoning_items"] = filtered
        else:
            msg.pop("reasoning_items", None)

    if changed:
        logger.info(
            "Stripped encrypted reasoning items from request history "
            "(drop_ids=%s) after invalid_encrypted_content",
            sorted(id_set) if id_set else "all-encrypted",
        )
    return changed


def strip_encrypted_reasoning_for_error(
    messages: Iterable[Any],
    exc: BaseException,
) -> bool:
    """Strip offending (or all encrypted) reasoning items for ``exc``."""
    return strip_encrypted_reasoning_from_messages(
        messages,
        drop_ids=offending_reasoning_ids(exc),
    )
