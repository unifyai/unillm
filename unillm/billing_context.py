"""Billing context for credit deduction attribution.

Host applications (e.g. Unity) set the billing context so that
``_safe_deduct_credits`` can include assistant/user metadata in
the ledger entry without UniLLM needing direct access to session state.

Usage::

    from unillm.billing_context import set_billing_context

    set_billing_context(assistant_id=42, user_id="user-abc")
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BillingContext:
    assistant_id: Optional[int] = None
    user_id: Optional[str] = None
    organization_id: Optional[int] = None
    source: Optional[str] = None
    label: Optional[str] = None


_BILLING_CONTEXT: ContextVar[BillingContext] = ContextVar(
    "unillm_billing_context",
    default=BillingContext(),
)


def set_billing_context(
    *,
    assistant_id: int | None = None,
    user_id: str | None = None,
    organization_id: int | None = None,
    source: str | None = None,
    label: str | None = None,
) -> None:
    """Set billing attribution for the current async/thread context.

    Args:
        assistant_id: The assistant producing the work.
        user_id: The user whose interaction triggered the cost.
        organization_id: The organization owning the billing account.
        source: What triggered this LLM call — ``"chat"``, ``"call"``,
            ``"tool"``, etc.  Stored in ``detail.source`` on the ledger.
        label: A short, human-readable description of what the assistant
            is working on (e.g. ``"Researching leads"``).  Stored in
            ``detail.label`` on the ledger so usage line items can show
            *what* the spend was for, not just the category.
    """
    _BILLING_CONTEXT.set(
        BillingContext(
            assistant_id=assistant_id,
            user_id=user_id,
            organization_id=organization_id,
            source=source,
            label=label,
        )
    )


def get_billing_context() -> BillingContext:
    """Read the current billing context."""
    return _BILLING_CONTEXT.get()
