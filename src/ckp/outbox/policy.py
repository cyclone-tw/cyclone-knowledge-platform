"""Versioned replay dispositions for base movement and writer refusals.

The outbox never runs Git itself; C7 owns the repository. What this policy
freezes is how each C7 outcome maps onto the record lifecycle: only
decidably transient movement earns a bounded retry, incompatible outcomes
land in a stable conflict or quarantine, and an unknown code -- a writer
namespace this policy version has never seen -- fails closed instead of
being optimistically replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal
from ckp.writer.errors import WriterErrorCode

POLICY_VERSION = "outbox-base-policy-v1"


class ReplayDispositionKind(StrEnum):
    #: Base moved under the operation; retry from the current base, bounded.
    RETRY_TRANSIENT = "retry-transient"
    #: The writer leader was busy; retry without consuming a transient attempt.
    RETRY_FREE = "retry-free"
    #: A stable, human-visible conflict; never silently rewritten.
    CONFLICT = "conflict"
    #: Admission no longer holds or the outcome is undecidable; fail closed.
    QUARANTINE = "quarantine"


@dataclass(frozen=True, slots=True)
class ReplayDisposition:
    kind: ReplayDispositionKind


_TRANSIENT = frozenset({WriterErrorCode.BASE_MOVED})
_FREE = frozenset({WriterErrorCode.LEADER_UNAVAILABLE})
_CONFLICT = frozenset(
    {WriterErrorCode.PATH_CONFLICT, WriterErrorCode.IDEMPOTENCY_CONFLICT}
)


class BaseMovementPolicyV1:
    """Map one writer refusal code plus the attempt count to a disposition."""

    def __init__(self, *, max_transient_attempts: int) -> None:
        if (
            not isinstance(max_transient_attempts, int)
            or isinstance(max_transient_attempts, bool)
            or max_transient_attempts < 1
        ):
            raise OutboxRefusal(OutboxErrorCode.INTERNAL_ERROR)
        self._max_transient_attempts = max_transient_attempts

    @property
    def version(self) -> str:
        return POLICY_VERSION

    def dispose(self, code: str, attempts: int) -> ReplayDisposition:
        try:
            writer_code = WriterErrorCode(code)
        except ValueError:
            return ReplayDisposition(kind=ReplayDispositionKind.QUARANTINE)
        if writer_code in _TRANSIENT:
            if attempts < self._max_transient_attempts:
                return ReplayDisposition(kind=ReplayDispositionKind.RETRY_TRANSIENT)
            return ReplayDisposition(kind=ReplayDispositionKind.QUARANTINE)
        if writer_code in _FREE:
            return ReplayDisposition(kind=ReplayDispositionKind.RETRY_FREE)
        if writer_code in _CONFLICT:
            return ReplayDisposition(kind=ReplayDispositionKind.CONFLICT)
        return ReplayDisposition(kind=ReplayDispositionKind.QUARANTINE)


__all__ = [
    "POLICY_VERSION",
    "BaseMovementPolicyV1",
    "ReplayDisposition",
    "ReplayDispositionKind",
]
