"""Frozen Writer request, receipt, and refusal contracts."""

# Writer enqueue paths will take this policy dependency explicitly.  Re-export
# the type now so the package cannot grow an ungated enqueue implementation.
from ckp.privacy import PrivacyGate
from ckp.writer.composition import build_c7_synthetic_writer
from ckp.writer.errors import (
    C2_REASON_CODES,
    ERROR_PREFIX,
    WriterErrorCode,
    WriterRefusal,
    c2_reason_to_writer_code,
    map_c2_reason,
)
from ckp.writer.models import RejectReceipt, SuccessReceipt, WriteRequest
from ckp.writer.wiki_capture import (
    CoreInboxCaptureAdapter,
    CoreInboxCaptureReceipt,
    CoreInboxCaptureRequest,
    WikiCaptureErrorCode,
    WikiCaptureRefusal,
)

__all__ = [
    "C2_REASON_CODES",
    "ERROR_PREFIX",
    "PrivacyGate",
    "RejectReceipt",
    "SuccessReceipt",
    "WriteRequest",
    "CoreInboxCaptureAdapter",
    "CoreInboxCaptureReceipt",
    "CoreInboxCaptureRequest",
    "WikiCaptureErrorCode",
    "WikiCaptureRefusal",
    "WriterErrorCode",
    "WriterRefusal",
    "build_c7_synthetic_writer",
    "c2_reason_to_writer_code",
    "map_c2_reason",
]
