"""C8 durable outbox: synthetic-only, closed by default, admission first.

The package exports classes and builders only -- no instance, no prebuilt
store, no default key material. The default application never composes an
outbox and no HTTP surface exposes one; ``tests/test_c8_contract.py`` pins
both properties.
"""

from ckp.outbox.composition import build_c8_synthetic_outbox
from ckp.outbox.crypto import (
    CIPHER_VERSION,
    OutboxCipherV1,
    OutboxKeyProvider,
    SyntheticKeyProvider,
)
from ckp.outbox.errors import (
    ERROR_PREFIX,
    OutboxErrorCode,
    OutboxRefusal,
    writer_code_to_outbox_code,
)
from ckp.outbox.models import (
    RECORD_VERSION,
    TERMINAL_STATES,
    WRITER_CONTRACT,
    EncryptedEnvelope,
    OutboxEnqueueReceipt,
    OutboxRecord,
    OutboxRejectReceipt,
    OutboxReplayReceipt,
    OutboxState,
    compute_binding_hash,
)
from ckp.outbox.policy import (
    POLICY_VERSION,
    BaseMovementPolicyV1,
    ReplayDisposition,
    ReplayDispositionKind,
)
from ckp.outbox.service import EnqueueReceipt, OutboxService, ReplayClock
from ckp.outbox.store import OutboxStore

__all__ = [
    "CIPHER_VERSION",
    "ERROR_PREFIX",
    "POLICY_VERSION",
    "RECORD_VERSION",
    "TERMINAL_STATES",
    "WRITER_CONTRACT",
    "BaseMovementPolicyV1",
    "EncryptedEnvelope",
    "EnqueueReceipt",
    "OutboxCipherV1",
    "OutboxEnqueueReceipt",
    "OutboxErrorCode",
    "OutboxKeyProvider",
    "OutboxRecord",
    "OutboxRefusal",
    "OutboxRejectReceipt",
    "OutboxReplayReceipt",
    "OutboxService",
    "OutboxState",
    "OutboxStore",
    "ReplayClock",
    "ReplayDisposition",
    "ReplayDispositionKind",
    "SyntheticKeyProvider",
    "build_c8_synthetic_outbox",
    "compute_binding_hash",
    "writer_code_to_outbox_code",
]
