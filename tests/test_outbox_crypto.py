"""The payload cipher: roundtrip, tamper evidence, and fail-closed keys."""

from __future__ import annotations

import base64

import pytest

from ckp.outbox.crypto import (
    CIPHER_VERSION,
    OutboxCipherV1,
    SyntheticKeyProvider,
)
from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal

KEY_ID = "synthetic-outbox-key-1"
KEY = b"SYNTHETIC-OUTBOX-KEY-0123456789abcdef"
PLAINTEXT = b'{"body": "SYNTHETIC-OUTBOX-BODY-NEEDLE"}'


def _cipher(key: bytes = KEY) -> OutboxCipherV1:
    return OutboxCipherV1(SyntheticKeyProvider({KEY_ID: key}))


def test_roundtrip_and_ciphertext_hides_plaintext() -> None:
    cipher = _cipher()
    envelope = cipher.encrypt(KEY_ID, PLAINTEXT)
    assert envelope.cipher == CIPHER_VERSION
    raw = base64.b64decode(envelope.ciphertext)
    assert b"SYNTHETIC-OUTBOX-BODY-NEEDLE" not in raw
    assert cipher.decrypt(envelope) == PLAINTEXT


def test_nonces_are_unique_per_encryption() -> None:
    cipher = _cipher()
    first = cipher.encrypt(KEY_ID, PLAINTEXT)
    second = cipher.encrypt(KEY_ID, PLAINTEXT)
    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext


@pytest.mark.parametrize("field", ["ciphertext", "nonce", "key_id", "mac"])
def test_any_tamper_fails_closed(field: str) -> None:
    cipher = OutboxCipherV1(SyntheticKeyProvider({KEY_ID: KEY, "other-key": b"x" * 32}))
    envelope = cipher.encrypt(KEY_ID, PLAINTEXT)
    tampered_values = {
        "ciphertext": base64.b64encode(
            bytes([base64.b64decode(envelope.ciphertext)[0] ^ 1])
            + base64.b64decode(envelope.ciphertext)[1:]
        ).decode("ascii"),
        "nonce": ("0" * len(envelope.nonce)),
        "key_id": "other-key",
        "mac": "0" * 64,
    }
    tampered = envelope.model_copy(update={field: tampered_values[field]})
    with pytest.raises(OutboxRefusal) as refusal:
        cipher.decrypt(tampered)
    assert refusal.value.code is OutboxErrorCode.RECORD_CORRUPT


def test_wrong_key_material_is_indistinguishable_from_corruption() -> None:
    envelope = _cipher().encrypt(KEY_ID, PLAINTEXT)
    wrong = _cipher(key=b"DIFFERENT-SYNTHETIC-KEY-0123456789ab")
    with pytest.raises(OutboxRefusal) as refusal:
        wrong.decrypt(envelope)
    assert refusal.value.code is OutboxErrorCode.RECORD_CORRUPT


def test_unknown_key_id_fails_closed_as_key_denied() -> None:
    envelope = _cipher().encrypt(KEY_ID, PLAINTEXT)
    missing = OutboxCipherV1(SyntheticKeyProvider({"another": b"y" * 32}))
    with pytest.raises(OutboxRefusal) as refusal:
        missing.decrypt(envelope)
    assert refusal.value.code is OutboxErrorCode.KEY_DENIED


def test_unknown_cipher_version_fails_closed() -> None:
    envelope = _cipher().encrypt(KEY_ID, PLAINTEXT)
    future = envelope.model_copy(update={"cipher": "ckp-outbox-cipher-v2"})
    with pytest.raises(OutboxRefusal) as refusal:
        _cipher().decrypt(future)
    assert refusal.value.code is OutboxErrorCode.VERSION_UNKNOWN


def test_key_provider_refuses_weak_or_missing_material() -> None:
    with pytest.raises(OutboxRefusal):
        SyntheticKeyProvider({})
    with pytest.raises(OutboxRefusal):
        SyntheticKeyProvider({KEY_ID: b"short"})
    provider = SyntheticKeyProvider({KEY_ID: KEY})
    with pytest.raises(OutboxRefusal) as refusal:
        provider.key("missing-key-id")
    assert refusal.value.code is OutboxErrorCode.KEY_DENIED


def test_provider_failures_never_chain_key_material() -> None:
    """A provider's own error text may embed secrets; the refusal must not
    carry it in __cause__ or __context__."""

    class LeakyProvider:
        def key(self, key_id: str) -> bytes:
            del key_id
            raise RuntimeError("SYNTHETIC-LEAKED-KEY-MATERIAL")

    cipher = OutboxCipherV1(LeakyProvider())
    with pytest.raises(OutboxRefusal) as refusal:
        cipher.encrypt("any-key-id", PLAINTEXT)
    assert refusal.value.code is OutboxErrorCode.KEY_DENIED
    assert refusal.value.__cause__ is None
    assert refusal.value.__context__ is None
