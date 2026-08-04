"""Injected local key-provider adapter and the outbox v1 payload cipher.

Encryption is not authorization: this module makes payload bytes unreadable
at rest, it never widens what may be enqueued. The construction is stdlib
only -- an HMAC-SHA256 PRF in counter mode with encrypt-then-MAC over
domain-separated subkeys -- because this repo ships no cryptography
dependency and C8 is synthetic-only. Key material arrives exclusively through
the injected provider; nothing here loads a production secret, and neither
keys nor plaintext ever reach records, logs, receipts, or exceptions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Protocol

from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal
from ckp.outbox.models import EncryptedEnvelope

CIPHER_VERSION = "ckp-outbox-cipher-v1"

_ENC_DOMAIN = b"ckp-outbox-enc-v1"
_MAC_DOMAIN = b"ckp-outbox-mac-v1"
_NONCE_BYTES = 16
_MIN_KEY_BYTES = 16
_BLOCK_BYTES = hashlib.sha256().digest_size


class OutboxKeyProvider(Protocol):
    def key(self, key_id: str) -> bytes: ...


class SyntheticKeyProvider:
    """Fixture-only provider over an explicit mapping; unknown ids fail closed."""

    def __init__(self, keys: dict[str, bytes]) -> None:
        if not keys or any(
            not isinstance(key_id, str)
            or not key_id
            or not isinstance(value, bytes)
            or len(value) < _MIN_KEY_BYTES
            for key_id, value in keys.items()
        ):
            raise OutboxRefusal(OutboxErrorCode.KEY_DENIED)
        self._keys = dict(keys)

    def key(self, key_id: str) -> bytes:
        value = self._keys.get(key_id)
        if value is None:
            raise OutboxRefusal(OutboxErrorCode.KEY_DENIED)
        return value


class OutboxCipherV1:
    """Encrypt-then-MAC with per-record random nonces and framed MAC input."""

    def __init__(self, key_provider: OutboxKeyProvider) -> None:
        self._key_provider = key_provider

    def encrypt(self, key_id: str, plaintext: bytes) -> EncryptedEnvelope:
        enc_key, mac_key = self._subkeys(key_id)
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = _keystream_xor(enc_key, nonce, plaintext)
        mac = _mac(mac_key, key_id, nonce, ciphertext)
        return EncryptedEnvelope(
            cipher=CIPHER_VERSION,
            key_id=key_id,
            nonce=nonce.hex(),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            mac=mac,
        )

    def decrypt(self, envelope: EncryptedEnvelope) -> bytes:
        if envelope.cipher != CIPHER_VERSION:
            raise OutboxRefusal(OutboxErrorCode.VERSION_UNKNOWN)
        enc_key, mac_key = self._subkeys(envelope.key_id)
        try:
            nonce = bytes.fromhex(envelope.nonce)
            ciphertext = base64.b64decode(
                envelope.ciphertext.encode("ascii"), validate=True
            )
        except ValueError as exc:
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT) from exc
        expected = _mac(mac_key, envelope.key_id, nonce, ciphertext)
        if not hmac.compare_digest(expected, envelope.mac):
            # A wrong key and a corrupted record are indistinguishable here by
            # design; both fail closed under the corruption code.
            raise OutboxRefusal(OutboxErrorCode.RECORD_CORRUPT)
        return _keystream_xor(enc_key, nonce, ciphertext)

    def _subkeys(self, key_id: str) -> tuple[bytes, bytes]:
        # The provider's exception -- including a provider-minted
        # OutboxRefusal with its own chain -- may embed key material in its
        # message or context. Only the stable code survives; the refusal is
        # re-minted outside the handler so it carries neither a __cause__
        # nor a __context__.
        root: object = None
        refusal_code: OutboxErrorCode | None = None
        try:
            root = self._key_provider.key(key_id)
        except OutboxRefusal as exc:
            refusal_code = (
                exc.code
                if isinstance(exc.code, OutboxErrorCode)
                else OutboxErrorCode.KEY_DENIED
            )
        except Exception:
            root = None
        if refusal_code is not None:
            raise OutboxRefusal(refusal_code)
        if not isinstance(root, bytes) or len(root) < _MIN_KEY_BYTES:
            raise OutboxRefusal(OutboxErrorCode.KEY_DENIED)
        enc_key = hmac.new(root, _ENC_DOMAIN, hashlib.sha256).digest()
        mac_key = hmac.new(root, _MAC_DOMAIN, hashlib.sha256).digest()
        return enc_key, mac_key


def _keystream_xor(enc_key: bytes, nonce: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    offset = 0
    counter = 0
    while offset < len(data):
        block = hmac.new(
            enc_key, nonce + counter.to_bytes(8, "big"), hashlib.sha256
        ).digest()
        chunk = data[offset : offset + _BLOCK_BYTES]
        for index, value in enumerate(chunk):
            out[offset + index] = value ^ block[index]
        offset += _BLOCK_BYTES
        counter += 1
    return bytes(out)


def _mac(mac_key: bytes, key_id: str, nonce: bytes, ciphertext: bytes) -> str:
    digest = hmac.new(mac_key, digestmod=hashlib.sha256)
    for value in (
        CIPHER_VERSION.encode("ascii"),
        key_id.encode("utf-8"),
        nonce,
        ciphertext,
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


__all__ = [
    "CIPHER_VERSION",
    "OutboxCipherV1",
    "OutboxKeyProvider",
    "SyntheticKeyProvider",
]
