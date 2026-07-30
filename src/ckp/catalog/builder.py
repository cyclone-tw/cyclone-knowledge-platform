"""Build a generated Catalog from one immutable, privacy-gated snapshot."""

from __future__ import annotations

import re

from ckp.bundle import BundleSnapshot
from ckp.catalog.models import (
    CatalogSnapshot,
    CatalogUnavailableError,
    PrivacyBindingError,
)
from ckp.catalog.parser import project_member
from ckp.privacy import Admitted, PrivacyGate

_PUBLIC_COMMIT_ID = re.compile(r"[0-9a-f]{7,64}")


def _public_bundle_commit(value: str | None) -> str | None:
    """Project only commit-shaped provenance into public result payloads."""
    if value is None or _PUBLIC_COMMIT_ID.fullmatch(value) is None:
        return None
    return value


class CatalogBuilder:
    """The sole Catalog builder; there is no hand-maintained second Catalog."""

    def __init__(self, privacy_gate: PrivacyGate) -> None:
        self._privacy_gate = privacy_gate

    @property
    def privacy_gate(self) -> PrivacyGate:
        return self._privacy_gate

    def build(self, snapshot: BundleSnapshot) -> CatalogSnapshot:
        if snapshot.index_revision is None:
            raise CatalogUnavailableError("bundle-unavailable")

        bundle_commit = _public_bundle_commit(snapshot.bundle_commit)
        entries = []
        for member in snapshot.members:
            verdict = self._privacy_gate.admit_member(member)
            if not isinstance(verdict, Admitted):
                continue
            if (
                verdict.member_key != member.relative_path
                or verdict.content_sha256 != member.content_sha256
            ):
                raise PrivacyBindingError("privacy admission did not bind snapshot")
            entry = project_member(
                member,
                index_revision=snapshot.index_revision,
                bundle_commit=bundle_commit,
            )
            if entry is not None:
                entries.append(entry)

        return CatalogSnapshot(
            entries=tuple(sorted(entries, key=lambda item: item.concept_id)),
            index_revision=snapshot.index_revision,
            bundle_commit=bundle_commit,
        )


__all__ = ["CatalogBuilder"]
