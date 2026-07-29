"""The classification gate: red line R1 made structural.

Every read path (C3 Gateway/Catalog) and every enqueue path (C8 outbox) must
pass items through a :class:`PrivacyGate` **before** doing anything else with
them, and must take the gate -- or a classifier to build one -- as a required
constructor parameter, never as an optional flag. A conventions test arms
itself against those packages appearing without wiring this module in.

Three properties are enforced by shape rather than by discipline:

* **An undetermined item cannot pass.** The classifier's failure type carries
  no privacy class, and :class:`Admitted` cannot be minted without one, so
  there is no code path from "undetermined" to "admitted" to get wrong.
* **Admission tokens are sealed.** ``Admitted`` can only be constructed by
  the gate. Downstream code that wants an admitted item has exactly one way
  to get one, so "forgot to call the gate" is a type error, not a latent leak.
* **``student-private`` is never admissible here.** Decision §2's matrix row
  is refusal across every sink this repo will ever host; the only approved
  destination is a local-only restricted store that deliberately does not
  exist in this codebase. Enforced twice -- at construction, and again per
  verdict with a dedicated reason code -- so removing either check alone
  still refuses, and a test pinning the *specific* code catches the removal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import final

from ckp.privacy.classes import PrivacyClass
from ckp.privacy.classifier import Classified, Classifier, Unclassified

#: Provisional reason codes (Decision §8; semantics frozen, names aligned to
#: the Writer error namespace when Phase 3 freezes it).
REASON_STUDENT_PRIVATE_SINK = "student-private-sink"
REASON_NOT_ADMISSIBLE = "privacy-not-admissible"


class PrivacyPolicyError(ValueError):
    """A gate was configured to violate the privacy Decision."""


class _Seal:
    """Module-private capability token; only the gate holds the instance."""

    __slots__ = ()


_SEAL = _Seal()


@dataclass(frozen=True)
class Admitted:
    """Proof that one item passed one gate. Mintable only by the gate.

    The seal is a capability check, not cryptography: it stops the honest
    mistake of constructing an admission by hand instead of running the gate.
    """

    path: Path
    privacy: PrivacyClass
    _seal: _Seal = field(repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise PrivacyPolicyError(
                "Admitted can only be minted by PrivacyGate.admit; "
                "construct a gate and pass the item through it"
            )


@dataclass(frozen=True)
class Refused:
    """A refusal receipt: stable reason code, no payload echo (Decision §7)."""

    path: Path
    reason: str


Verdict = Admitted | Refused


@final
class PrivacyGate:
    """Admit only items whose determined class is inside ``admissible``.

    Both parameters are required. A default classifier would make "nobody
    chose one" look identical to "the right one was chosen", and a default
    admissible set would hide the sink policy inside this module instead of
    at the call site where it can be reviewed.
    """

    def __init__(
        self,
        classifier: Classifier,
        admissible: frozenset[PrivacyClass],
    ) -> None:
        if PrivacyClass.STUDENT_PRIVATE in admissible:
            # Decision §2: every sink in this repo refuses student-private.
            # Decision §3: widening a refused combination is the
            # allowed-sinks-escalation case and fails closed.
            raise PrivacyPolicyError(
                "student-private is never admissible in this repo; "
                "the only approved destination is a local-only restricted "
                "store, which this codebase deliberately does not implement"
            )
        self._classifier = classifier
        self._admissible = admissible

    def admit(self, path: Path) -> Verdict:
        outcome = self._classifier.classify(path)
        if isinstance(outcome, Unclassified):
            return Refused(path=path, reason=outcome.reason)
        assert isinstance(outcome, Classified)
        # Checked per verdict as well as at construction: if the constructor
        # check is ever weakened, this branch still refuses -- and with a
        # *dedicated* code, so the contract test pinning it stays red under
        # either mutation alone.
        if outcome.privacy is PrivacyClass.STUDENT_PRIVATE:
            return Refused(path=path, reason=REASON_STUDENT_PRIVATE_SINK)
        if outcome.privacy not in self._admissible:
            return Refused(path=path, reason=REASON_NOT_ADMISSIBLE)
        return Admitted(path=path, privacy=outcome.privacy, _seal=_SEAL)

    def partition(self, paths: Iterable[Path]) -> tuple[list[Admitted], list[Refused]]:
        """Every path, split into admitted and refused. Nothing is dropped.

        Counting or aggregating over notes must apply privacy *first*
        (contract §2.4); this is the helper that makes doing so easier than
        not doing so. Refusals are returned, not discarded, because a caller
        that cannot see refusals cannot report them.
        """
        admitted: list[Admitted] = []
        refused: list[Refused] = []
        for path in paths:
            verdict = self.admit(path)
            if isinstance(verdict, Admitted):
                admitted.append(verdict)
            else:
                refused.append(verdict)
        return admitted, refused


__all__ = [
    "REASON_NOT_ADMISSIBLE",
    "REASON_STUDENT_PRIVATE_SINK",
    "Admitted",
    "PrivacyGate",
    "PrivacyPolicyError",
    "Refused",
    "Verdict",
]
