"""The four privacy classes, transcribed from the frozen wiki Decision.

Rule source: ``cyclone-wiki`` ``Core/decision-cyclone-privacy-sink-v1.md`` §1,
which itself expands OKF contract §2.4. This module is a *transcription*, not
a schema of its own: the Decision is the authority, revisions land there first
(wiki issue → PR → review) and this repo follows. Do not add, remove, or
reinterpret classes here.

Ordering follows Decision §3: privacy may be maintained or raised, never
lowered. ``public`` is the least restrictive, ``student-private`` the most.
"""

from __future__ import annotations

import enum


class PrivacyClass(enum.Enum):
    """One of the four privacy classes of Decision §1. Nothing else exists.

    The values are the literal frontmatter tokens. There is deliberately no
    parsing helper here that guesses at near-misses -- mapping text onto a
    class is the classifier's job, and every deviation must land on the
    reject side, not on the closest class.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    STUDENT_PRIVATE = "student-private"

    @property
    def strictness(self) -> int:
        """Position in the Decision §3 ordering; higher is more restrictive."""
        return _STRICTNESS[self]

    def is_at_least(self, floor: PrivacyClass) -> bool:
        """True when this class is ``floor`` or stricter.

        The upgrade-only rule (Decision §3): a template or writer may hold or
        raise privacy, never lower it. Callers enforcing that rule ask whether
        the incoming class ``is_at_least`` the declared floor.
        """
        return self.strictness >= floor.strictness


_STRICTNESS = {
    PrivacyClass.PUBLIC: 0,
    PrivacyClass.INTERNAL: 1,
    PrivacyClass.SENSITIVE: 2,
    PrivacyClass.STUDENT_PRIVATE: 3,
}


__all__ = ["PrivacyClass"]
