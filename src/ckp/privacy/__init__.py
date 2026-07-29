"""Privacy prefilter: class vocabulary, classifier, and admission gate.

Rule source: ``cyclone-wiki`` ``Core/decision-cyclone-privacy-sink-v1.md``
(frozen). This package transcribes it; it does not fork it.

Later children wire this in as a **required** constructor dependency:
the C3 Catalog/Gateway read path and the C8 outbox enqueue path both take a
classifier or gate at construction, never as an optional flag (red line R1).
"""

from ckp.privacy.classes import PrivacyClass
from ckp.privacy.classifier import (
    Classification,
    Classified,
    Classifier,
    FrontmatterClassifier,
    Unclassified,
)
from ckp.privacy.gate import (
    Admitted,
    PrivacyGate,
    PrivacyPolicyError,
    Refused,
    Verdict,
)

__all__ = [
    "Admitted",
    "Classification",
    "Classified",
    "Classifier",
    "FrontmatterClassifier",
    "PrivacyClass",
    "PrivacyGate",
    "PrivacyPolicyError",
    "Refused",
    "Unclassified",
    "Verdict",
]
