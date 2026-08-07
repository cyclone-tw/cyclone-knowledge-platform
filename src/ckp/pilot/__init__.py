"""P1 pilot corpus binding (issue #23): a read-only manifest over the real
Cyclone-Wiki checkout, never a fixture and never vendored into this repo.

Excluded from the runtime image (Dockerfile) and from the packaged wheel's
default install surface -- this is benchmark-support tooling for Epic #21's
later phases (P2/P5/P9), not part of the served API.
"""

from ckp.pilot.manifest import (
    PILOT_NOTE_PATHS,
    PilotBindingError,
    PilotManifest,
    PilotManifestEntry,
    bind_pilot_corpus,
    load_frozen_manifest,
)

__all__ = [
    "PILOT_NOTE_PATHS",
    "PilotBindingError",
    "PilotManifest",
    "PilotManifestEntry",
    "bind_pilot_corpus",
    "load_frozen_manifest",
]
