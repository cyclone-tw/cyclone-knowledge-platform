"""The real QMD baseline adapter (issue #24, Epic #21 D3 frozen decision).

Everything in this package is benchmark-support tooling, like
``ckp.pilot`` -- excluded from the runtime image and from the packaged
wheel's served surface. It shells out to the real ``qmd`` CLI against a
*named* index (D3: never the flagless default, which reads a stale
mirror -- see ``cyclone-wiki`` ``scripts/qmd-refresh.sh``) and never reads,
stores, or returns Wiki note bodies -- only relative paths and scores, the
same "path and hash only, never content" shape ``ckp.pilot.manifest``
already uses for the same reason (RP1: this repo never contains Wiki
content).
"""

from benchmarks.qmd.adapter import (
    DEFAULT_QMD_BINARY,
    DEFAULT_QMD_TIMEOUT_SECONDS,
    QmdBaselineConfig,
    QmdHit,
    QmdQueryResult,
    QmdUnavailable,
    qmd_binary_available,
    qmd_token_cost,
    run_qmd_query,
)

__all__ = [
    "DEFAULT_QMD_BINARY",
    "DEFAULT_QMD_TIMEOUT_SECONDS",
    "QmdBaselineConfig",
    "QmdHit",
    "QmdQueryResult",
    "QmdUnavailable",
    "qmd_binary_available",
    "qmd_token_cost",
    "run_qmd_query",
]
