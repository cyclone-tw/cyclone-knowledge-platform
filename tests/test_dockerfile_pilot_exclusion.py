"""Issue #23: `src/ckp/pilot/` must never reach the runtime image.

`scripts/smoke-container.sh` only runs a fixed, named list of test files
inside the built image (see its `docker run ... -m pytest -q tests/test_*`
invocations) -- it never imports `ckp.pilot`, so a Dockerfile regression that
lets the pilot package back into the image would build, boot, and pass every
existing container smoke check without anyone noticing. This is a structural
guard on the Dockerfile text itself: cheap, fast, and it fires on exactly the
one thing that actually matters -- ordering.

Not a substitute for actually building the image; a docker build is
expensive to run on every `pytest` invocation, and the ordering assertion
below is what a Dockerfile edit could plausibly break by accident (moving
`RUN rm -rf ./src/ckp/pilot` above `COPY src/ ./src/`, or dropping it while
touching nearby lines).
"""

from __future__ import annotations

from conftest import REPO_ROOT

_PILOT_REMOVAL = "RUN rm -rf ./src/ckp/pilot"
_COPY_SRC = "COPY src/ ./src/"
# The production install line, deliberately without the dev/index extras --
# distinct from the c7-writer-smoke stage's `".[dev,index]"` install so the
# two stages cannot be confused with each other.
_PIP_INSTALL = "RUN python -m pip install --no-cache-dir --disable-pip-version-check ."


def _runtime_stage_lines() -> list[str]:
    """The lines of the first (`runtime`) build stage only.

    Later stages (`c7-writer-smoke`, `final`) build *from* `runtime` and
    never re-run `COPY src/`, so this guard is only meaningful scoped to the
    stage that actually assembles `/app/src`.
    """
    dockerfile = REPO_ROOT / "Dockerfile"
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    stage_starts = [
        index for index, line in enumerate(lines) if line.startswith("FROM ")
    ]
    assert stage_starts, "Dockerfile has no FROM lines"
    first_stage_end = stage_starts[1] if len(stage_starts) > 1 else len(lines)
    return lines[stage_starts[0] : first_stage_end]


def test_pilot_removed_before_pip_install_in_the_runtime_stage() -> None:
    stage = _runtime_stage_lines()

    copy_index = next(
        (i for i, line in enumerate(stage) if line.strip() == _COPY_SRC), None
    )
    removal_index = next(
        (i for i, line in enumerate(stage) if line.strip() == _PILOT_REMOVAL), None
    )
    install_index = next(
        (i for i, line in enumerate(stage) if line.strip() == _PIP_INSTALL), None
    )

    assert copy_index is not None, f"Dockerfile runtime stage is missing {_COPY_SRC!r}"
    assert removal_index is not None, (
        f"Dockerfile runtime stage is missing {_PILOT_REMOVAL!r} -- "
        "src/ckp/pilot/ would ship in the runtime image (issue #23)"
    )
    assert install_index is not None, (
        f"Dockerfile runtime stage is missing {_PIP_INSTALL!r}"
    )

    assert copy_index < removal_index < install_index, (
        "src/ckp/pilot/ must be copied in, then removed, then installed -- "
        f"found COPY at line {copy_index}, rm -rf at {removal_index}, "
        f"pip install at {install_index} within the runtime stage"
    )
