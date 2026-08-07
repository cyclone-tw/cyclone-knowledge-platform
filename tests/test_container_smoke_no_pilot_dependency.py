"""Issue #26 Round 3 (Codex): the test files ``scripts/smoke-container.sh``
actually *runs* inside the built image must execute cleanly with
``ckp.pilot`` entirely absent -- not just importable at module load time.

Round 2 fixed a module-level ``from ckp.pilot import ...`` in
``benchmarks/questions.py`` (issue #23 excludes ``ckp.pilot`` from the
runtime image; the C5 container smoke imports ``benchmarks`` at module load
time). That fix was necessary but incomplete: ``tests/test_shadow_
benchmark.py`` -- one of the exact files the smoke actually runs with
``pytest`` inside the image, not just imports -- had a *function-body*
``from ckp.pilot import PILOT_NOTE_PATHS`` that only executes when pytest
calls the test, not when the module is collected. The container smoke
genuinely calls it, so it failed the same way, one hop further down the same
file (AGENTS.md §9's third question: "does the same root cause still exist
at the other read/write points on the same data path?").

This guard therefore does not stop at "the module imports" (a plain
``importlib.import_module`` over the smoke file list would have missed the
Round 3 bug entirely, the same way Round 2's fix alone did): it actually
*runs* every test file the container smoke invokes, in a subprocess with
``ckp.pilot`` unimportable, and requires the run to finish clean.
"""

from __future__ import annotations

import re
import subprocess
import sys

from conftest import REPO_ROOT

_SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke-container.sh"

# The blocker is inserted at ``sys.meta_path[0]`` -- ahead of *any* other
# resolution mechanism, including a PEP 660 editable-install finder, which
# would otherwise still resolve ``ckp.pilot`` via the real on-disk
# ``src/ckp/pilot/`` regardless of ``sys.path`` order. This is the same
# technique used to hand-verify this fix (see the coordinator's Round 3
# request), now automated instead of a one-off manual check.
_BOOTSTRAP_TEMPLATE = """
import sys

class _BlockPilot:
    def find_spec(self, name, path=None, target=None):
        if name == "ckp.pilot" or name.startswith("ckp.pilot."):
            raise ModuleNotFoundError(f"No module named {{name!r}}")
        return None

sys.meta_path.insert(0, _BlockPilot())
sys.modules.pop("ckp.pilot", None)

import pytest

raise SystemExit(pytest.main({files!r} + ["-q"]))
"""


def _smoke_test_files() -> list[str]:
    """Parse the exact file list out of ``scripts/smoke-container.sh``
    itself, rather than hand-maintaining a second copy here that could
    silently drift from what the script actually runs (the same reasoning
    as the ``_PILOT_NOTE_PATHS`` mirror drift guard in
    ``tests/test_shadow_benchmark.py``).
    """
    text = _SMOKE_SCRIPT.read_text(encoding="utf-8")
    files = sorted(set(re.findall(r"^\s*(tests/test_\w+\.py)", text, re.MULTILINE)))
    return files


def test_smoke_script_still_names_a_nonempty_file_list() -> None:
    """A parser that silently returns an empty list would make the real
    guard below vacuously pass -- pin the parse itself first.
    """
    files = _smoke_test_files()
    assert len(files) >= 20, files
    assert "tests/test_shadow_benchmark.py" in files
    assert "tests/test_c5_contract.py" in files
    assert all(f.startswith("tests/test_") and f.endswith(".py") for f in files)


def test_every_container_smoke_test_file_runs_without_ckp_pilot() -> None:
    """Actually run (not just import) every smoke-invoked test file with
    ``ckp.pilot`` unimportable, in a subprocess so the blocker cannot leak
    into -- or be defeated by -- this process's own already-imported
    ``ckp.pilot`` (several other test files in this suite import it
    directly and would otherwise poison ``sys.modules`` for every test that
    runs after them in the same process).
    """
    files = _smoke_test_files()
    bootstrap = _BOOTSTRAP_TEMPLATE.format(files=files)

    result = subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert "No module named 'ckp.pilot'" not in combined, combined
    assert result.returncode == 0, combined
