#!/usr/bin/env bash
# Issue #25 semantic-provider mutation matrix. Every mutant runs in an
# isolated TMPDIR copy; the issue worktree and any user checkout are never
# edited or cleaned.
#
# Deliberately scoped to tests/test_semantic_contract.py only -- the
# weight-free guards (base-dependency leaks, the offline import fence, the
# digest-verification boundary, the stack's self-validation). The
# weight-gated behavior tests in tests/test_semantic_embedding_provider.py
# need the pinned model cache and are exercised by the normal ``pytest``
# step with CKP_REQUIRE_SEMANTIC=1 (mirroring how the C5 Qdrant integration
# tests are excluded from scripts/test-c5-mutations.sh), so this script runs
# the same on a laptop with no model cache and no network as it does in CI.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$(CDPATH= cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
fi
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-c25-mutations.XXXXXX")"
BASELINE="$MUTATION_ROOT/baseline"
CASE_DIR="$MUTATION_ROOT/case"
CASE_NUMBER=0

mkdir -p "$BASELINE" "$CASE_DIR"
tar -C "$SOURCE_ROOT" \
  --exclude='./.git' \
  --exclude='./.venv' \
  --exclude='./.pytest_cache' \
  --exclude='./.ruff_cache' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -cf - . | tar -C "$BASELINE" -xf -
tar -C "$BASELINE" -cf - . | tar -C "$CASE_DIR" -xf -

new_case() {
  CASE_NUMBER=$((CASE_NUMBER + 1))
  tar -C "$BASELINE" -cf - . | tar -C "$CASE_DIR" -xf -
}

replace_count() {
  local relative="$1"
  local expected="$2"
  local before="$3"
  local after="$4"
  local target="$CASE_DIR/$relative"
  BEFORE="$before" AFTER="$after" EXPECTED="$expected" perl -0pi -e '
    BEGIN {
      $before = $ENV{"BEFORE"};
      $after = $ENV{"AFTER"};
      $expected = $ENV{"EXPECTED"};
      $count = 0;
    }
    $count += s/\Q$before\E/$after/g;
    END {
      die "mutation replacement count $count, expected $expected\n"
        unless $count == $expected;
    }
  ' "$target"
}

replace_once() {
  replace_count "$1" 1 "$2" "$3"
}

expect_red() {
  local label="$1"
  shift
  local log="$MUTATION_ROOT/case-$CASE_NUMBER.log"
  local status
  set +e
  (
    cd "$CASE_DIR"
    PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$CASE_DIR/src:$CASE_DIR/tests" \
      "$PYTHON_BIN" -m pytest -p no:cacheprovider "$@"
  ) >"$log" 2>&1
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    echo "mutation survived: $label" >&2
    sed -n '1,180p' "$log" >&2
    exit 1
  fi
  if ! grep -Eq '([1-9][0-9]* failed|FAILED )' "$log"; then
    echo "mutation did not produce an assertion failure: $label" >&2
    sed -n '1,220p' "$log" >&2
    exit 1
  fi
  echo "mutation red: $label"
}

expect_green() {
  local label="$1"
  shift
  local log="$MUTATION_ROOT/baseline.log"
  local status
  set +e
  (
    cd "$CASE_DIR"
    PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$CASE_DIR/src:$CASE_DIR/tests" \
      "$PYTHON_BIN" -m pytest -p no:cacheprovider "$@"
  ) >"$log" 2>&1
  status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    echo "mutation baseline failed: $label" >&2
    sed -n '1,240p' "$log" >&2
    exit 1
  fi
  echo "mutation green: $label"
}

C25_TESTS=(
  tests/test_semantic_contract.py
)

expect_green "pristine issue #25 semantic contract" "${C25_TESTS[@]}"

echo "==> base-dependency and runtime-image leaks"
new_case
replace_once \
  "pyproject.toml" \
  '  "uvicorn>=0.30",' \
  '  "uvicorn>=0.30",
  "onnxruntime>=1.18,<2",'
expect_red \
  "onnxruntime leaks into the base dependencies" \
  tests/test_semantic_contract.py::test_onnxruntime_and_tokenizers_are_not_base_dependencies

new_case
replace_once \
  "pyproject.toml" \
  '  "uvicorn>=0.30",' \
  '  "uvicorn>=0.30",
  "tokenizers>=0.20,<1",'
expect_red \
  "tokenizers leaks into the base dependencies" \
  tests/test_semantic_contract.py::test_onnxruntime_and_tokenizers_are_not_base_dependencies

new_case
replace_once \
  "Dockerfile" \
  'RUN python -m pip install --no-cache-dir --disable-pip-version-check .' \
  'RUN python -m pip install --no-cache-dir --disable-pip-version-check ".[semantic]"'
expect_red \
  "the semantic extra leaks into the shipped runtime image" \
  tests/test_semantic_contract.py::test_the_semantic_extra_is_excluded_from_the_runtime_image

echo "==> the offline import fence holds"
new_case
replace_once \
  "src/ckp/semantic/assets.py" \
  "from ckp.semantic.errors import SemanticErrorCode, SemanticRefusal" \
  "import onnxruntime  # noqa: F401 - mutant leaks the inference backend here
from ckp.semantic.errors import SemanticErrorCode, SemanticRefusal"
expect_red \
  "onnxruntime imported outside the provider module" \
  tests/test_semantic_contract.py::test_onnxruntime_and_tokenizers_are_only_imported_by_the_provider_module

new_case
replace_once \
  "src/ckp/semantic/composition.py" \
  "from ckp.semantic.provider import SemanticEmbeddingProvider" \
  "from tokenizers import Tokenizer  # noqa: F401 - mutant leaks the tokenizer library here
from ckp.semantic.provider import SemanticEmbeddingProvider"
expect_red \
  "tokenizers imported outside the provider module" \
  tests/test_semantic_contract.py::test_onnxruntime_and_tokenizers_are_only_imported_by_the_provider_module

echo "==> asset digest verification cannot be bypassed"
new_case
replace_once \
  "src/ckp/semantic/assets.py" \
  '    if _digest(path) != spec.sha256:
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIGEST_MISMATCH)' \
  '    if False and _digest(path) != spec.sha256:
        raise SemanticRefusal(SemanticErrorCode.ASSET_DIGEST_MISMATCH)'
expect_red \
  "content digest check removed from asset resolution" \
  tests/test_semantic_contract.py::test_a_tampered_asset_file_refuses_with_a_coded_error

new_case
replace_once \
  "src/ckp/semantic/assets.py" \
  '    if not path.is_file():
        raise SemanticRefusal(SemanticErrorCode.ASSETS_MISSING)' \
  '    if False and not path.is_file():
        raise SemanticRefusal(SemanticErrorCode.ASSETS_MISSING)'
expect_red \
  "missing-file check removed: a raw FileNotFoundError escapes instead of a coded refusal" \
  tests/test_semantic_contract.py::test_missing_asset_file_refuses_with_a_coded_error

echo "==> the semantic stack validates itself rather than trusting its builder"
new_case
replace_once \
  "src/ckp/semantic/composition.py" \
  "        embedding_descriptor = require_provider(self.embedding, ProviderKind.EMBEDDING)" \
  "        embedding_descriptor = self.embedding.descriptor  # mutant trusts the caller"
expect_red \
  "a semantic stack can be assembled around an unadmitted provider" \
  tests/test_semantic_contract.py::test_stack_validates_itself_rather_than_trusting_its_builder

new_case
replace_once \
  "src/ckp/semantic/composition.py" \
  "            if revision != compute_provider_revision(descriptor):" \
  "            if False and revision != compute_provider_revision(descriptor):"
expect_red \
  "a semantic stack can carry a revision its own descriptor cannot reproduce" \
  tests/test_semantic_contract.py::test_stack_validates_itself_rather_than_trusting_its_builder

echo "C25 semantic mutation matrix complete: $CASE_NUMBER mutants, all red."
rm -rf "$MUTATION_ROOT"
