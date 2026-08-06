#!/usr/bin/env bash
# C5 index mutation matrix. Every mutant runs in an isolated TMPDIR copy;
# the issue worktree and any user checkout are never edited or cleaned.
# Qdrant-dependent tests skip when no local service is reachable -- every
# mutant below is killed by a test that needs no service.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$(CDPATH= cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
fi
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-c5-mutations.XXXXXX")"
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
      PYTHONPATH="$CASE_DIR/src:$CASE_DIR/tests:$CASE_DIR" \
      "$PYTHON_BIN" -m pytest -p no:cacheprovider "$@"
  ) >"$log" 2>&1
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    echo "mutation survived: $label" >&2
    sed -n '1,180p' "$log" >&2
    exit 1
  fi
  if ! grep -Eq '([1-9][0-9]* failed|FAILED |[1-9][0-9]* error)' "$log"; then
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
      PYTHONPATH="$CASE_DIR/src:$CASE_DIR/tests:$CASE_DIR" \
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

C5_TESTS=(
  tests/test_index_models.py
  tests/test_index_memory.py
  tests/test_index_rebuild.py
  tests/test_index_neutrality.py
  tests/test_index_qdrant.py
  tests/test_shadow_benchmark.py
  tests/test_c5_contract.py
)

expect_green "pristine C5 index contract" "${C5_TESTS[@]}"

echo "==> rebuild determinism"
new_case
replace_once \
  "src/ckp/index/memory.py" \
  '        self.wipe()
        stored = plan.points' \
  '        stored = plan.points  # mutant skips the wipe'
replace_once \
  "src/ckp/index/memory.py" \
  '        self._plan = plan
        return self._report(plan)

    def search(' \
  '        self._plan = self._plan or plan  # mutant keeps stale corpus
        return self._report(plan)

    def search('
expect_red \
  "M1 rebuild keeps the previous corpus instead of replacing it" \
  tests/test_index_memory.py::test_full_rebuild_replaces_the_previous_corpus

new_case
replace_once \
  "src/ckp/index/models.py" \
  '    payload = digest_key.encode("utf-8")' \
  '    import uuid

    payload = uuid.uuid4().hex.encode("utf-8")'
expect_red \
  "M2 point ids become random instead of derived" \
  tests/test_index_rebuild.py::test_same_commit_same_revision_and_digest

new_case
replace_once \
  "src/ckp/index/revision.py" \
  '        _require_revision_string(embedding_revision).encode("ascii"),' \
  '        b"",  # mutant drops the embedding revision from the composition'
expect_red \
  "M3 composed revision no longer covers the embedding revision" \
  tests/test_index_rebuild.py::test_a_different_embedding_dimension_changes_the_composed_revision

new_case
replace_once \
  "src/ckp/index/models.py" \
  '        if not admitted_public:' \
  '        if False:  # mutant admits every member regardless of privacy'
expect_red \
  "M4 privacy gate bypassed during plan_rebuild" \
  tests/test_index_models.py::test_plan_rebuild_orders_gates_and_counts \
  tests/test_shadow_benchmark.py::test_no_sentinel_or_non_public_path_ever_reaches_the_report

new_case
replace_once \
  "src/ckp/index/models.py" \
  'class SearchHit(BaseModel):
    """One placed result: identifiers, score, rank. Deliberately no body."""

    model_config = ConfigDict(extra="forbid", frozen=True)' \
  'class SearchHit(BaseModel):
    """One placed result: identifiers, score, rank. Deliberately no body."""

    model_config = ConfigDict(extra="allow", frozen=True)'
expect_red \
  "M5 search hits stop forbidding extra payload fields" \
  tests/test_index_models.py::test_hit_models_never_carry_note_bodies

echo "==> offline fence and neutrality"
new_case
replace_once \
  "src/ckp/index/qdrant.py" \
  '        if not isinstance(host, str) or host not in _LOOPBACK_HOSTS:' \
  '        if False:  # mutant admits any remote host'
expect_red \
  "M6 qdrant loopback fence removed" \
  tests/test_index_qdrant.py::test_remote_hosts_are_refused_before_any_connection

new_case
replace_once \
  "src/ckp/index/models.py" \
  '    ordered = sorted(members, key=lambda item: (item.digest_key, item.relative_path))' \
  '    ordered = list(members)  # mutant indexes in caller order'
expect_red \
  "M7 corpus ordering follows input order instead of digest keys" \
  tests/test_index_models.py::test_plan_rebuild_orders_gates_and_counts

new_case
replace_once \
  "src/ckp/index/models.py" \
  '    def require_frozen_total_order(self) -> SearchResult:
        seen: set[str] = set()' \
  '    def require_frozen_total_order(self) -> SearchResult:
        return self  # mutant disables the order validator
        seen: set[str] = set()'
expect_red \
  "M8 search results stop enforcing the frozen total order" \
  tests/test_index_models.py::test_search_hit_and_result_enforce_the_frozen_total_order

echo "==> shadow benchmark honesty"
new_case
replace_once \
  "benchmarks/shadow.py" \
  '        privacy_violations += (len(raw_lexical_paths) - len(lexical_paths)) + (
            len(raw_vector_paths) - len(vector_paths)
        )' \
  '        privacy_violations += 0  # mutant stops counting leaks'
expect_red \
  "M9 privacy violation counter silenced" \
  tests/test_shadow_benchmark.py::test_privacy_violations_are_counted_and_redacted

new_case
replace_once \
  "benchmarks/shadow.py" \
  '        "schema": REPORT_SCHEMA,' \
  '        "schema": REPORT_SCHEMA,
        "generated_at": time.perf_counter(),'
expect_red \
  "M10 wall-clock leaks into the reproducible report" \
  tests/test_shadow_benchmark.py::test_report_is_deterministic_apart_from_latency

new_case
replace_once \
  "benchmarks/shadow.py" \
  '    descriptor = require_index_provider(index_provider)' \
  '    from ckp.index.memory import InMemoryVectorIndex

    index_provider = InMemoryVectorIndex()  # mutant hard-binds one provider
    descriptor = require_index_provider(index_provider)'
expect_red \
  "M11 caller hard-binds the reference provider" \
  tests/test_index_neutrality.py::test_the_same_caller_runs_both_providers_unchanged \
  tests/test_index_neutrality.py::test_caller_source_names_no_concrete_provider

new_case
replace_once \
  "pyproject.toml" \
  '  "uvicorn>=0.30",' \
  '  "uvicorn>=0.30",
  "qdrant-client>=1.14,<1.16",'
expect_red \
  "M12 qdrant client leaks into the base dependencies" \
  tests/test_c5_contract.py::test_qdrant_client_is_not_a_base_dependency

echo "==> snapshot and search bounds"
new_case
replace_once \
  "src/ckp/index/models.py" \
  '    if compute_payload_digest(points) != plan.payload_digest:
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID)' \
  '    if False:  # mutant restores without verifying the digest
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID)'
expect_red \
  "M13 snapshot restore skips digest verification" \
  tests/test_index_memory.py::test_snapshot_restore_roundtrip_and_tamper_detection

new_case
replace_once \
  "src/ckp/index/memory.py" \
  '                for rank, (point, score) in enumerate(scored[:limit])' \
  '                for rank, (point, score) in enumerate(scored)'
expect_red \
  "M14 top_k truncation ignored" \
  tests/test_index_memory.py::test_search_orders_truncates_and_guards

echo "==> plan seal and snapshot metadata"
new_case
replace_once \
  "src/ckp/index/models.py" \
  '    if not isinstance(plan, RebuildPlan) or plan.seal is not _PLAN_SEAL:' \
  '    if False:  # mutant accepts hand-built plans'
expect_red \
  "M15 plan seal admission removed" \
  tests/test_index_memory.py::test_an_unsealed_plan_is_refused

new_case
replace_once \
  "src/ckp/index/models.py" \
  '    if plan.composed_revision != recomputed:
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID)' \
  '    if False:  # mutant trusts snapshot metadata
        raise IndexRefusal(IndexErrorCode.SNAPSHOT_INVALID)'
expect_red \
  "M16 snapshot metadata verification removed" \
  tests/test_index_memory.py::test_snapshot_metadata_tampering_fails_closed

echo "all C5 mutations red (baseline green)"
