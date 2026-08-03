#!/usr/bin/env bash
# C7 Writer mutation matrix. Every mutant runs in an isolated TMPDIR copy;
# the issue worktree and any user checkout are never edited or cleaned.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$(CDPATH= cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
fi
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-c7-mutations.XXXXXX")"
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

C7_TESTS=(
  tests/test_writer_models.py
  tests/test_writer_errors.py
  tests/test_writer_identity.py
  tests/test_writer_target.py
  tests/test_writer_validation.py
  tests/test_writer_transaction.py
  tests/test_c7_contract.py
)

expect_green "pristine C7 Writer contract" "${C7_TESTS[@]}"

echo "==> request, identity, and provenance guards"
new_case
replace_count \
  "src/ckp/writer/models.py" \
  3 \
  '    model_config = ConfigDict(extra="forbid")' \
  '    model_config = ConfigDict(extra="ignore")'
expect_red \
  "caller identity fields are silently accepted" \
  'tests/test_writer_transaction.py::test_preflight_rejections_happen_before_filesystem_mutation'

new_case
replace_once \
  "src/ckp/writer/identity.py" \
  '_UNREGISTERED_WRITER_ACTORS_V1 = frozenset({"grok", "asheron", "polylong"})' \
  '_UNREGISTERED_WRITER_ACTORS_V1 = frozenset()'
expect_red \
  "unregistered C6 actors gain Writer identity" \
  tests/test_writer_identity.py::test_unregistered_c6_read_actors_cannot_become_writer_actors

new_case
replace_once \
  "src/ckp/writer/identity.py" \
  '        if actor is None:' \
  '        if False and actor is None:'
expect_red \
  "unknown actor credential guard removed" \
  tests/test_writer_identity.py::test_missing_or_wrong_actor_credential_fails_closed

new_case
replace_once \
  "src/ckp/writer/identity.py" \
  '                    not isinstance(model_id, str)' \
  '                    False'
expect_red \
  "LLM model identity guard removed" \
  tests/test_writer_identity.py::test_generated_agent_write_requires_runtime_model_identity

new_case
replace_once \
  "src/ckp/writer/identity.py" \
  '            or context._seal.registry_token is not self._registry_token' \
  '            or False'
expect_red \
  "Writer context registry seal removed" \
  tests/test_writer_identity.py::test_context_from_another_registry_cannot_cross_the_writer_boundary

new_case
replace_once \
  "src/ckp/writer/service.py" \
  '            if forbidden:' \
  '            if False and forbidden:'
expect_red \
  "caller provenance spoof guard removed" \
  tests/test_writer_transaction.py::test_preflight_rejections_happen_before_filesystem_mutation

echo "==> privacy and target guards"
new_case
replace_once \
  "src/ckp/writer/service.py" \
  '            if isinstance(verdict, Refused):' \
  '            if False and isinstance(verdict, Refused):'
expect_red \
  "C2 privacy refusal mapping removed" \
  tests/test_writer_transaction.py::test_preflight_rejections_happen_before_filesystem_mutation

new_case
replace_once \
  "src/ckp/writer/service.py" \
  '            self._payload_scanner.scan(content)' \
  '            pass  # mutant removes preflight payload scan'
expect_red \
  "secret and synthetic student scanner removed" \
  tests/test_writer_transaction.py::test_preflight_rejections_happen_before_filesystem_mutation

new_case
replace_once \
  "src/ckp/writer/target.py" \
  '        if self.prefix != _C7_PREFIXES.get(self.zone):' \
  '        if False and self.prefix != _C7_PREFIXES.get(self.zone):'
expect_red \
  "fixed inbox prefix guard removed" \
  tests/test_writer_target.py::test_injected_policy_cannot_broaden_c7_to_formal_routes

new_case
replace_once \
  "src/ckp/writer/target.py" \
  '        if self.privacy_classes != _C7_PRIVACY_CLASSES[self.zone]:' \
  '        if False and self.privacy_classes != _C7_PRIVACY_CLASSES[self.zone]:'
expect_red \
  "route privacy ceiling guard removed" \
  tests/test_writer_target.py::test_injected_policy_cannot_broaden_c7_to_formal_routes

new_case
replace_once \
  "src/ckp/writer/target.py" \
  '        if _SAFE_LEAF.fullmatch(relative_name) is None:' \
  '        if False and _SAFE_LEAF.fullmatch(relative_name) is None:'
expect_red \
  "canonical safe leaf guard removed" \
  tests/test_writer_target.py::test_formal_arbitrary_traversal_and_unsafe_targets_are_refused

new_case
replace_once \
  "src/ckp/writer/target.py" \
  '        if privacy not in target.route.privacy_classes:' \
  '        if False and privacy not in target.route.privacy_classes:'
expect_red \
  "cross-zone privacy guard removed" \
  tests/test_writer_target.py::test_cross_zone_and_student_private_pairs_fail_closed

new_case
replace_once \
  "src/ckp/writer/target.py" \
  '        if privacy is PrivacyClass.STUDENT_PRIVATE:' \
  '        if False and privacy is PrivacyClass.STUDENT_PRIVATE:'
expect_red \
  "student-private dedicated target refusal removed" \
  tests/test_writer_target.py::test_cross_zone_and_student_private_pairs_fail_closed

echo "==> synthetic repository and path guards"
new_case
replace_once \
  "src/ckp/writer/git.py" \
  '        if marker_bytes != SYNTHETIC_MARKER_CONTENT:' \
  '        if False and marker_bytes != SYNTHETIC_MARKER_CONTENT:'
expect_red \
  "synthetic capability marker content guard removed" \
  tests/test_writer_transaction.py::test_repository_capability_requires_physical_committed_marker_and_external_state

new_case
replace_once \
  "src/ckp/writer/git.py" \
'        if _contains(canonical_root, canonical_state) or _contains(
            canonical_state, canonical_root
        ):' \
'        if False and (
            _contains(canonical_root, canonical_state)
            or _contains(canonical_state, canonical_root)
        ):'
expect_red \
  "repository and state-root isolation guard removed" \
  tests/test_writer_transaction.py::test_repository_capability_requires_physical_committed_marker_and_external_state

new_case
replace_once \
  "src/ckp/writer/git.py" \
  '        if branch != "main" or status:' \
  '        if False and (branch != "main" or status):'
expect_red \
  "shared checkout pristine guard removed" \
  tests/test_writer_transaction.py::test_dirty_shared_checkout_is_refused_and_never_cleaned

new_case
replace_once \
  "src/ckp/writer/git.py" \
  '            if mode == "120000":' \
  '            if False and mode == "120000":'
expect_red \
  "base-tree symlink guard removed" \
  tests/test_writer_transaction.py::test_symlink_target_is_refused_without_following_it

echo "==> idempotency binding guards"
new_case
replace_once \
  "src/ckp/writer/git.py" \
  '        if operation_sha is None or idempotency_sha is None:' \
  '        if False and (operation_sha is None or idempotency_sha is None):'
expect_red \
  "partial operation-ref state accepted" \
  tests/test_writer_transaction.py::test_partial_operation_ref_state_fails_closed

new_case
replace_once \
  "src/ckp/writer/git.py" \
  '            self.operation_id == record.operation_id' \
  '            True'
expect_red \
  "idempotency operation_id binding removed" \
  tests/test_writer_transaction.py::test_operation_identity_match_binds_every_identity_field

for field in idempotency_hash actor task_hash target_path request_hash; do
  new_case
  replace_once \
    "src/ckp/writer/git.py" \
    "            and self.$field == record.$field" \
    '            and True'
  expect_red \
    "idempotency $field binding removed" \
    tests/test_writer_transaction.py::test_operation_identity_match_binds_every_identity_field
done

new_case
replace_once \
  "src/ckp/writer/service.py" \
  '        content_hash.encode("ascii"),' \
  '        b"",  # mutant drops content hash binding'
expect_red \
  "content hash removed from versioned operation binding" \
  tests/test_c7_contract.py::test_operation_binding_is_versioned_and_covers_every_frozen_dimension

echo "==> leader, worktree, retry, and validation guards"
new_case
replace_once \
  "src/ckp/writer/git.py" \
  '                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)' \
  '                    None  # mutant removes exclusive leader acquisition'
expect_red \
  "exclusive Writer leader removed" \
  tests/test_writer_transaction.py::test_concurrent_writers_have_one_leader_and_isolated_worktree

new_case
replace_once \
  "src/ckp/writer/git.py" \
  'MAX_BASE_ATTEMPTS = 2' \
  'MAX_BASE_ATTEMPTS = 3'
expect_red \
  "base retry exceeds two total attempts" \
  tests/test_writer_transaction.py::test_base_movement_stops_after_two_total_attempts

new_case
replace_once \
  "src/ckp/writer/service.py" \
  '                        if self._git.read_base() != base_commit:' \
  '                        if False and self._git.read_base() != base_commit:'
expect_red \
  "transient base movement detection removed" \
  tests/test_writer_transaction.py::test_single_transient_base_move_retries_from_fresh_base

for stage in profile lint privacy; do
  new_case
  replace_once \
    "src/ckp/writer/validation.py" \
    "        self._$stage.validate(worktree, target_path)" \
    "        pass  # mutant removes $stage validation"
  expect_red \
    "$stage validation stage removed" \
    tests/test_writer_validation.py::test_validation_pipeline_order_is_profile_lint_privacy
done

new_case
replace_once \
  "src/ckp/writer/git.py" \
  '        if status != expected_untracked:' \
  '        if False and status != expected_untracked:'
replace_once \
  "src/ckp/writer/git.py" \
  '        if cached:' \
  '        if False and cached:'
replace_once \
  "src/ckp/writer/git.py" \
  '            if status != b"A  " + target.encode("utf-8") + b"\x00":' \
  '            if False and status != b"A  " + target.encode("utf-8") + b"\x00":'
replace_once \
  "src/ckp/writer/git.py" \
  '            if names != target.encode("utf-8") + b"\x00":' \
  '            if False and names != target.encode("utf-8") + b"\x00":'
expect_red \
  "diff allowlist guards removed" \
  tests/test_writer_transaction.py::test_diff_allowlist_rejects_validator_side_effect

new_case
replace_count \
  "src/ckp/writer/git.py" \
  2 \
  'raise WriterRefusal(WriterErrorCode.COMMIT_FAILED) from exc' \
  'raise WriterRefusal(WriterErrorCode.INTERNAL_ERROR) from exc'
expect_red \
  "commit failure namespace mapping removed" \
  tests/test_writer_transaction.py::test_commit_failure_is_stable_and_does_not_stage_shared_checkout

new_case
replace_once \
  "src/ckp/writer/validation.py" \
  '        if f"sha256:{digest}" != self._tool_sha256:' \
  '        if False and f"sha256:{digest}" != self._tool_sha256:'
expect_red \
  "Profile validator digest pin removed" \
  tests/test_writer_validation.py::test_profile_adapter_rechecks_digest_before_each_call

new_case
replace_once \
  "src/ckp/writer/validation.py" \
'            "draft",
            "--strict",' \
'            "draft",'
expect_red \
  "Profile validator strict mode removed" \
  tests/test_writer_validation.py::test_profile_adapter_pins_tool_and_uses_single_target_strict_mode

echo "==> receipt, error namespace, and atomic publication guards"
new_case
replace_once \
  "src/ckp/writer/models.py" \
  '    content_hash: str = Field(min_length=1)' \
'    content_hash: str = Field(min_length=1)
    prompt: str | None = None'
expect_red \
  "success receipt grows a prompt field" \
  tests/test_writer_models.py::test_receipts_have_exact_fields_and_strict_success_evidence

new_case
replace_once \
  "src/ckp/writer/models.py" \
'    code: str = Field(
        json_schema_extra={"enum": [code.value for code in WriterErrorCode]}
    )' \
'    code: str = Field(
        json_schema_extra={"enum": [code.value for code in WriterErrorCode]}
    )
    body: str | None = None'
expect_red \
  "reject receipt grows a body field" \
  tests/test_writer_models.py::test_receipts_have_exact_fields_and_strict_success_evidence

new_case
replace_once \
  "src/ckp/writer/errors.py" \
  '    REQUEST_INVALID = "writer/v1/request-invalid"' \
  '    REQUEST_INVALID = "writer/v2/request-invalid"'
expect_red \
  "stable Writer v1 namespace drifts" \
  tests/test_writer_errors.py::test_writer_v1_namespace_and_codes_are_frozen

new_case
replace_once \
  "src/ckp/writer/errors.py" \
  '    "frontmatter-missing": WriterErrorCode.PRIVACY_UNCLASSIFIED,' \
  '    "frontmatter-missing": WriterErrorCode.REQUEST_INVALID,'
expect_red \
  "C2 provisional compatibility mapping drifts" \
  tests/test_writer_errors.py::test_c2_reason_mapping_is_exact_and_unknown_reasons_fail_closed

new_case
replace_once \
  "src/ckp/writer/git.py" \
  '            "prepare\n"' \
  '            ""  # mutant removes atomic prepare'
expect_red \
  "atomic custom-ref transaction prepare removed" \
  tests/test_c7_contract.py::test_git_backend_uses_atomic_refs_and_has_no_shared_cleanup_commands

echo "mutation: all $CASE_NUMBER C7 mutants turned red"
echo "mutation: isolated artifacts retained at $MUTATION_ROOT"
