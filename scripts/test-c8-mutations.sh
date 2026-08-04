#!/usr/bin/env bash
# C8 outbox mutation matrix. Every mutant runs in an isolated TMPDIR copy;
# the issue worktree and any user checkout are never edited or cleaned.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$(CDPATH= cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
fi
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-c8-mutations.XXXXXX")"
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

C8_TESTS=(
  tests/test_outbox_errors.py
  tests/test_outbox_models.py
  tests/test_outbox_crypto.py
  tests/test_outbox_store.py
  tests/test_outbox_enqueue_gate.py
  tests/test_outbox_privacy_routing.py
  tests/test_outbox_replay.py
  tests/test_outbox_recovery.py
  tests/test_c8_contract.py
)

expect_green "pristine C8 outbox contract" "${C8_TESTS[@]}"

echo "==> R1: privacy routing before enqueue"
new_case
replace_once \
  "src/ckp/privacy/classifier.py" \
'        if not declarations:
            return Unclassified(reason=REASON_PRIVACY_MISSING)' \
'        if not declarations:
            return Classified(
                privacy=PrivacyClass.PUBLIC,
                source="frontmatter",
                member_key=member.relative_path,
                content_sha256=member.content_sha256,
            )'
expect_red \
  "classifier defaults an undeclared note to public" \
  tests/test_outbox_enqueue_gate.py::test_an_undetermined_privacy_class_cannot_be_enqueued

new_case
replace_once \
  "src/ckp/outbox/service.py" \
  '            if isinstance(verdict, Refused):' \
  '            if False and isinstance(verdict, Refused):'
expect_red \
  "enqueue privacy refusal mapping removed" \
  tests/test_outbox_enqueue_gate.py::test_an_undetermined_privacy_class_cannot_be_enqueued

new_case
replace_once \
  "src/ckp/outbox/service.py" \
  '            self._validate_in_sandbox(target.path, content)' \
  '            pass  # mutant removes admission sandbox validation'
expect_red \
  "admission sandbox validators removed" \
  tests/test_outbox_enqueue_gate.py::test_worktree_validators_run_in_a_sandbox_before_persistence

echo "==> R2: generic outbox hard deny"
new_case
replace_once \
  "src/ckp/privacy/gate.py" \
  '        if outcome.privacy is PrivacyClass.STUDENT_PRIVATE:' \
  '        if False and outcome.privacy is PrivacyClass.STUDENT_PRIVATE:'
expect_red \
  "student-private dedicated verdict refusal removed" \
  tests/test_outbox_privacy_routing.py::test_raw_student_private_is_refused_with_its_dedicated_code

new_case
replace_once \
  "src/ckp/privacy/gate.py" \
  '        if PrivacyClass.STUDENT_PRIVATE in admissible:' \
  '        if False and PrivacyClass.STUDENT_PRIVATE in admissible:'
expect_red \
  "admissible set can be widened to student-private" \
  tests/test_outbox_privacy_routing.py::test_the_outbox_gate_cannot_be_built_to_admit_student_private

echo "==> encryption at rest"
new_case
replace_once \
  "src/ckp/outbox/crypto.py" \
  '        ciphertext = _keystream_xor(enc_key, nonce, plaintext)' \
  '        ciphertext = plaintext'
expect_red \
  "payload persisted as plaintext passthrough" \
  tests/test_outbox_crypto.py::test_roundtrip_and_ciphertext_hides_plaintext \
  tests/test_outbox_enqueue_gate.py::test_successful_enqueue_ran_every_validator_and_persisted_once

new_case
replace_once \
  "src/ckp/outbox/crypto.py" \
  '        if not hmac.compare_digest(expected, envelope.mac):' \
  '        if False and not hmac.compare_digest(expected, envelope.mac):'
expect_red \
  "envelope MAC verification removed" \
  tests/test_outbox_crypto.py::test_any_tamper_fails_closed

echo "==> durable binding and dedup"
new_case
replace_once \
  "src/ckp/outbox/models.py" \
  '        content_hash.encode("ascii"),' \
  '        b"",  # mutant drops content hash from the binding'
expect_red \
  "content hash removed from the versioned record binding" \
  tests/test_outbox_models.py::test_binding_hash_covers_every_frozen_dimension

new_case
replace_once \
  "src/ckp/outbox/service.py" \
  '                    if not _identity_matches(existing, record):' \
  '                    if False and not _identity_matches(existing, record):'
expect_red \
  "same-key different-binding enqueue accepted as dedup" \
  tests/test_outbox_replay.py::test_duplicate_enqueue_dedups_and_different_binding_conflicts

new_case
replace_once \
  "src/ckp/outbox/models.py" \
  '    if payload.get("record_version") != RECORD_VERSION:' \
  '    if False and payload.get("record_version") != RECORD_VERSION:'
expect_red \
  "unknown record version accepted for parsing" \
  tests/test_outbox_models.py::test_record_from_json_fails_closed_on_version_and_binding_drift

echo "==> leader, lease, expiry, and purge"
new_case
replace_once \
  "src/ckp/outbox/store.py" \
  '                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)' \
  '                    None  # mutant removes exclusive lock acquisition'
expect_red \
  "exclusive consumer leader removed" \
  tests/test_outbox_store.py::test_locks_are_exclusive_and_fail_closed_on_contention \
  tests/test_outbox_replay.py::test_concurrent_consumers_have_exactly_one_leader

new_case
replace_once \
  "src/ckp/outbox/service.py" \
  '                    and _parse(record.lease_until) > now' \
  '                    and True'
expect_red \
  "bounded lease expiry ignored" \
  tests/test_outbox_replay.py::test_a_live_lease_is_respected_and_recovered_after_expiry

new_case
replace_once \
  "src/ckp/outbox/service.py" \
  '                if now >= _parse(record.expires_at):' \
  '                if False and now >= _parse(record.expires_at):'
expect_red \
  "record expiry check removed" \
  tests/test_outbox_replay.py::test_expired_records_fail_closed_and_purge_their_payload

new_case
replace_once \
  "src/ckp/outbox/service.py" \
  '            update["envelope"] = None' \
  '            update["envelope"] = record.envelope'
expect_red \
  "terminal states retain the encrypted payload" \
  tests/test_outbox_replay.py::test_enqueue_then_replay_commits_exactly_once_with_evidence \
  tests/test_c8_contract.py::test_terminal_states_purge_and_sandbox_is_always_removed

echo "==> replay policy"
new_case
replace_once \
  "src/ckp/outbox/policy.py" \
'        except ValueError:
            return ReplayDisposition(kind=ReplayDispositionKind.QUARANTINE)' \
'        except ValueError:
            return ReplayDisposition(kind=ReplayDispositionKind.RETRY_FREE)'
expect_red \
  "unknown writer codes optimistically replayed" \
  tests/test_outbox_recovery.py::test_unknown_writer_codes_are_never_optimistically_replayed

echo "==> review round-1 hardening guards"
new_case
replace_once \
  "src/ckp/outbox/store.py" \
'        self._write_index(record.idempotency_hash, name)
        try:
            self._atomic_write(path, _encode(record))
        except OutboxRefusal:' \
'        self._atomic_write(path, _encode(record))
        self._write_index(record.idempotency_hash, name)
        try:
            pass
        except OutboxRefusal:'
expect_red \
  "record committed before its provisional index" \
  tests/test_outbox_store.py::test_torn_enqueue_is_refused_and_never_resurrects

new_case
replace_once \
  "src/ckp/outbox/store.py" \
  '        for leftover in self._admission.iterdir():' \
  '        for leftover in ():'
expect_red \
  "crashed admission sandboxes are never swept" \
  tests/test_outbox_recovery.py::test_crashed_admission_sandboxes_are_swept_on_recover

new_case
replace_once \
  "src/ckp/outbox/store.py" \
  '            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):' \
  '            if False and stat.S_ISLNK(value.st_mode):'
expect_red \
  "symlinked store subdirectory accepted" \
  tests/test_outbox_store.py::test_open_refuses_symlinked_store_subdirectories

new_case
replace_once \
  "src/ckp/outbox/crypto.py" \
'        except Exception:
            root = None' \
'        except Exception as exc:
            raise OutboxRefusal(OutboxErrorCode.KEY_DENIED) from exc'
expect_red \
  "key provider exception chain reattached" \
  tests/test_outbox_crypto.py::test_provider_failures_never_chain_key_material

new_case
replace_once \
  "src/ckp/outbox/service.py" \
  '            shutil.rmtree(sandbox)' \
  '            shutil.rmtree(sandbox, ignore_errors=True)'
expect_red \
  "sandbox cleanup failures silently swallowed" \
  tests/test_c8_contract.py::test_terminal_states_purge_and_sandbox_is_always_removed

echo "==> review round-2 hardening guards"
new_case
replace_once \
  "src/ckp/outbox/store.py" \
  '            holder = self._record_holding_key(idempotency_hash)' \
  '            holder = None'
expect_red \
  "a lost index entry unbinds a live key" \
  tests/test_outbox_store.py::test_a_lost_index_entry_does_not_unbind_a_live_key

new_case
replace_once \
  "src/ckp/outbox/store.py" \
  '            if path.exists():
                with contextlib.suppress(OutboxRefusal):
                    self._quarantine_raw(path)' \
  '            if False and path.exists():
                with contextlib.suppress(OutboxRefusal):
                    self._quarantine_raw(path)'
expect_red \
  "unprovable record write survives its refusal" \
  tests/test_outbox_store.py::test_unprovable_record_write_is_rolled_back_not_enqueued

new_case
replace_once \
  "src/ckp/outbox/store.py" \
'    def load(self, name: str) -> OutboxRecord | None:
        if _RECORD_NAME.fullmatch(name) is None:' \
'    def load(self, name: str) -> OutboxRecord | None:
        if False and _RECORD_NAME.fullmatch(name) is None:'
expect_red \
  "record-name shape guard removed from load" \
  tests/test_outbox_store.py::test_index_content_is_never_turned_into_a_path

new_case
replace_once \
  "src/ckp/outbox/store.py" \
  '                _RECORD_NAME.fullmatch(target) is None' \
  '                False'
expect_red \
  "index-content shape guard removed from recover" \
  tests/test_outbox_store.py::test_index_content_is_never_turned_into_a_path

new_case
replace_once \
  "src/ckp/outbox/crypto.py" \
'        except OutboxRefusal as exc:
            refusal_code = (
                exc.code
                if isinstance(exc.code, OutboxErrorCode)
                else OutboxErrorCode.KEY_DENIED
            )' \
'        except OutboxRefusal:
            raise'
expect_red \
  "provider-minted refusal chain passes through" \
  tests/test_outbox_crypto.py::test_provider_minted_refusals_are_reminted_without_chain

echo "==> review round-3 hardening guards"
new_case
replace_once \
  "src/ckp/outbox/store.py" \
'            except OutboxRefusal as refusal:
                if refusal.code is OutboxErrorCode.STORAGE_FAILED:
                    raise
                continue' \
'            except OutboxRefusal:
                continue'
expect_red \
  "transient read failures free a bound key" \
  tests/test_outbox_store.py::test_transient_read_failures_never_free_a_bound_key

new_case
replace_once \
  "src/ckp/outbox/store.py" \
'            except OutboxRefusal as refusal:
                if refusal.code is OutboxErrorCode.STORAGE_FAILED:
                    raise
                self._quarantine_raw(path)
                continue' \
'            except OutboxRefusal:
                self._quarantine_raw(path)
                continue'
expect_red \
  "transient read failures evict healthy records to quarantine" \
  tests/test_outbox_store.py::test_scan_does_not_quarantine_on_transient_failures

new_case
replace_once \
  "src/ckp/outbox/store.py" \
'            if _RECORD_NAME.fullmatch(name) is None:
                self._quarantine_raw(path)
                continue' \
'            if False and _RECORD_NAME.fullmatch(name) is None:
                self._quarantine_raw(path)
                continue'
expect_red \
  "stray filenames break the scan instead of being quarantined" \
  tests/test_outbox_store.py::test_stray_filenames_are_quarantined_without_failing_the_pass

echo "==> receipt and namespace guards"
new_case
replace_once \
  "src/ckp/outbox/errors.py" \
  '    REQUEST_INVALID = "outbox/v1/request-invalid"' \
  '    REQUEST_INVALID = "outbox/v2/request-invalid"'
expect_red \
  "stable outbox v1 namespace drifts" \
  tests/test_outbox_errors.py::test_outbox_v1_namespace_and_codes_are_frozen

new_case
replace_once \
  "src/ckp/outbox/errors.py" \
  '    return WRITER_ADMISSION_CODES.get(code, OutboxErrorCode.INTERNAL_ERROR)' \
  '    return WRITER_ADMISSION_CODES.get(code, OutboxErrorCode.REQUEST_INVALID)'
expect_red \
  "unknown writer refusals stop failing closed at admission" \
  tests/test_outbox_errors.py::test_transaction_phase_and_unknown_codes_fail_closed

new_case
replace_once \
  "src/ckp/outbox/models.py" \
'    code: str = Field(
        json_schema_extra={"enum": [code.value for code in OutboxErrorCode]}
    )' \
'    code: str = Field(
        json_schema_extra={"enum": [code.value for code in OutboxErrorCode]}
    )
    body: str | None = None'
expect_red \
  "reject receipt grows a body field" \
  tests/test_outbox_models.py::test_receipts_have_exact_fields_and_strict_evidence

new_case
replace_once \
  "src/ckp/outbox/models.py" \
  '    timestamp: str | None = None' \
'    timestamp: str | None = None
    prompt: str | None = None'
expect_red \
  "replay receipt grows a prompt field" \
  tests/test_outbox_models.py::test_receipts_have_exact_fields_and_strict_evidence

echo "mutation: all $CASE_NUMBER C8 mutants turned red"
echo "mutation: isolated artifacts retained at $MUTATION_ROOT"
