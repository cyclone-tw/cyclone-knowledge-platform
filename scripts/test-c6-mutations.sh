#!/usr/bin/env bash
# C6 mutation matrix. Mutants run only in an isolated copy under TMPDIR; the
# issue worktree is never edited. The temp tree is intentionally left for the
# OS to reap, avoiding any cleanup command that could target user files.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-c6-mutations.XXXXXX")"
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
  # Overlaying the immutable baseline restores every source path touched by
  # the preceding mutant. Mutations never add or remove source files.
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

echo "==> control and caller-owned scope"
new_case
replace_once \
  "src/ckp/packer/models.py" \
  '    model_config = ConfigDict(extra="forbid")' \
  '    model_config = ConfigDict(extra="ignore")'
expect_red \
  "control: context accepts caller-owned scope fields" \
  tests/test_scoped_gateway.py::test_request_cannot_add_scope_and_validation_receipt_does_not_echo_query

echo "==> credential and task-scope mutations"
new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '        if actor is not grant.actor:' \
  '        if False and actor is not grant.actor:'
expect_red \
  "wrong actor check removed" \
  tests/test_auth_scope.py::test_resolution_failures_are_stable_and_do_not_echo_inputs \
  tests/test_scoped_gateway.py::test_offline_qmd_revalidates_actor_domain_and_expiry

new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '        if now >= grant.expires_at:' \
  '        if False and now >= grant.expires_at:'
expect_red \
  "expired scope check removed" \
  tests/test_auth_scope.py::test_expired_and_not_yet_valid_grants_fail_closed \
  tests/test_scoped_gateway.py::test_offline_qmd_revalidates_actor_domain_and_expiry

new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '        if requested_domain not in grant.domains:' \
  '        if False and requested_domain not in grant.domains:'
expect_red \
  "cross-domain check removed" \
  tests/test_auth_scope.py::test_resolution_failures_are_stable_and_do_not_echo_inputs \
  tests/test_scoped_gateway.py::test_offline_qmd_revalidates_actor_domain_and_expiry

new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '        if not required_capabilities <= grant.capabilities:' \
  '        if False and not required_capabilities <= grant.capabilities:'
expect_red \
  "required capability check removed" \
  tests/test_auth_scope.py::test_resolution_failures_are_stable_and_do_not_echo_inputs \
  tests/test_scoped_gateway.py::test_wrong_actor_expired_missing_capability_and_cross_domain_fail_closed

new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '        if not isinstance(self.domains, frozenset):' \
  '        if False and not isinstance(self.domains, frozenset):'
expect_red \
  "mutable domain scope accepted" \
  tests/test_auth_scope.py::test_grant_scope_and_bounds_are_immutable_and_structurally_valid

new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '        if not isinstance(self.capabilities, frozenset):' \
  '        if False and not isinstance(self.capabilities, frozenset):'
expect_red \
  "mutable capability scope accepted" \
  tests/test_auth_scope.py::test_grant_scope_and_bounds_are_immutable_and_structurally_valid

new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '        if not isinstance(self.privacy_classes, frozenset):' \
  '        if False and not isinstance(self.privacy_classes, frozenset):'
expect_red \
  "mutable privacy scope accepted" \
  tests/test_auth_scope.py::test_grant_scope_and_bounds_are_immutable_and_structurally_valid

new_case
replace_once \
  "src/ckp/auth/policy.py" \
'        if (
            isinstance(self.max_items, bool)
            or not isinstance(self.max_items, int)
            or not 1 <= self.max_items <= 20
        ):' \
  '        if False:'
expect_red \
  "invalid item bounds accepted" \
  tests/test_auth_scope.py::test_grant_scope_and_bounds_are_immutable_and_structurally_valid

new_case
replace_once \
  "src/ckp/auth/policy.py" \
'        if (
            isinstance(self.max_token_upper_bound, bool)
            or not isinstance(self.max_token_upper_bound, int)
            or not 64 <= self.max_token_upper_bound <= 32_768
        ):' \
  '        if False:'
expect_red \
  "invalid token bounds accepted" \
  tests/test_auth_scope.py::test_grant_scope_and_bounds_are_immutable_and_structurally_valid

new_case
replace_count \
  "src/ckp/auth/policy.py" \
  2 \
'        if (
            not isinstance(self.credential_sha256, str)
            or _SHA256_HEX.fullmatch(self.credential_sha256) is None
        ):' \
  '        if False:'
expect_red \
  "malformed credential digest accepted" \
  tests/test_auth_scope.py::test_credential_digests_and_grant_ids_are_unambiguous

new_case
replace_once \
  "src/ckp/auth/policy.py" \
  '            if credential.grant.grant_id in grant_ids:' \
  '            if False and credential.grant.grant_id in grant_ids:'
expect_red \
  "duplicate grant ID accepted" \
  tests/test_auth_scope.py::test_credential_digests_and_grant_ids_are_unambiguous

echo "==> privacy and domain ordering mutations"
new_case
replace_once \
  "src/ckp/gateway/scoped.py" \
  'from ckp.privacy import Classifier, PrivacyGate' \
  'from ckp.privacy import Classifier, PrivacyClass, PrivacyGate'
replace_once \
  "src/ckp/gateway/scoped.py" \
  '        gate = PrivacyGate(self._classifier, context.privacy_classes)' \
  '        gate = PrivacyGate(self._classifier, frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL, PrivacyClass.SENSITIVE}))'
expect_red \
  "grant privacy set ignored before aggregation" \
  tests/test_scoped_gateway.py::test_lower_privacy_grant_cannot_leak_sensitive_counts_facets_or_citations

new_case
replace_once \
  "src/ckp/gateway/scoped.py" \
  '        scoped_entries = self._domain_registry.select(admitted.entries, context.domain)' \
  '        scoped_entries = admitted.entries'
expect_red \
  "domain binding bypassed before search and projections" \
  tests/test_scoped_gateway.py::test_scoped_read_filters_privacy_and_domain_before_every_projection

new_case
replace_once \
  "src/ckp/auth/policy.py" \
'            if entry.id in seen_ids:
                raise DomainPolicyError("duplicate-domain-id")' \
'            if False and entry.id in seen_ids:
                raise DomainPolicyError("duplicate-domain-id")'
expect_red \
  "duplicate durable ID no longer fails closed" \
  tests/test_scoped_gateway.py::test_duplicate_durable_id_makes_scoped_snapshot_unavailable

new_case
replace_once \
  "src/ckp/gateway/scoped.py" \
'        if admitted.bundle_commit is None:
            raise DomainPolicyError("missing-bundle-commit")' \
'        if False and admitted.bundle_commit is None:
            raise DomainPolicyError("missing-bundle-commit")'
expect_red \
  "protected read accepts a snapshot without bundle commit" \
  tests/test_scoped_gateway.py::test_scoped_read_requires_bundle_commit_but_public_c3_remains_available

echo "==> bounded context and citation mutations"
new_case
replace_once \
  "src/ckp/packer/service.py" \
  '        for result in results[:max_items]:' \
  '        for result in results:'
expect_red \
  "item cap removed" \
  tests/test_context_packer.py::test_item_and_token_guards_each_limit_an_otherwise_small_result_set

new_case
replace_once \
  "src/ckp/packer/service.py" \
  '            remaining = max_token_upper_bound - used - separator_bytes' \
  '            remaining = max_token_upper_bound'
expect_red \
  "cumulative token upper bound removed" \
  tests/test_context_packer.py::test_item_and_token_guards_each_limit_an_otherwise_small_result_set

new_case
replace_once \
  "src/ckp/packer/service.py" \
'            if (
                citation.concept_id != result.concept_id
                or citation.index_revision != revision.index_revision
                or citation.bundle_commit != revision.bundle_commit
            ):' \
  '            if False:'
expect_red \
  "citation and response revision binding removed" \
  tests/test_context_packer.py::test_packer_rejects_a_result_not_bound_to_the_response_revision

new_case
replace_once \
  "src/ckp/packer/service.py" \
  '    def __init__(self, privacy_gate: PrivacyGate) -> None:' \
  '    def __init__(self, privacy_gate: PrivacyGate | None = None) -> None:'
expect_red \
  "packer PrivacyGate gains an optional default" \
  tests/test_context_packer.py::test_packer_requires_a_privacy_gate

echo "==> reject receipt mutations"
new_case
replace_once \
  "src/ckp/app.py" \
  '        return rejection(error.code, error.http_status)' \
'        return JSONResponse(
            status_code=error.http_status,
            content={"request_id": "0" * 32, "status": "rejected", "code": error.code, "credential": _request.headers.get("X-CKP-Actor-Credential")},
        )'
expect_red \
  "auth reject receipt echoes actor credential" \
  tests/test_scoped_gateway.py::test_wrong_actor_expired_missing_capability_and_cross_domain_fail_closed

new_case
replace_once \
  "src/ckp/app.py" \
  '        return rejection("invalid-request", 422)' \
'        return JSONResponse(
            status_code=422,
            content={"detail": _error.body},
        )'
expect_red \
  "validation reject receipt echoes query body" \
  tests/test_scoped_gateway.py::test_request_cannot_add_scope_and_validation_receipt_does_not_echo_query

echo "mutation: all $CASE_NUMBER C6 mutants turned red"
echo "mutation: isolated artifacts retained at $MUTATION_ROOT"
