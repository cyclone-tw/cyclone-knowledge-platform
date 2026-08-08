#!/usr/bin/env bash
# C3 mutation matrix. Every mutant runs in a private copied tree; the source
# worktree is never mounted or edited. A control mutation must turn red before
# any security mutation is trusted.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-c3-mutations.XXXXXX")"
BASELINE="$MUTATION_ROOT/baseline"
CASE_DIR=""
CASE_NUMBER=0

cleanup() {
  if [ -d "$MUTATION_ROOT" ]; then
    chmod -R u+w "$MUTATION_ROOT" 2>/dev/null || true
    rm -r "$MUTATION_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$BASELINE"
tar -C "$SOURCE_ROOT" \
  --exclude='./.git' \
  --exclude='./.venv' \
  --exclude='./.pytest_cache' \
  --exclude='./.ruff_cache' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -cf - . | tar -C "$BASELINE" -xf -

if find "$BASELINE" -type d -name '__pycache__' -print -quit | grep -q .; then
  echo "mutation: copied baseline contains __pycache__" >&2
  exit 1
fi

new_case() {
  CASE_NUMBER=$((CASE_NUMBER + 1))
  CASE_DIR="$MUTATION_ROOT/case-$CASE_NUMBER"
  mkdir -p "$CASE_DIR"
  cp -R "$BASELINE/." "$CASE_DIR"
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
      "$PYTHON_BIN" -m pytest "$@"
  ) >"$log" 2>&1
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    echo "mutation survived: $label" >&2
    sed -n '1,160p' "$log" >&2
    exit 1
  fi
  if ! grep -Eq '([1-9][0-9]* failed|FAILED )' "$log"; then
    echo "mutation did not produce an assertion failure: $label" >&2
    sed -n '1,200p' "$log" >&2
    exit 1
  fi
  echo "mutation red: $label"
}

echo "==> control mutation"
new_case
replace_once \
  "src/ckp/gateway/models.py" \
  '    limit: int = Field(default=10, ge=1, le=20)' \
  '    limit: int = Field(default=10, ge=1, le=200)'
expect_red \
  "control: query limit guard removed" \
  tests/test_gateway.py::test_query_schema_rejects_policy_override_and_unbounded_inputs

echo "==> privacy and policy mutations"
new_case
replace_once \
  "src/ckp/catalog/builder.py" \
'            verdict = self._privacy_gate.admit_member(member)
            if not isinstance(verdict, Admitted):
                continue
            if (
                verdict.member_key != member.relative_path
                or verdict.content_sha256 != member.content_sha256
            ):
                raise PrivacyBindingError("privacy admission did not bind snapshot")
' \
'            # Mutant: project every member without admission.
'
expect_red \
  "Catalog result paths bypass PrivacyGate" \
  tests/test_catalog.py::test_only_admitted_public_notes_enter_the_catalog \
  tests/test_gateway.py::test_catalog_filters_privacy_before_count_facets_and_pagination

new_case
replace_once \
  "src/ckp/privacy/gate.py" \
  '        admissible: frozenset[PrivacyClass],' \
  '        admissible: frozenset[PrivacyClass] = frozenset({PrivacyClass.PUBLIC}),'
expect_red \
  "PrivacyGate admissible gains a permissive default" \
  tests/test_c3_contract.py::test_every_policy_dependency_stays_required

new_case
replace_once \
  "src/ckp/privacy/classifier.py" \
  '    def __init__(self, bundle_root: Path | AnchoredBundleReader) -> None:' \
  '    def __init__(self, bundle_root: Path | AnchoredBundleReader = Path(".")) -> None:'
expect_red \
  "FrontmatterClassifier bundle root gains a default" \
  tests/test_c3_contract.py::test_every_policy_dependency_stays_required

new_case
replace_once \
  "src/ckp/privacy/classifier.py" \
  '            return Unclassified(reason=REASON_PRIVACY_MISSING)' \
'            return Classified(
                privacy=PrivacyClass.PUBLIC,
                source="frontmatter",
                member_key=member.relative_path,
                content_sha256=member.content_sha256,
            )'
expect_red \
  "undetermined privacy becomes public" \
  tests/test_gateway.py::test_query_excludes_every_non_public_or_undetermined_match

new_case
replace_once \
  "src/ckp/app.py" \
  'PUBLIC_HTTP_ADMISSIBLE = frozenset({PrivacyClass.PUBLIC})' \
  'PUBLIC_HTTP_ADMISSIBLE = frozenset({PrivacyClass.PUBLIC, PrivacyClass.SENSITIVE})'
expect_red \
  "sensitive result enters anonymous HTTP" \
  tests/test_gateway.py::test_query_excludes_every_non_public_or_undetermined_match

new_case
replace_once \
  "src/ckp/app.py" \
  'PUBLIC_HTTP_ADMISSIBLE = frozenset({PrivacyClass.PUBLIC})' \
  'PUBLIC_HTTP_ADMISSIBLE = frozenset({PrivacyClass.PUBLIC, PrivacyClass.STUDENT_PRIVATE})'
replace_once \
  "src/ckp/privacy/gate.py" \
  '        if PrivacyClass.STUDENT_PRIVATE in admissible:' \
  '        if False and PrivacyClass.STUDENT_PRIVATE in admissible:'
replace_once \
  "src/ckp/privacy/gate.py" \
  '        if outcome.privacy is PrivacyClass.STUDENT_PRIVATE:' \
  '        if False and outcome.privacy is PrivacyClass.STUDENT_PRIVATE:'
expect_red \
  "student-private result enters anonymous HTTP" \
  tests/test_gateway.py::test_query_excludes_every_non_public_or_undetermined_match

new_case
replace_once \
  "src/ckp/gateway/service.py" \
'    def _catalog(self) -> CatalogSnapshot:
        snapshot = self._snapshot_cache.get()
        return self._catalog_builder.build(snapshot)
' \
'    def _catalog(self) -> CatalogSnapshot:
        snapshot = self._snapshot_cache.get()
        self._ungated_count = len(snapshot.members)
        return self._catalog_builder.build(snapshot)
'
replace_once \
  "src/ckp/gateway/service.py" \
  '        total = len(filtered)' \
  '        total = self._ungated_count'
expect_red \
  "Catalog count aggregates before privacy filtering" \
  tests/test_gateway.py::test_catalog_filters_privacy_before_count_facets_and_pagination

new_case
replace_once \
  "src/ckp/gateway/models.py" \
'class CatalogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")' \
'class CatalogRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")'
expect_red \
  "Catalog accepts caller privacy policy parameters" \
  tests/test_gateway.py::test_catalog_rejects_caller_owned_privacy_policy_parameters

new_case
replace_once \
  "src/ckp/gateway/models.py" \
'class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")' \
'class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")'
expect_red \
  "query accepts caller privacy policy parameters" \
  tests/test_gateway.py::test_query_schema_rejects_policy_override_and_unbounded_inputs

echo "==> citation and snapshot mutations"
new_case
replace_count \
  "src/ckp/gateway/models.py" \
  2 \
  '    citation: CitationResponse' \
  '    citation: CitationResponse | None = None'
expect_red \
  "result citation becomes optional" \
  tests/test_c3_contract.py::test_result_citations_are_required_fields

new_case
replace_once \
  "src/ckp/gateway/service.py" \
  '        concept_id=entry.citation.concept_id,' \
  '        concept_id="wrong-note",'
expect_red \
  "citation points at the wrong note" \
  tests/test_gateway.py::test_query_is_lexical_deterministic_bounded_and_cited

new_case
replace_once \
  "src/ckp/gateway/service.py" \
  '        index_revision=entry.citation.index_revision,' \
  '        index_revision="sha256:stale",'
expect_red \
  "citation crosses snapshot revisions" \
  tests/test_gateway.py::test_query_is_lexical_deterministic_bounded_and_cited

new_case
replace_once \
  "src/ckp/catalog/builder.py" \
  '        entries = []' \
'        entries = []
        if False:
            __import__("pathlib").Path("bypass").read_bytes()'
expect_red \
  "result pipeline directly calls Path.read_bytes" \
  tests/test_c3_contract.py::test_result_pipeline_has_no_direct_path_reads

echo "==> anchored reader mutations"
new_case
replace_once \
  "src/ckp/bundle.py" \
  '            if not stat.S_ISDIR(root_stat.st_mode):' \
  '            if False and not stat.S_ISDIR(root_stat.st_mode):'
expect_red \
  "opened bundle root is not required to be a directory" \
  tests/test_bundle_reader.py::test_root_fd_must_be_a_directory_even_if_open_flags_are_ignored

new_case
replace_once \
  "src/ckp/bundle.py" \
'    if stat.S_ISLNK(value.st_mode):
        return REFUSAL_SYMLINK' \
'    if False and stat.S_ISLNK(value.st_mode):
        return REFUSAL_SYMLINK'
replace_once \
  "src/ckp/bundle.py" \
  '            if stat.S_ISLNK(before.st_mode):' \
  '            if False and stat.S_ISLNK(before.st_mode):'
expect_red \
  "parent and final symlink guards removed" \
  tests/test_bundle_reader.py::test_final_symlink_is_refused_even_when_it_points_inside \
  tests/test_bundle_reader.py::test_parent_symlink_is_refused

new_case
replace_once \
  "src/ckp/bundle.py" \
  '    if not parts or any(part in {"", ".", ".."} for part in parts):' \
  '    if not parts or any(part in {"", "."} for part in parts):'
expect_red \
  "path traversal reaches an existing outside file" \
  tests/test_bundle_reader.py::test_path_traversal_cannot_read_an_existing_outside_file

new_case
replace_once \
  "src/ckp/bundle.py" \
  '    if not stat.S_ISREG(value.st_mode):' \
  '    if False and not stat.S_ISREG(value.st_mode):'
expect_red \
  "non-regular file becomes readable" \
  tests/test_bundle_reader.py::test_non_regular_member_is_refused_without_opening_it

new_case
replace_once \
  "src/ckp/bundle.py" \
  '    if value.st_nlink != 1:' \
  '    if False and value.st_nlink != 1:'
expect_red \
  "multiple-hardlink refusal removed" \
  tests/test_bundle_reader.py::test_multiple_hardlinks_are_conservatively_refused

new_case
replace_once \
  "src/ckp/bundle.py" \
  '            if not _same_directory(before, after):' \
  '            if False and not _same_directory(before, after):'
expect_red \
  "parent replacement during anchored walk accepted" \
  tests/test_bundle_reader.py::test_parent_replacement_during_walk_is_not_accepted

new_case
replace_once \
  "src/ckp/bundle.py" \
  '                if opened_signature != before_signature:' \
  '                if False and opened_signature != before_signature:'
expect_red \
  "check-to-open inode replacement accepted" \
  tests/test_bundle_reader.py::test_check_to_open_replacement_is_not_accepted

new_case
replace_once \
  "src/ckp/bundle.py" \
  '                if _reason_for_stat(after_stat) is not None:' \
  '                if False and _reason_for_stat(after_stat) is not None:'
expect_red \
  "post-read file type or hardlink change accepted" \
  tests/test_bundle_reader.py::test_post_read_file_type_or_link_change_is_not_accepted

new_case
replace_once \
  "src/ckp/bundle.py" \
  '                if _file_signature(after_stat) != opened_signature:' \
  '                if False and _file_signature(after_stat) != opened_signature:'
expect_red \
  "post-read fstat signature change accepted" \
  tests/test_bundle_reader.py::test_post_read_fstat_signature_change_is_not_accepted

new_case
replace_once \
  "src/ckp/bundle.py" \
  '            if _file_signature(namespace_stat) != opened_signature:' \
  '            if False and _file_signature(namespace_stat) != opened_signature:'
expect_red \
  "post-read namespace replacement accepted" \
  tests/test_bundle_reader.py::test_post_read_namespace_replacement_is_not_accepted

new_case
replace_once \
  "src/ckp/bundle.py" \
  '            if not self._lineage_is_stable(lineage):' \
  '            if False and not self._lineage_is_stable(lineage):'
expect_red \
  "post-read parent lineage replacement accepted" \
  tests/test_bundle_reader.py::test_parent_replacement_after_file_read_is_not_accepted

new_case
replace_once \
  "src/ckp/bundle.py" \
  '            if expected is not None and before_signature != expected:' \
  '            if False and expected is not None and before_signature != expected:'
expect_red \
  "capture accepts a stale member probe" \
  tests/test_bundle_reader.py::test_capture_rejects_a_stale_member_probe

new_case
replace_once \
  "src/ckp/bundle.py" \
  '            if root_identity != expected.root_identity:' \
  '            if False and root_identity != expected.root_identity:'
expect_red \
  "capture accepts a retargeted root symlink" \
  tests/test_bundle_reader.py::test_capture_rejects_a_retargeted_root_symlink

new_case
replace_once \
  "src/ckp/bundle.py" \
  '                readable=all(
                    item.refusal not in _BLOCKING_MEMBER_REFUSALS
                    for item in expected.members
                ),' \
  '                readable=True,'
expect_red \
  "refused candidates publish a healthy partial snapshot" \
  tests/test_bundle_reader.py::test_capture_marks_an_open_refusal_unreadable \
  tests/test_gateway.py::test_hardlink_refusal_never_publishes_a_healthy_partial_snapshot

echo "==> revision cache mutations"
new_case
replace_once \
  "src/ckp/bundle.py" \
  '                if before.token != after.token or live_commit != live_commit_after:' \
  '                if False and (before.token != after.token or live_commit != live_commit_after):'
expect_red \
  "cache publishes after before and after probes disagree" \
  tests/test_revision_cache.py::test_cache_does_not_publish_when_before_and_after_probes_disagree

new_case
replace_once \
  "src/ckp/bundle.py" \
  '        return (self._generation, snapshot_token)' \
  '        return (snapshot_token,)'
expect_red \
  "explicit cache invalidation becomes ineffective" \
  tests/test_revision_cache.py::test_explicit_invalidation_forces_a_rebuild

new_case
replace_once \
  "src/ckp/bundle.py" \
  '                if self._snapshot is not None and cache_key == self._cache_token:' \
  '                if self._snapshot is not None:'
expect_red \
  "changed bundle bytes return a stale cached revision" \
  tests/test_revision_cache.py::test_changed_bytes_are_revalidated_without_a_stale_revision

echo "==> production staleness flag mutations (#39)"
new_case
replace_once \
  "src/ckp/app.py" \
'        stale = (
            served_before is not None
            and served_before.index_revision != current.index_revision
        )' \
'        stale = False'
expect_red \
  "production stale flag removed -- resilience benchmark must not fall back to a self-comparator" \
  tests/test_app.py::test_revision_flags_stale_after_the_bundle_changes_underneath_it \
  tests/test_resilience_benchmark.py::test_stale_snapshot_scenario_passes_against_the_real_endpoint

new_case
replace_once \
  "src/ckp/bundle.py" \
'        Returns ``None`` before the first ``get`` call on this instance.
        """
        with self._lock:
            return self._snapshot' \
'        Returns ``None`` before the first ``get`` call on this instance.
        """
        with self._lock:
            return None'
expect_red \
  "SnapshotCache.peek() always reports nothing was ever served" \
  tests/test_revision_cache.py::test_peek_returns_the_last_snapshot_get_materialized \
  tests/test_app.py::test_revision_flags_stale_after_the_bundle_changes_underneath_it \
  tests/test_resilience_benchmark.py::test_stale_snapshot_scenario_passes_against_the_real_endpoint

echo "mutation: control plus $((CASE_NUMBER - 1)) C3 mutants all turned red"
