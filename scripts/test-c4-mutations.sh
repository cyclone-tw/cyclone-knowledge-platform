#!/usr/bin/env bash
# C4 embedding mutation matrix. Every mutant runs in an isolated TMPDIR copy;
# the issue worktree and any user checkout are never edited or cleaned.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$(CDPATH= cd -- "$(dirname -- "$PYTHON_BIN")" && pwd -P)/$(basename -- "$PYTHON_BIN")"
fi
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-c4-mutations.XXXXXX")"
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

C4_TESTS=(
  tests/test_embedding_models.py
  tests/test_embedding_provider.py
  tests/test_embedding_registry.py
  tests/test_embedding_neutrality.py
  tests/test_c4_contract.py
)

expect_green "pristine C4 embedding contract" "${C4_TESTS[@]}"

echo "==> determinism"
new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '            digest = hashlib.sha256(_TOKEN_DOMAIN + token.encode("utf-8")).digest()
            counts[int.from_bytes(digest[:8], "big") % dimension] += 1' \
  '            counts[hash(token) % dimension] += 1'
expect_red \
  "bucket derived from the salted built-in hash()" \
  tests/test_embedding_provider.py::test_vectors_are_identical_across_processes_and_hash_seeds

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        norm = math.sqrt(sum(count * count for count in counts))' \
  '        norm = 1.0  # mutant drops normalization'
expect_red \
  "vectors left unnormalized while the descriptor claims otherwise" \
  tests/test_embedding_provider.py::test_the_in_process_provider_matches_the_frozen_golden_digests \
  tests/test_embedding_models.py::test_a_normalized_provider_cannot_return_a_non_unit_vector

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        return EmbeddingVector(
            descriptor=self._descriptor,
            values=self._embed_one(text),
        )' \
  '        return EmbeddingVector(
            descriptor=self._descriptor,
            values=self._embed_one("query: " + text),
        )'
expect_red \
  "query path diverges from the document path" \
  tests/test_embedding_provider.py::test_query_and_document_paths_are_bit_identical

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '_TOKEN = re.compile(r"[a-z0-9]+|[^\x00-\x7f" + re.escape(_IGNORED_NON_ASCII) + "]")' \
  '_TOKEN = re.compile(r"[a-z0-9]+|[^\W_]")'
expect_red \
  "tokenizer back on a Unicode-database character class" \
  tests/test_embedding_provider.py::test_the_tokenizer_reads_no_unicode_database

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '    return _TOKEN.findall(normalized.translate(_ASCII_FOLD))' \
  '    return _TOKEN.findall(normalized.casefold())'
expect_red \
  "case folding back on the interpreter Unicode tables" \
  tests/test_embedding_provider.py::test_the_tokenizer_reads_no_unicode_database

echo "==> batch and input gates"
new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '            vectors=tuple(self._embed_one(text) for text in texts),' \
  '            vectors=tuple(self._embed_one(text) for text in list(texts)[:-1]),'
expect_red \
  "batch returns fewer vectors than it was given texts" \
  tests/test_embedding_provider.py::test_a_batch_preserves_length_and_order

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        if isinstance(texts, str) or not isinstance(texts, Sequence):' \
  '        if False and isinstance(texts, str):'
expect_red \
  "a bare string accepted as a batch of documents" \
  tests/test_embedding_provider.py::test_a_bare_string_is_not_a_batch_of_documents

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        if not tokens:
            # Non-empty but with nothing indexable in it (punctuation only).
            # Reporting a zero vector would be inventing a value.
            raise EmbeddingRefusal(EmbeddingErrorCode.TEXT_EMPTY)' \
  '        if not tokens:
            return tuple(0.0 for _ in range(self._descriptor.dimension or 0))'
expect_red \
  "untokenizable text answered with a fabricated zero vector" \
  tests/test_embedding_provider.py::test_empty_untokenizable_and_non_text_inputs_fail_closed

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    if not value.strip():' \
  '    if False and not value.strip():'
expect_red \
  "empty text accepted by the single input gate" \
  tests/test_embedding_provider.py::test_require_text_is_the_single_input_gate \
  tests/test_embedding_provider.py::test_empty_untokenizable_and_non_text_inputs_fail_closed

echo "==> descriptor and revision"
new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '        if self.kind is ProviderKind.EMBEDDING and self.dimension is None:
            raise ValueError("an embedding provider must declare a dimension")' \
  '        if self.kind is ProviderKind.EMBEDDING and self.dimension is None:
            pass  # mutant lets an embedding provider omit its dimension'
expect_red \
  "embedding descriptor accepted without a dimension" \
  tests/test_embedding_models.py::test_an_embedding_descriptor_must_declare_a_dimension

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '        descriptor.metric.value.encode("ascii"),' \
  '        b"",  # mutant drops metric from the revision'
expect_red \
  "metric dropped from the provider revision" \
  tests/test_embedding_models.py::test_provider_revision_reads_every_descriptor_field

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '        _flag(descriptor.semantic),' \
  '        b"",  # mutant drops the honesty flag from the revision'
expect_red \
  "semantic flag dropped from the provider revision" \
  tests/test_embedding_models.py::test_provider_revision_changes_when_any_varying_field_changes \
  tests/test_embedding_models.py::test_provider_revision_reads_every_descriptor_field

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    if descriptor.dimension is None or len(values) != descriptor.dimension:
        raise ValueError("vector length must equal the declared dimension")' \
  '    if descriptor.dimension is None or len(values) != descriptor.dimension:
        pass  # mutant accepts a vector of any length'
expect_red \
  "vector length no longer has to match the declared dimension" \
  tests/test_embedding_models.py::test_a_vector_must_match_the_declared_dimension_and_be_finite

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    if descriptor.normalized:' \
  '    if False and descriptor.normalized:'
expect_red \
  "unit-vector check removed for a normalized provider" \
  tests/test_embedding_models.py::test_a_normalized_provider_cannot_return_a_non_unit_vector

echo "==> rerank total order"
new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '            if key_previous >= key_item:' \
  '            if False and key_previous >= key_item:'
expect_red \
  "result order no longer has to be the frozen total order" \
  tests/test_embedding_models.py::test_rerank_result_rejects_ascending_scores \
  tests/test_embedding_models.py::test_rerank_result_rejects_a_tie_broken_against_candidate_id

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    """One placed candidate: id, score, rank. Deliberately no text field."""

    model_config = ConfigDict(extra="forbid", frozen=True)' \
  '    """One placed candidate: id, score, rank. Deliberately no text field."""

    model_config = ConfigDict(extra="allow", frozen=True)'
expect_red \
  "rerank output can carry candidate text again" \
  tests/test_embedding_models.py::test_a_ranked_candidate_has_no_text_field

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        scored.sort(key=lambda item: (-item[1], item[0]))' \
  '        scored.sort(key=lambda item: -item[1])'
expect_red \
  "ties broken by caller input order instead of candidate id" \
  tests/test_embedding_provider.py::test_identical_texts_tie_break_on_candidate_id_not_input_order

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '                for rank, (identifier, score) in enumerate(scored[:top_k])' \
  '                for rank, (identifier, score) in enumerate(scored)'
expect_red \
  "top_k ignored by the reranker" \
  tests/test_embedding_provider.py::test_rerank_orders_by_score_and_truncates_to_top_k

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:' \
  '        if False and top_k < 1:'
expect_red \
  "top_k validation removed" \
  tests/test_embedding_provider.py::test_rerank_fails_closed_on_top_k_duplicates_and_bad_candidates

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        if len(set(identifiers)) != len(identifiers):' \
  '        if False and len(set(identifiers)) != len(identifiers):'
expect_red \
  "duplicate candidate ids accepted by the reranker" \
  tests/test_embedding_provider.py::test_rerank_fails_closed_on_top_k_duplicates_and_bad_candidates

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        if not descriptor.normalized:' \
  '        if False and not descriptor.normalized:'
expect_red \
  "cosine reranker composed over a non-unit embedder" \
  tests/test_embedding_provider.py::test_a_cosine_reranker_refuses_an_embedder_that_is_not_normalized

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    if descriptor.contract_version != EMBEDDING_CONTRACT:
        raise ValueError("result descriptor must declare this contract version")' \
  '    if False and descriptor.contract_version != EMBEDDING_CONTRACT:
        raise ValueError("result descriptor must declare this contract version")'
expect_red \
  "a result can be stamped with a foreign contract version" \
  tests/test_embedding_models.py::test_a_result_must_carry_this_contract_version

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    if descriptor.kind is not kind:
        raise ValueError(f"result descriptor must be of kind {kind.value}")' \
  '    if False and descriptor.kind is not kind:
        raise ValueError(f"result descriptor must be of kind {kind.value}")'
expect_red \
  "a result can be stamped with the other kind of descriptor" \
  tests/test_embedding_models.py::test_a_result_must_carry_a_descriptor_of_its_own_kind

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    def require_provider_version_shape(cls, value: str) -> str:
        if _PROVIDER_VERSION.fullmatch(value) is None:' \
  '    def require_provider_version_shape(cls, value: str) -> str:
        if False and _PROVIDER_VERSION.fullmatch(value) is None:'
expect_red \
  "provider_version shape guard removed" \
  tests/test_embedding_models.py::test_descriptor_rejects_a_provider_version_that_is_not_the_frozen_shape

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '    candidate_id: str = Field(min_length=1, max_length=256)
    text: str

    @field_validator("candidate_id")
    @classmethod
    def require_candidate_id_shape(cls, value: str) -> str:
        if _CANDIDATE_ID.fullmatch(value) is None:' \
  '    candidate_id: str = Field(min_length=1, max_length=256)
    text: str

    @field_validator("candidate_id")
    @classmethod
    def require_candidate_id_shape(cls, value: str) -> str:
        if False and _CANDIDATE_ID.fullmatch(value) is None:'
expect_red \
  "candidate_id shape guard removed on the rerank input" \
  tests/test_embedding_models.py::test_candidate_ids_must_match_the_frozen_id_pattern

echo "==> the offline fence holds through composition"
new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '        descriptor = require_provider(embedder, ProviderKind.EMBEDDING)' \
  '        descriptor = embedder.descriptor  # mutant skips embedder admission'
expect_red \
  "reranker admits any embedder: cloud, non-deterministic, or not one at all" \
  tests/test_embedding_provider.py::test_a_cosine_reranker_refuses_an_offline_violating_embedder \
  tests/test_embedding_provider.py::test_a_cosine_reranker_refuses_an_embedder_that_is_not_one

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '            deterministic=descriptor.deterministic,
            requires_network=descriptor.requires_network,' \
  '            deterministic=True,
            requires_network=False,'
expect_red \
  "reranker asserts offline flags it cannot back instead of inheriting them" \
  tests/test_embedding_provider.py::test_a_reranker_reports_the_flags_of_the_embedder_it_wraps

new_case
replace_once \
  "src/ckp/embedding/provider.py" \
  '        if not callable(getattr(provider, method, None)):' \
  '        if False and not callable(getattr(provider, method, None)):'
expect_red \
  "a provider with non-callable methods registers cleanly" \
  tests/test_embedding_registry.py::test_a_provider_whose_methods_are_not_callable_cannot_be_registered \
  tests/test_embedding_provider.py::test_a_cosine_reranker_refuses_an_embedder_that_is_not_one

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '            deterministic=descriptor.deterministic,
            requires_network=descriptor.requires_network,' \
  '            deterministic=embedder.descriptor.deterministic,
            requires_network=embedder.descriptor.requires_network,'
expect_red \
  "flags re-read from the property instead of the admitted descriptor" \
  tests/test_embedding_provider.py::test_admission_reads_the_descriptor_once_and_keeps_that_answer

new_case
replace_once \
  "src/ckp/embedding/models.py" \
  '        value.encode("utf-8")' \
  '        pass  # mutant lets an unencodable string through'
expect_red \
  "a lone surrogate escapes as UnicodeEncodeError instead of a coded refusal" \
  tests/test_embedding_provider.py::test_text_that_cannot_be_encoded_fails_closed_with_a_code

new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  '    normalized = unicodedata.normalize("NFC", text)' \
  '    normalized = text  # mutant drops canonical normalization'
expect_red \
  "NFD text vectorizes differently from the same text in NFC" \
  tests/test_embedding_provider.py::test_composed_and_decomposed_text_embed_identically

new_case
replace_once \
  "src/ckp/embedding/composition.py" \
  '        embedding_descriptor = require_provider(self.embedding, ProviderKind.EMBEDDING)' \
  '        embedding_descriptor = self.embedding.descriptor  # mutant trusts the caller'
expect_red \
  "a stack can be assembled around an unadmitted provider" \
  tests/test_embedding_registry.py::test_a_stack_validates_itself_rather_than_trusting_its_builder

new_case
replace_once \
  "src/ckp/embedding/composition.py" \
  '            if revision != compute_provider_revision(descriptor):' \
  '            if False and revision != compute_provider_revision(descriptor):'
expect_red \
  "a stack can carry a revision its own descriptor cannot reproduce" \
  tests/test_embedding_registry.py::test_a_stack_validates_itself_rather_than_trusting_its_builder

echo "==> registry fail-closed boundary"
new_case
replace_once \
  "src/ckp/embedding/registry.py" \
  '        if not isinstance(name, str) or name not in slot:
            # No default, no nearest match, no fallback.
            raise EmbeddingRefusal(EmbeddingErrorCode.PROVIDER_UNKNOWN)' \
  '        if not isinstance(name, str) or name not in slot:
            return next(iter(slot.values()))'
expect_red \
  "unknown provider name silently falls back to whatever is registered" \
  tests/test_embedding_registry.py::test_an_unknown_name_refuses_instead_of_falling_back

new_case
replace_once \
  "src/ckp/embedding/provider.py" \
  '    if descriptor.requires_network:' \
  '    if False and descriptor.requires_network:'
expect_red \
  "a network-requiring provider can be registered in Phase 3" \
  tests/test_embedding_registry.py::test_a_network_provider_cannot_be_registered_in_phase_3

new_case
replace_once \
  "src/ckp/embedding/provider.py" \
  '    if not descriptor.deterministic:' \
  '    if False and not descriptor.deterministic:'
expect_red \
  "a non-deterministic provider can back a recomputable revision" \
  tests/test_embedding_registry.py::test_a_nondeterministic_provider_cannot_be_registered

new_case
replace_once \
  "src/ckp/embedding/provider.py" \
  '    if descriptor.contract_version != EMBEDDING_CONTRACT:' \
  '    if False and descriptor.contract_version != EMBEDDING_CONTRACT:'
expect_red \
  "a foreign contract version can be registered" \
  tests/test_embedding_registry.py::test_a_foreign_contract_version_cannot_be_registered

new_case
replace_once \
  "src/ckp/embedding/provider.py" \
  '    if descriptor.kind is not kind:' \
  '    if False and descriptor.kind is not kind:'
expect_red \
  "a provider can be registered into the wrong slot" \
  tests/test_embedding_registry.py::test_a_provider_cannot_be_registered_into_the_wrong_slot

new_case
replace_once \
  "src/ckp/embedding/registry.py" \
  '        if name in slot:' \
  '        if False and name in slot:'
expect_red \
  "re-registering a name silently overrides the first provider" \
  tests/test_embedding_registry.py::test_registering_the_same_name_twice_refuses_rather_than_overrides

echo "==> scope tripwires (AGENTS.md §9: the guards are subjects under test too)"
new_case
replace_once \
  "src/ckp/embedding/hashing.py" \
  'import hashlib
import math
import re' \
  'import hashlib
import math
import os
import re'
expect_red \
  "an out-of-scope stdlib import slipped into the package" \
  tests/test_c4_contract.py::test_the_package_imports_nothing_that_could_reach_a_network_or_a_secret

new_case
replace_once \
  "src/ckp/embedding/__init__.py" \
  '__all__ = [
    "COSINE_RERANKER_ID",' \
  'DEFAULT_REGISTRY = ProviderRegistry()

__all__ = [
    "DEFAULT_REGISTRY",
    "COSINE_RERANKER_ID",'
expect_red \
  "the package ships a prebuilt registry nobody chose" \
  tests/test_c4_contract.py::test_the_package_exports_no_prebuilt_provider_or_registry

new_case
replace_once \
  "src/ckp/embedding/composition.py" \
  'def build_c4_deterministic_stack(
    *,
    dimension: int,' \
  'def build_c4_deterministic_stack(
    *,
    dimension: int = 384,'
expect_red \
  "a permissive default dimension appeared in the builder" \
  tests/test_c4_contract.py::test_the_c4_builders_have_no_permissive_defaults

new_case
replace_once \
  "src/ckp/embedding/registry.py" \
  'from ckp.embedding.models import ProviderKind' \
  'from ckp.embedding.hashing import HASH_EMBEDDING_ID  # noqa: F401
from ckp.embedding.models import ProviderKind'
expect_red \
  "the abstraction layer learned the name of a concrete provider" \
  tests/test_embedding_neutrality.py::test_the_abstraction_layer_does_not_know_any_concrete_provider

echo "C4 mutation matrix complete: $CASE_NUMBER mutants, all red."
rm -rf "$MUTATION_ROOT"
