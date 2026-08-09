#!/usr/bin/env bash
# Lexical scorer / cutoff mutation matrix (issues #62 -> #66).
#
# Every constant in the gateway's BM25F-family scorer and two-regime
# relevance cutoff is distribution-derived (the derivations live in the
# constants' own docstring in src/ckp/gateway/service.py); this script is
# the tracked, rerunnable proof that each one is pinned by at least one
# test -- Codex's #68 round-1 review found the pins existed only as an
# untracked session-side self-check. Same harness as the other matrices:
# every mutant runs in a private copied tree, the source worktree is never
# edited, and a control mutation must turn red before the rest is trusted.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
SOURCE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
MUTATION_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ckp-scorer-mutations.XXXXXX")"
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

SERVICE="src/ckp/gateway/service.py"

echo "==> control mutation"
new_case
replace_once "$SERVICE" \
  '_CUTOFF_FLOOR_COVERAGE = 2' \
  '_CUTOFF_FLOOR_COVERAGE = 0'
expect_red \
  "control: coverage floor removed entirely" \
  tests/test_gateway.py::test_query_all_weak_matches_serve_nothing

echo "==> coverage floor and token dedup mutations"
new_case
replace_once "$SERVICE" \
  '_CUTOFF_FLOOR_COVERAGE = 2' \
  '_CUTOFF_FLOOR_COVERAGE = 1'
expect_red \
  "floor lowered to one matched token" \
  tests/test_gateway.py::test_query_all_weak_matches_serve_nothing

new_case
replace_once "$SERVICE" \
  '    return tuple(dict.fromkeys(_normalise(query).split()))' \
  '    return tuple(t for t in _normalise(query).split() if t)'
expect_red \
  "query-token dedup removed -- repeats buy floor coverage again" \
  tests/test_gateway.py::test_query_duplicate_query_tokens_do_not_inflate_the_floor

echo "==> saturation and length-normalization mutations"
new_case
replace_once "$SERVICE" \
  '_K1 = 3.0' \
  '_K1 = 0.5'
expect_red \
  "k1 collapsed -- scores fall back toward coverage counting" \
  tests/test_gateway.py::test_query_sparse_field_still_cuts_a_distant_runner_up \
  tests/test_gateway.py::test_query_length_normalized_frequency_outranks_a_padded_mention

new_case
replace_once "$SERVICE" \
  '_K1 = 3.0' \
  '_K1 = 48.0'
expect_red \
  "k1 exploded -- one dense token outweighs whole-query coverage" \
  tests/test_gateway.py::test_query_sparse_field_serves_the_runner_up_a_crowd_would_cut

new_case
replace_once "$SERVICE" \
  '            weighted_tf += weight * occurrences * average / len(field_text)' \
  '            weighted_tf += weight * occurrences'
expect_red \
  "length normalization removed -- padded notes outrank compact ones" \
  tests/test_gateway.py::test_query_is_lexical_deterministic_bounded_and_cited \
  tests/test_gateway.py::test_query_length_normalized_frequency_outranks_a_padded_mention

echo "==> two-regime boundary mutations"
new_case
replace_once "$SERVICE" \
  '_SPARSE_CANDIDATE_MAX = 2' \
  '_SPARSE_CANDIDATE_MAX = 3'
expect_red \
  "regime boundary raised -- three-candidate fields go lenient" \
  tests/test_gateway.py::test_query_sparse_field_serves_the_runner_up_a_crowd_would_cut

new_case
replace_once "$SERVICE" \
  '_SPARSE_CANDIDATE_MAX = 2' \
  '_SPARSE_CANDIDATE_MAX = 1'
expect_red \
  "regime boundary lowered -- two-candidate fields go strict" \
  tests/test_gateway.py::test_query_sparse_field_serves_the_runner_up_a_crowd_would_cut \
  tests/test_shadow_benchmark.py::test_both_engines_answer_the_frozen_corpus

echo "==> ratio bar mutations"
new_case
replace_once "$SERVICE" \
  '_CUTOFF_RATIO_SPARSE_DEN = 3' \
  '_CUTOFF_RATIO_SPARSE_DEN = 2'
expect_red \
  "sparse bar tightened to 1/2 -- q06's second answer dies in the frozen corpus" \
  tests/test_shadow_benchmark.py::test_both_engines_answer_the_frozen_corpus

new_case
replace_once "$SERVICE" \
  '_CUTOFF_RATIO_SPARSE_DEN = 3' \
  '_CUTOFF_RATIO_SPARSE_DEN = 5'
expect_red \
  "sparse bar loosened to 1/5 -- distant runner-ups ride along" \
  tests/test_gateway.py::test_query_sparse_field_still_cuts_a_distant_runner_up

new_case
replace_once "$SERVICE" \
  '_CUTOFF_RATIO_CROWDED_NUM = 3' \
  '_CUTOFF_RATIO_CROWDED_NUM = 2'
expect_red \
  "crowded bar loosened to 1/2 -- weak thirds survive a crowded field" \
  tests/test_gateway.py::test_query_crowded_field_keeps_a_close_second

new_case
replace_once "$SERVICE" \
  '_CUTOFF_RATIO_CROWDED_NUM = 3' \
  '_CUTOFF_RATIO_CROWDED_NUM = 9'
replace_once "$SERVICE" \
  '_CUTOFF_RATIO_CROWDED_DEN = 4' \
  '_CUTOFF_RATIO_CROWDED_DEN = 10'
expect_red \
  "crowded bar tightened to 9/10 -- the close second (r02's shape) dies" \
  tests/test_gateway.py::test_query_crowded_field_keeps_a_close_second

echo "mutation: control plus 11 scorer mutants all turned red"
