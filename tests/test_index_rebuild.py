"""Empty-volume rebuild determinism: the C5 acceptance in executable form.

``GOLDEN_*`` are recorded from the frozen implementation over the frozen
benchmark corpus. If either literal changes, every existing index and
snapshot silently invalidated -- bump ``INDEX_SCHEMA_VERSION`` (or the
embedding ``PROVIDER_VERSION``) instead of editing a literal in place.

The cross-seed subprocess check is the oracle that no code path leans on
builtin ``hash()`` or set/dict iteration order: two interpreters with
different ``PYTHONHASHSEED`` must derive the identical plan.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ckp.index import InMemoryVectorIndex, plan_rebuild
from index_fixtures import corpus_members, deterministic_stack, public_gate

GOLDEN_COMPOSED_REVISION = (
    "sha256:9a059cb25467503076b357f7b9cab04c1a93e9424493aadd14bf9d6296eed67d"
)
GOLDEN_PAYLOAD_DIGEST = (
    "sha256:25a27fd616eefd9538cd7280f2d7fb5ca19d4f2c3c06bac1587fde207a7314d7"
)
GOLDEN_BUNDLE_REVISION = (
    "sha256:9e595c3a83635eb64910be2e4296ea13ef3d305fa749e2c6cd62b9f62c424945"
)

_SUBPROCESS_SCRIPT = """
import json
import sys

sys.path.insert(0, {tests_dir!r})
from index_fixtures import corpus_members, deterministic_stack, public_gate

from ckp.index import plan_rebuild

plan = plan_rebuild(
    members=corpus_members(), stack=deterministic_stack(), gate=public_gate()
)
print(
    json.dumps(
        {{
            "composed": plan.composed_revision,
            "payload": plan.payload_digest,
            "bundle": plan.bundle_index_revision,
            "indexed": plan.indexed_count,
        }}
    )
)
"""


def test_same_commit_same_revision_and_digest() -> None:
    first = plan_rebuild(
        members=corpus_members(), stack=deterministic_stack(), gate=public_gate()
    )
    second = plan_rebuild(
        members=corpus_members(), stack=deterministic_stack(), gate=public_gate()
    )
    assert first.composed_revision == GOLDEN_COMPOSED_REVISION
    assert first.payload_digest == GOLDEN_PAYLOAD_DIGEST
    assert first.bundle_index_revision == GOLDEN_BUNDLE_REVISION
    assert second.composed_revision == first.composed_revision
    assert second.payload_digest == first.payload_digest


def test_rebuild_is_reproducible_across_hash_seeds() -> None:
    tests_dir = str(Path(__file__).resolve().parent)
    script = _SUBPROCESS_SCRIPT.format(tests_dir=tests_dir)
    outputs = []
    for seed in ("0", "1"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
            env=environment,
        )
        outputs.append(json.loads(completed.stdout))
    assert outputs[0] == outputs[1]
    assert outputs[0]["composed"] == GOLDEN_COMPOSED_REVISION
    assert outputs[0]["payload"] == GOLDEN_PAYLOAD_DIGEST


def test_two_empty_volumes_rebuild_identically() -> None:
    plan = plan_rebuild(
        members=corpus_members(), stack=deterministic_stack(), gate=public_gate()
    )
    left = InMemoryVectorIndex()
    right = InMemoryVectorIndex()
    left_report = left.rebuild(plan)
    right_report = right.rebuild(plan)
    assert left_report == right_report

    query = deterministic_stack().embedding.embed_query("tide caves mapping").values
    from ckp.privacy import PrivacyClass

    public = frozenset({PrivacyClass.PUBLIC})
    assert left.search(query, top_k=5, filter_privacy=public) == right.search(
        query, top_k=5, filter_privacy=public
    )


def test_a_content_change_changes_the_composed_revision() -> None:
    members = corpus_members()
    baseline = plan_rebuild(
        members=members, stack=deterministic_stack(), gate=public_gate()
    )
    from index_fixtures import make_member

    edited = tuple(
        make_member(
            member.relative_path,
            member.content.replace(b"espresso", b"ristretto"),
        )
        for member in members
    )
    changed = plan_rebuild(
        members=edited, stack=deterministic_stack(), gate=public_gate()
    )
    assert changed.composed_revision != baseline.composed_revision
    assert changed.payload_digest != baseline.payload_digest


def test_a_different_embedding_dimension_changes_the_composed_revision() -> None:
    baseline = plan_rebuild(
        members=corpus_members(), stack=deterministic_stack(), gate=public_gate()
    )
    other = plan_rebuild(
        members=corpus_members(),
        stack=deterministic_stack(dimension=128),
        gate=public_gate(),
    )
    assert other.composed_revision != baseline.composed_revision
