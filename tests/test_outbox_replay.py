"""Red line R3: at-least-once replay yields exactly one logical commit.

The oracle throughout is Git itself: the number of published operation refs
after any sequence of crashes, duplicates, and retries must be exactly one,
and a failed replay must leave a safely replayable or stably quarantined
record -- never a second commit, never a guessed rewrite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ckp.outbox.errors import OutboxErrorCode, OutboxRefusal
from ckp.outbox.models import OutboxRejectReceipt, OutboxState
from outbox_fixtures import (
    MovingBaseValidator,
    assert_no_plaintext_at_rest,
    build_outbox_harness,
    enqueue,
    operation_commits,
    outbox_request,
)
from writer_fixtures import (
    make_synthetic_repository,
    move_main,
    repository_snapshot,
    run_git,
)


def _record_path(harness) -> Path:
    files = list((harness.store_root / "records").iterdir())
    assert len(files) == 1
    return files[0]


def _single_record(harness):
    records = harness.store.scan()
    assert len(records) == 1
    return records[0][1]


def test_enqueue_then_replay_commits_exactly_once_with_evidence(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    accepted = enqueue(harness.service)
    assert not isinstance(accepted, OutboxRejectReceipt)

    receipts = harness.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.COMMITTED]
    committed = receipts[0]
    assert committed.content_hash == accepted.content_hash
    assert committed.result_commit is not None

    assert len(operation_commits(harness.repository)) == 1
    shown = run_git(
        harness.repository.root,
        "show",
        f"{committed.result_commit}:{committed.target_path}",
        text=False,
    )
    assert isinstance(shown, bytes)
    assert b"SYNTHETIC-OUTBOX-BODY-NEEDLE" in shown

    record = _single_record(harness)
    assert record.state is OutboxState.COMMITTED
    assert record.envelope is None
    assert_no_plaintext_at_rest(harness.store_root)
    # The shared checkout stays pristine; publication is refs, not main.
    branch, _head, status, staged = repository_snapshot(harness.repository)
    assert branch == "main" and status == b"" and staged == b""


def test_crash_between_commit_and_rewrite_never_doubles_the_commit(
    tmp_path: Path,
) -> None:
    """The at-least-once window, replayed on purpose."""
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    snapshot = _record_path(harness).read_bytes()

    first = harness.service.replay_pending()
    assert [receipt.state for receipt in first] == [OutboxState.COMMITTED]
    result_commit = first[0].result_commit

    # Simulate the crash: the Git commit is published but the record rewrite
    # was lost, so the durable state still says queued-with-payload.
    _record_path(harness).write_bytes(snapshot)

    second = harness.service.replay_pending()
    assert [receipt.state for receipt in second] == [OutboxState.COMMITTED]
    assert second[0].result_commit == result_commit
    assert len(operation_commits(harness.repository)) == 1
    assert _single_record(harness).state is OutboxState.COMMITTED


def test_duplicate_enqueue_dedups_and_different_binding_conflicts(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    first = enqueue(harness.service)
    assert not isinstance(first, OutboxRejectReceipt)

    harness.clock.advance(10)
    duplicate = enqueue(harness.service)
    assert not isinstance(duplicate, OutboxRejectReceipt)
    # The duplicate returns the original binding, not a second record.
    assert duplicate.created_at == first.created_at
    assert duplicate.content_hash == first.content_hash
    assert len(list((harness.store_root / "records").iterdir())) == 1

    reused_key = enqueue(
        harness.service,
        outbox_request(
            operation_id="synthetic.outbox-2",
            body="# Synthetic Outbox Note\n\nDifferent synthetic content.\n",
        ),
    )
    assert isinstance(reused_key, OutboxRejectReceipt)
    assert reused_key.code == OutboxErrorCode.BINDING_CONFLICT.value

    same_operation_new_request = enqueue(
        harness.service,
        outbox_request(
            idempotency_key="synthetic-outbox-key-0002",
            body="# Synthetic Outbox Note\n\nDifferent synthetic content.\n",
        ),
    )
    assert isinstance(same_operation_new_request, OutboxRejectReceipt)
    assert same_operation_new_request.code == OutboxErrorCode.BINDING_CONFLICT.value

    # Both index keys agree here, so only the service-level identity binding
    # can tell a rewritten request from a genuine duplicate.
    same_keys_rewritten_content = enqueue(
        harness.service,
        outbox_request(
            body="# Synthetic Outbox Note\n\nRewritten synthetic content.\n",
        ),
    )
    assert isinstance(same_keys_rewritten_content, OutboxRejectReceipt)
    assert same_keys_rewritten_content.code == OutboxErrorCode.BINDING_CONFLICT.value
    assert len(list((harness.store_root / "records").iterdir())) == 1


def test_concurrent_consumers_have_exactly_one_leader(tmp_path: Path) -> None:
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)
    with harness.store.consumer_leader():
        with pytest.raises(OutboxRefusal) as refusal:
            harness.service.replay_pending()
        assert refusal.value.code is OutboxErrorCode.LEASE_UNAVAILABLE
    # Once the competing consumer is gone, replay proceeds normally.
    receipts = harness.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.COMMITTED]
    assert len(operation_commits(harness.repository)) == 1


def test_a_live_lease_is_respected_and_recovered_after_expiry(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path, lease_seconds=60.0)
    enqueue(harness.service)
    record = _single_record(harness)
    stale_holder = record.model_copy(
        update={
            "state": OutboxState.LEASED,
            "lease_until": "2026-08-04T10:21:30Z",
        }
    )
    harness.store.rewrite(stale_holder)

    held = harness.service.replay_pending()
    assert held == []
    assert _single_record(harness).state is OutboxState.LEASED

    harness.clock.advance(61)
    recovered = harness.service.replay_pending()
    assert [receipt.state for receipt in recovered] == [OutboxState.COMMITTED]
    assert len(operation_commits(harness.repository)) == 1


def test_transient_base_movement_retries_bounded_then_commits_once(
    tmp_path: Path,
) -> None:
    repository = make_synthetic_repository(tmp_path)
    mover = MovingBaseValidator(repository, times=3)
    harness = build_outbox_harness(
        tmp_path,
        repository=repository,
        profile=mover,
        max_transient_attempts=2,
    )
    enqueue(harness.service)

    first = harness.service.replay_pending()
    assert [receipt.state for receipt in first] == [OutboxState.QUEUED]
    record = _single_record(harness)
    assert record.attempts == 1
    assert record.last_code == "writer/v1/base-moved"
    assert operation_commits(harness.repository) == ()

    second = harness.service.replay_pending()
    assert [receipt.state for receipt in second] == [OutboxState.COMMITTED]
    assert len(operation_commits(harness.repository)) == 1


def test_exhausted_transient_retries_quarantine_instead_of_looping(
    tmp_path: Path,
) -> None:
    repository = make_synthetic_repository(tmp_path)
    mover = MovingBaseValidator(repository, times=99)
    harness = build_outbox_harness(
        tmp_path,
        repository=repository,
        profile=mover,
        max_transient_attempts=1,
    )
    enqueue(harness.service)

    first = harness.service.replay_pending()
    assert [receipt.state for receipt in first] == [OutboxState.QUEUED]
    second = harness.service.replay_pending()
    assert [receipt.state for receipt in second] == [OutboxState.QUARANTINED]
    assert second[0].code == OutboxErrorCode.REPLAY_QUARANTINED
    record = _single_record(harness)
    assert record.state is OutboxState.QUARANTINED
    assert record.envelope is None
    assert operation_commits(harness.repository) == ()


def test_replay_after_synthetic_migration_commits_on_the_new_base(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    accepted = enqueue(harness.service)
    assert not isinstance(accepted, OutboxRejectReceipt)
    migrated_base = move_main(harness.repository)
    assert migrated_base != accepted.enqueue_base_commit

    receipts = harness.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.COMMITTED]
    assert receipts[0].base_commit == migrated_base
    # The record keeps the original base as audit evidence.
    record = _single_record(harness)
    assert record.enqueue_base_commit == accepted.enqueue_base_commit
    assert len(operation_commits(harness.repository)) == 1


def test_incompatible_migrated_base_is_a_stable_conflict_not_a_rewrite(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    enqueue(harness.service)

    target = "Core/_inbox/c7-synthetic/synthetic-outbox-note.md"
    occupied = harness.repository.root / target
    occupied.write_text("# Pre-existing migrated note\n", encoding="utf-8")
    run_git(harness.repository.root, "add", "--", target)
    run_git(
        harness.repository.root,
        "commit",
        "-m",
        "synthetic migration occupies the target",
    )

    receipts = harness.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.CONFLICT]
    assert receipts[0].code == OutboxErrorCode.REPLAY_CONFLICT
    record = _single_record(harness)
    assert record.state is OutboxState.CONFLICT
    assert record.envelope is None
    assert record.last_code == "writer/v1/path-conflict"
    # The occupying note was never rewritten or guessed around.
    assert occupied.read_text(encoding="utf-8") == "# Pre-existing migrated note\n"
    assert operation_commits(harness.repository) == ()


def test_expired_records_fail_closed_and_purge_their_payload(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path, retention_seconds=3600.0)
    enqueue(harness.service)
    harness.clock.advance(3601)

    receipts = harness.service.replay_pending()
    assert [receipt.state for receipt in receipts] == [OutboxState.EXPIRED]
    assert receipts[0].code == OutboxErrorCode.RECORD_EXPIRED
    record = _single_record(harness)
    assert record.state is OutboxState.EXPIRED
    assert record.envelope is None
    assert operation_commits(harness.repository) == ()
    assert_no_plaintext_at_rest(harness.store_root)

    # Terminal records never replay again.
    assert harness.service.replay_pending() == []
