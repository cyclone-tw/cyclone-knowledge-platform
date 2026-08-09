# AGENTS.md — cyclone-knowledge-platform

Authoritative agent rules for this repo. Read this before touching any file.

Upstream rule sources, in precedence order when they conflict:

1. This file (most specific).
2. `cyclone-wiki` `Core/project-cyclone-okf-knowledge-contract.md` — the OKF
   contract. This repo is Phase 3 of it.
3. `cyclone-agent-config` `shared/shared-agent-rules.md` §5.1–5.5, §6.3.
4. The global `AGENTS.md` / `CLAUDE.md` in the agent's home config.

Report any conflict you find instead of silently picking one.

## 1. What this repo owns

Per contract §4 (Repo、issue 與 Agent ownership):

| Owns | Does **not** own |
| --- | --- |
| Gateway API and MCP schema | Cyclone Profile schema (owned by `cyclone-wiki`) |
| Writer, Indexer, durable outbox | Wiki Markdown content |
| Qdrant runtime, snapshots, rebuild | Dashboard UI components |
| Auth scope, privacy prefilter, context packer | Per-agent live runtime state |
| Docker / portable packaging, config layering | Hermes / OpenAB runtime deployment |

Consumers pin a version of this repo's API schema and run a compatibility
test. Nobody copies a locally editable duplicate of the schema.

**This repo never contains Wiki content.** Fixtures are synthetic or public.
No note bodies, no student data, no adult private meeting notes.

## 2. Issue-first

Global rule — canonical text: `cyclone-agent-config`
`shared/shared-agent-rules.md` §5.2 (issue-first ship pipeline) and §5.3
(acceptance task lists ticked as work lands, PR–issue linkage, closing
keywords) plus the global rules' GitHub Issue First section (English
`feat:`/`fix:`/`ops:`/`docs:` titles, Traditional Chinese bodies). Repo
additions that bind here:

- Issue before feature / behavior / deployment / automation / runtime
  configuration changes — and before changes to this file, CI, or the review
  contract.
- PRs always reference the issue: `Closes #NN` on the closing PR, `Ref #NN`
  or `Part of #NN` on a staged one; staged commit messages never use closing
  keywords.

## 3. Worktree isolation

Global rule — canonical text: shared-agent-rules §5.4 (never `git checkout` /
`git switch` in the shared checkout; per-task ephemeral worktree; read-only
git operations unrestricted; cleanup with `git worktree remove`). Repo
specifics and additions that bind here:

- Worktree `~/Cyclone-System/worktrees/ckp-<issue>-<slug>`, branch
  `<type>/<issue>-<slug>`, base `origin/main`.
- Never leave staged-but-uncommitted changes in the shared main checkout —
  another agent's `git commit` will carry your index away. Stage and commit
  in one atomic invocation, or work in a worktree.
- Record the base commit, `git status`, and active worktrees before starting.
  Never stash, reset, clean, prune, or move another agent's worktree. If a
  file you need is already being modified by another agent, stop at that
  child-issue boundary and pick non-overlapping work.
- One PR = one story. Schema, migration, UI and runtime deploy never share a
  PR.

## 4. Review contract

Global rule — canonical text: shared-agent-rules §5.1.1 (reviews only through
the sanctioned wrappers) and §5.2 (pairing table, machine-readable PR markers,
attestation merge gate, auto-merge thresholds, stop-and-ask cases, hard-stop
list). Pinned clauses and repo defaults that bind here (guarded by
`tests/test_repo_conventions.py`):

- **Coder ≠ Reviewer.** Codex is the primary reviewer for Claude Code; there
  is no fallback reviewer tier (Cursor left the review chain 2026-07-27).
- Codex reviews go only through cyclone-agent-config
  `scripts/codex-review.sh`; Codex is single-instance — run reviews
  sequentially, never in parallel.
- The review prompt passes **file paths**, not pasted diffs, and must require
  a literal `VERDICT: approved | nits-only | changes-requested` line.
- Every review artifact carries the machine-readable trailer block, including
  `Review-Status:` and `Reviewed-Commit:` bound to the PR HEAD under review.
- Merge automatically — **do not ask permission to merge** — once the review
  is `approved`, or `nits-only` with CI green.
- Stop and ask a human only when: review unavailable (default time limit
  30 minutes), `Review-Round` exceeded **2** without reaching `approved` or
  `nits-only`, or a hard stop (§5). When stopping, use: 30-second background,
  one single question, the cost of each option, and your recommendation.

## 5. Hard stops

Never proceed, never automate around these:

- Secrets, tokens, or `.env` content written into the repo or leaked.
- Student-identifiable personal data entering git or any cloud service.
- Force push.
- Destructive production operations.
- A coder self-approving and auto-merging its own work.

## 6. Phase 3 red lines (contract §3, verbatim intent)

These are product invariants, not style preferences. Each one names the issue
that owns its enforcement — see the platform Epic (#1).

1. **Privacy routing completes before enqueue.** No item enters any queue with
   its privacy class still undetermined. Deferring classification to the
   dequeue side is forbidden.
2. **Raw or identifiable `student-private` never enters the generic outbox —
   encryption does not make it acceptable.** It may only go to an explicitly
   approved local-only restricted store.
3. **An outbox that has not passed its replay test may not carry the migration
   freeze.** If replay is unproven at Phase 5, the fallback is a full hard
   freeze that rejects writes immediately — never a pretend-accepting queue.
4. **Production direct-formal templates stay closed for all of Phase 3.**
5. **The migration manifest is frozen** (baseline `bbd2568`, freeze anchor
   `7278252`). Nothing in this repo re-mints UUIDs or writes to `reports/okf/`
   in `cyclone-wiki`.

## 7. Portability

Contract Phase 8: this package must cold-rebuild on Mac mini and then on Mac
Studio without source edits.

- **No absolute host paths in tracked source** — no `/Users/...`, no
  `/home/...`. Host-specific locations arrive through config or environment
  only. `tests/test_portability.py` enforces this; do not weaken it.
- Host-specific data lives in volumes or a secret store, never in the repo.
- Moving hosts changes endpoint and credential binding only — never the
  bundle, Profile, API, or agent skills.

## 8. Honest metadata

Inherited from the Wiki side of the contract and it applies to computed
values here too:

- When evidence for a field does not exist, report it as unknown (`null`).
  Never fabricate a plausible value to make a field look populated.
- Derived revision values must be reproducible from the same inputs. A value
  that cannot be recomputed from its commit is a rollback trigger
  (contract §5.5).

## 9. Guard self-verification

After adding any guard, assertion, or validation, immediately mutation-test it
with three questions:

1. Remove the guard — does a test go red?
2. Change the subject under test to something **compliant-looking** — does the
   guard still let it through?
3. Does the same root cause still exist at the other read/write points on the
   same data path?

A new guard is a new subject under test, not a finish line.

## 10. Dev commands

```bash
python3 -m pip install -e ".[dev]"   # editable install with dev extras
python3 -m pytest                    # tests
python3 -m ruff check .              # lint
python3 -m ruff format --check .     # format check
```

CI (`.github/workflows/ci.yml`) runs lint, tests, and — once the walking
skeleton lands — a container build plus an endpoint smoke test. CI must be
green before merge.
