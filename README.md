# cyclone-knowledge-platform

Self-hosted knowledge platform for the Cyclone-Wiki OKF migration.
Phase 3 of `cyclone-tw/cyclone-wiki#142`.

**Agents: read [`AGENTS.md`](AGENTS.md) first.**

## Current status (2026-10-15)

**Not running anywhere, and no longer being deployed.** No Mac mini process,
LaunchAgent or checkout exists, and Cyclone-Dashboard still reads its local
`export`. The last code change is `bc7dfe0` (2026-08-17). The Gateway cutover
(#78) and the Writer action contract (#79) are closed as not planned.

The repo's role is now **reference material for Wiki Seeker's retrieval
evaluation** (`cyclone-tw/cyclone-wiki-seeker#54`). Seeker took the method
and the scoring idea, reimplemented them in Rust, and measures them against
its own verified question set. No Python from here runs in production.

| Here | What Seeker took | Seeker counterpart |
| --- | --- | --- |
| `benchmarks/questions.py` | path-level answer keys, hand-checked real questions, synthetic vs real provenance | `evals/retrieval-baseline-v1.json` (33 cases, seven categories, SHA-256 per answer) |
| `benchmarks/shadow.py`, `benchmarks/gate.py` | same questions against each engine, thresholds written before the run | `eval-retrieval` CLI, plus a decision rule written into the issue before scoring |
| `benchmarks/token_cost.py` | context-size proxy | not ported |
| `src/ckp/gateway/service.py` (`_K1`, `_FIELD_WEIGHTS`, relevance cutoff) | BM25F field weights 4/2/2/1, k1 = 3 saturation, length normalization, two-stage cutoff | `src/ranking.rs`, behind `[wiki.ranking] lexical = "bm25f"`; off by default |

Differences worth knowing before reusing anything here:

- The Gateway scorer splits on whitespace only. Unspaced Chinese becomes one
  token, so it rarely matches. Seeker added a CJK bigram switch for this.
- A hit here means "every expected path was returned". Seeker reports
  hit@1/3/k, recall@k and MRR per question and per category.
- The real question set here has six questions taken from filenames and
  H1s, not natural questions.

The sections below describe the platform as it was built. Keep them as
history; they are not instructions for a live system.

## What this is

A portable, self-hosted knowledge service over the Cyclone-Wiki bundle:
Catalog builder, Knowledge Gateway (read), Writer (write), durable outbox,
vector index, auth and privacy prefilter — packaged so it cold-rebuilds on a
Mac mini today and a Mac Studio tomorrow without source edits.

It owns the **API and MCP schema**. `cyclone-wiki` owns the **Cyclone
Profile** — the metadata schema of the notes themselves. Neither repo keeps a
private editable copy of the other's contract.

## What this is not

- Not a copy of the Wiki. No note content lives here; fixtures are synthetic.
- Not the Dashboard. UI components stay in `Cyclone-Dashboard`.
- Not a Hermes/OpenAB deployment target. Those integrate in Phase 7.

## Status as built

Phase 3, through C8: Catalog and Gateway read, auth scope and context packing,
the Writer contract over synthetic Git fixtures with a local-only Cyclone-Wiki
Core inbox bridge, and a durable encrypted outbox with replay.

Issue #16 adds C4: provider-neutral embedding and reranker **interfaces** plus
one deterministic offline hash provider. It is a frozen contract, not
retrieval — nothing composes it yet, and no HTTP surface exposes it. C5 wires
it to Qdrant and `index_revision`.

See issue #1 (platform Epic) for the child map and dependency order.

## Quick start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m ruff check .

python3 -m ckp                       # serve on 127.0.0.1:8080
curl -s localhost:8080/health
curl -s localhost:8080/revision
```

Container, including the endpoint smoke test:

```bash
bash scripts/smoke-container.sh
```

## Mac mini runtime (issue #78, retired)

Retired: the LaunchAgent no longer exists on the Mac mini (checked
2026-10-13). The scripts stay for reference only.

A persistent Gateway on the Mac mini ran as a login-session LaunchAgent,
following the Cyclone-Dashboard deploy pattern:

```bash
./deploy/install-launchd.sh      # venv + env file + plist, then health check
./deploy/update.sh               # pull origin/main, reinstall, restart, verify
```

Host-specific values live in `~/.config/cyclone/ckp-gateway.env` (created on
first install; only valid `CKP_<SECTION>_<KEY>` names may appear there). The
default binding is loopback (`127.0.0.1:8092`) with the bundle root pointed at
the machine's real `cyclone-wiki` checkout. Note the anonymous read surface
serves `privacy: public` notes only; the current wiki corpus has no public
formal notes, so its catalog lists nothing today — `/health` and `/revision`
are the meaningful smoke endpoints for this runtime.

## Local Core inbox bridge

`ckp-wiki-capture` delegates one authenticated, create-only capture to
Cyclone-Wiki's existing `scripts/write_inbox_capture.sh`. The adapter fixes the
route to `Codex + agent-discussion + Core/_inbox/agent-captures`; callers cannot
select Private, a formal path, append mode or a different Git mode.

The caller provides `CKP_WRITER_ACTOR_CREDENTIAL` and its separately configured
`CKP_WRITER_ACTOR_CREDENTIAL_SHA256`. Neither value is forwarded to the Wiki
wrapper. Host paths remain runtime inputs:

```bash
ckp-wiki-capture \
  --wiki-root "$CYCLONE_WIKI_ROOT" \
  --state-root "$CKP_WRITER_STATE_ROOT" \
  --slug example-capture \
  --title "Example capture" \
  --body-file ./body.md \
  --request-id request-1 \
  --task-id task-1 \
  --model-id gpt-5.4 \
  --dry-run
```

This bridge is online-only. A wrapper, validation, freeze or push failure is a
rejection; it is never reported as queued success.

## Read surfaces

- `/catalog` and `/query` are the anonymous C3 surface and always remain
  `public`-only.
- `/scoped/catalog/{domain}` and `/scoped/query/{domain}` require separate
  opaque actor and task-grant credentials. Actor, task, domain, capability,
  privacy classes, expiry and limits resolve from server state; request JSON
  cannot provide them.
- `/context/{domain}` additionally requires report-generation capability and
  returns a deterministic item-bounded context whose `utf8-bytes-v1` count is
  a conservative provider-neutral token upper bound.

The task grant's `max_items` and token bound constrain `/context`; scoped
Catalog and query keep their C3 request limits after the same auth, privacy and
domain filters.

The default app has no standing protected grants or domain bindings, so every
protected call fails closed until a trusted composition injects both. C6 does
not load production credentials, deploy a runtime, or connect a real Private
bundle.

## Revision reporting

`/revision` answers "what exactly is this process serving", with the four
fields the OKF contract requires — and names the evidence behind each one, so
a derived value and a self-declared one stay distinguishable.

| Field | Source | Falls back to |
| --- | --- | --- |
| `profile_version` | the bundle's `bundle.toml` | config `profile.expected_version`, then `null` |
| `api_version` | `ckp.revision.API_VERSION` — also the published schema version | — |
| `bundle_commit` | `git rev-parse HEAD` in the bundle | a build-time `.bundle-commit` stamp, then `null` |
| `index_revision` | sha256 over the bundle's note paths and bytes | `null` when the bundle is unreadable or holds no notes |

Two properties are load-bearing rather than cosmetic. `index_revision` is
**derived**: same bundle, same digest, on any host — contract §5.5 makes a
non-reproducible rebuild a rollback trigger. And nothing is **fabricated**:
absent evidence reports as `null`, because a placeholder would make a broken
deployment look identical to a healthy one.

`/health` answers by running the same computation, so it cannot report ok on a
bundle `/revision` finds nothing in. The C3 anchored reader rejects symlinks,
hardlinks and snapshot races before privacy classification, Catalog projection
or citation can consume the bytes.

## Configuration

Three layers, most specific last:

1. `src/ckp/defaults.toml` — shipped in the package, and the key schema.
2. an optional TOML file, via `CKP_CONFIG_FILE` (see `config/example.toml`).
3. environment variables, `CKP_<SECTION>_<KEY>` — e.g. `CKP_BUNDLE_ROOT`.

An override naming a key the schema does not define is an **error**, not a
no-op. `/health` reports which layers actually contributed.

## Layout

```
src/ckp/          service package
config/           example for the file config layer
fixtures/         synthetic bundles — never real Wiki content
scripts/          container smoke test
tests/            pytest suite
```

## Related

| Repo | Role |
| --- | --- |
| `cyclone-tw/cyclone-wiki` | Cyclone Profile, metadata, taxonomy, validator, migration |
| `cyclone-tw/Cyclone-Dashboard` | Gateway client, Catalog UI, review controls |
| `cyclone-tw/cyclone-agent-config` | Coding agent rules, skills, review tooling |
| `cyclone-tw/cyclone-wiki-seeker` | Live Wiki question answering; ported this repo's evaluation method and lexical scorer |
