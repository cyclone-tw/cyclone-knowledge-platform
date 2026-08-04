# cyclone-knowledge-platform

Private Knowledge Platform for the Cyclone-Wiki OKF migration.
Phase 3 of `cyclone-tw/cyclone-wiki#142`.

**Agents: read [`AGENTS.md`](AGENTS.md) first.**

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

## Status

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
