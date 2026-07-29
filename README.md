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

Phase 3 bootstrap. See issue #1 (platform Epic) for the child map, dependency
order and file ownership.

## Quick start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 -m ruff check .
```

## Layout

```
src/ckp/          service package (added by the walking skeleton child)
tests/            pytest suite
config/           layered configuration defaults
fixtures/         synthetic bundles — never real Wiki content
```

## Related

| Repo | Role |
| --- | --- |
| `cyclone-tw/cyclone-wiki` | Cyclone Profile, metadata, taxonomy, validator, migration |
| `cyclone-tw/Cyclone-Dashboard` | Gateway client, Catalog UI, review controls |
| `cyclone-tw/cyclone-agent-config` | Coding agent rules, skills, review tooling |
