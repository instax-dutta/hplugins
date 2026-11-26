# docs

## Purpose

Product record, design decisions, implementation plan, contributor guide, and extracted source inventory.

## Ownership

Root `PRODUCT.md` and `DESIGN.md`, plus files under `docs/`, are durable project documentation. Browser-facing copy is not authored here; it comes from `data/catalog.json`.

## Local Contracts

- Design direction and token values belong in `DESIGN.md` and its sidecar `.impeccable/design.json`, never in the browser bundle.
- `docs/superpowers/specs/` holds the design spec; `docs/plans/` holds the implementation plan.
- `docs/contributing.md` is the instruction humans follow to add or update a catalog entry; entry pages are generated, never hand-written.
- `docs/HERETIC_PLUGINS.md` is the extracted inventory of upstream and community material. Treat it as source evidence, not as site copy.
- Surface briefs hold development-only direction contracts. Never copy that contract into browser-delivered files; the brief is for reviewers and documenters.

## Work Guidance

- Keep docs operational: stable contracts, not a diary of changes.
- When the catalog schema changes, update `docs/contributing.md` and `data/AGENTS.md` in the same change.

## Verification

- No automated checks. Links in contributor docs must resolve to real files in the repo.

## Child DOX Index

- `audits/AGENTS.md` - audit reports and their evidence rules
