# tests

## Purpose

Offline verification of the catalog and the build-time data layer.

## Ownership

Test files here are the executable contract for `tools/build.py`, `tools/automerge_check.py`, and `data/catalog.json`.

## Local Contracts

- Standard library `unittest` only.
- Tests must not need a browser, a network connection, or third-party packages.

## Work Guidance

- Test validation failure cases with in-memory fixtures, not by mutating the real catalog.
- Keep one test per rule so a failure names the broken contract directly.

## Verification

- `python3 -m unittest discover -s tests -v`

## Child DOX Index

None.
