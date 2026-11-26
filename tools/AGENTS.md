# tools

## Purpose

The build-time data layer and the auto-merge decision guard: catalog validation, index generation, entry-page generation, and merge safety rules.

## Ownership

`tools/build.py` is the only writer of `data/search-index.json` and of the `entries/<slug>/index.html` entry pages. `tools/automerge_check.py` owns the merge decision and reads only API payloads prepared by the workflow. CI runs the same validation in check mode.

## Local Contracts

- Python 3 standard library only. No third-party packages, no network calls.
- `--check` must be read-only and fail with a non-zero exit code on any schema or provenance violation.
- Default invocation regenerates the search index and entry pages, removes stale generated pages, and prints a short report.
- The build script is the "backend": GitHub Pages has no server runtime, so all transformation happens before publish.
- The auto-merge guard must fail closed on malformed payloads, moved PR heads, unresolved threads, requested changes, and disallowed paths. New allowlist or gate rules require a unit test.

## Work Guidance

- Add validation rules here, never in the browser. The browser assumes clean data.
- Keep error messages actionable: include the slug and the field that failed.

## Verification

- `python3 -m unittest discover -s tests -v`
- `python3 tools/build.py --check`

## Child DOX Index

None.
