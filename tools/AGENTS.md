# tools

## Purpose

The build-time data layer: validation, index generation, and CI wiring.

## Ownership

`tools/build.py` is the only writer of `data/search-index.json`. CI runs the same script in check mode.

## Local Contracts

- Python 3 standard library only. No third-party packages, no network calls.
- `--check` must be read-only and fail with a non-zero exit code on any schema or provenance violation.
- Default invocation regenerates the search index and prints a short report.
- The script is the "backend": GitHub Pages has no server runtime, so all transformation happens before publish.

## Work Guidance

- Add validation rules here, never in the browser. The browser assumes clean data.
- Keep error messages actionable: include the slug and the field that failed.

## Verification

- `python3 -m unittest discover -s tests -v`
- `python3 tools/build.py --check`

## Child DOX Index

None.
