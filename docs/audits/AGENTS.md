# audits

## Purpose

Durable audit reports for the published site. Currently the GEO audit, which records scores, findings, and the remaining off-site gaps.

## Ownership

Reports here are evidence records. Site metadata and generated files are owned by `assets/AGENTS.md` and `tools/AGENTS.md`; an audit report never edits them.

## Local Contracts

- Report claims must come from observed page data. Do not invent mentions, dates, benchmarks, or scores.
- Keep the current score and the remediation status accurate when the site changes.
- Reports are documentation, never browser-delivered copy.

## Work Guidance

- Re-run the audit after significant metadata or content changes and update the report in the same change.

## Verification

- `python3 tools/build.py --check` confirms the published artifacts the report describes.

## Child DOX Index

None.
