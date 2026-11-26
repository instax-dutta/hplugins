#!/usr/bin/env python3
"""Decide whether a verified pull request may be auto-merged.

The workflow fetches API payloads into JSON files, then calls this script.
It never fetches or mutates anything itself. Output is either ``MERGE`` or
``SKIP: <reason>`` so the reason is visible in the Actions log.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ALLOWED_PATH = re.compile(
    r"^(?:data/|docs/|entries/[a-z0-9-]+/index\.html$|"
    r"HERETIC_PLUGINS\.md$|README\.md$|DESIGN\.md$|PRODUCT\.md$|AGENTS\.md$)"
)
GOOD_CONCLUSIONS = {"success", "neutral", "skipped"}
BAD_MERGE_STATES = {"dirty", "blocked", "behind", "unstable"}


def merge_decision(
    run: dict,
    pr: dict,
    checks: object,
    reviews: object,
    threads: dict,
    files: object,
    head_sha: str,
) -> str:
    """Return MERGE or SKIP with a concrete reason. Pure, no I/O."""
    if run.get("conclusion") != "success" or run.get("event") != "pull_request":
        return "SKIP: triggering run is not a successful pull_request run"
    if run.get("name") != "check catalog":
        return "SKIP: triggering run is not the catalog check workflow"

    run_repo = run.get("head_repository", {}).get("full_name")
    pr_repo = pr.get("head", {}).get("repo", {}).get("full_name")
    if not run_repo or run_repo != pr_repo:
        return "SKIP: PR head repository does not match the workflow run"

    if pr.get("state") != "open":
        return "SKIP: PR is not open"
    if pr.get("draft") is not False:
        return "SKIP: PR is a draft"
    if pr.get("base", {}).get("ref") != "main":
        return "SKIP: PR does not target main"
    if pr.get("head", {}).get("sha") != head_sha:
        return "SKIP: PR head moved after checks ran"
    if pr.get("mergeable") is not True:
        return "SKIP: PR is not mergeable"
    if pr.get("mergeable_state") in BAD_MERGE_STATES:
        return "SKIP: PR merge state is " + str(pr.get("mergeable_state"))

    if not isinstance(reviews, list):
        return "SKIP: PR review list is unavailable"
    if any(
        isinstance(review, dict) and review.get("state") == "CHANGES_REQUESTED"
        for review in reviews
    ):
        return "SKIP: PR has a requested change"

    nodes = threads.get("data", {}).get("repository", {}).get("pullRequest", {}).get("reviewThreads")
    if not isinstance(nodes, dict):
        return "SKIP: review thread data is unavailable"
    if nodes.get("pageInfo", {}).get("hasNextPage"):
        return "SKIP: review threads were not fully checked"
    if any(thread.get("isResolved") is False for thread in nodes.get("nodes", [])):
        return "SKIP: PR has unresolved review threads"

    check_runs = checks.get("check_runs", checks) if isinstance(checks, dict) else checks
    if not isinstance(check_runs, list) or not check_runs:
        return "SKIP: no check runs found for the head commit"
    if not any(
        check.get("name") == "validate"
        and check.get("status") == "completed"
        and check.get("conclusion") == "success"
        for check in check_runs
    ):
        return "SKIP: the validate check is not successful"
    if any(
        check.get("status") != "completed" or check.get("conclusion") not in GOOD_CONCLUSIONS
        for check in check_runs
    ):
        return "SKIP: one or more checks are not green"

    if not isinstance(files, list) or not files:
        return "SKIP: PR file list is unavailable"
    for changed in files:
        for key in ("filename", "previous_filename"):
            path = changed.get(key)
            if path and not ALLOWED_PATH.match(path):
                return f"SKIP: manual review required for {path}"

    return "MERGE"


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    payloads = {}
    for name in ("run", "pr", "checks", "reviews", "threads", "files"):
        try:
            payloads[name] = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SKIP: could not read {name}.json: {exc}")
            return 0
    head_sha = payloads["run"].get("head_sha", "")
    print(merge_decision(head_sha=head_sha, **payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
