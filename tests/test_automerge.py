"""Tests for the auto-merge decision guard."""

import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("automerge_check", ROOT / "tools" / "automerge_check.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def fixtures():
    return {
        "run": {
            "conclusion": "success", "event": "pull_request", "name": "check catalog",
            "head_sha": "abc123", "head_repository": {"full_name": "owner/repo"},
        },
        "pr": {
            "state": "open", "draft": False,
            "head": {"sha": "abc123", "repo": {"full_name": "owner/repo"}},
            "base": {"ref": "main"}, "mergeable": True, "mergeable_state": "clean",
        },
        "checks": {"check_runs": [{"name": "validate", "status": "completed", "conclusion": "success"}]},
        "reviews": [],
        "threads": {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "pageInfo": {"hasNextPage": False}, "nodes": []}}}}},
        "files": [{"filename": "data/catalog.json"}, {"filename": "entries/krea2-drift/index.html"}],
        "head_sha": "abc123",
    }


def decide(**overrides):
    payloads = fixtures()
    payloads.update(overrides)
    return guard.merge_decision(**payloads)


class GuardTests(unittest.TestCase):
    def test_happy_path_merges(self):
        self.assertEqual(decide(), "MERGE")

    def test_head_must_match_verified_commit(self):
        self.assertTrue(decide(head_sha="different").startswith("SKIP"))

    def test_trigger_must_be_successful_pr_run(self):
        pr = copy.deepcopy(fixtures()["run"])
        pr["conclusion"] = "failure"
        self.assertTrue(decide(run=pr).startswith("SKIP"))
        pr = copy.deepcopy(fixtures()["run"])
        pr["event"] = "push"
        self.assertTrue(decide(run=pr).startswith("SKIP"))

    def test_only_the_catalog_workflow_counts(self):
        run = copy.deepcopy(fixtures()["run"])
        run["name"] = "something else"
        self.assertTrue(decide(run=run).startswith("SKIP"))

    def test_head_repository_must_match(self):
        run = copy.deepcopy(fixtures()["run"])
        run["head_repository"] = {"full_name": "attacker/repo"}
        self.assertTrue(decide(run=run).startswith("SKIP"))

    def test_pr_must_be_open_ready_and_target_main(self):
        pr = copy.deepcopy(fixtures()["pr"])
        pr["draft"] = True
        self.assertTrue(decide(pr=pr).startswith("SKIP"))
        pr = copy.deepcopy(fixtures()["pr"])
        pr["state"] = "closed"
        self.assertTrue(decide(pr=pr).startswith("SKIP"))
        pr = copy.deepcopy(fixtures()["pr"])
        pr["base"] = {"ref": "dev"}
        self.assertTrue(decide(pr=pr).startswith("SKIP"))

    def test_mergeability_required(self):
        pr = copy.deepcopy(fixtures()["pr"])
        pr["mergeable"] = False
        self.assertTrue(decide(pr=pr).startswith("SKIP"))
        pr = copy.deepcopy(fixtures()["pr"])
        pr["mergeable_state"] = "dirty"
        self.assertTrue(decide(pr=pr).startswith("SKIP"))

    def test_requested_changes_block_merge(self):
        reviews = [{"state": "CHANGES_REQUESTED"}]
        self.assertTrue(decide(reviews=reviews).startswith("SKIP"))

    def test_unresolved_threads_block_merge(self):
        threads = copy.deepcopy(fixtures()["threads"])
        threads["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"] = [{"isResolved": False}]
        self.assertTrue(decide(threads=threads).startswith("SKIP"))

    def test_malformed_review_payloads_block_merge(self):
        self.assertTrue(decide(reviews={}).startswith("SKIP"))
        self.assertTrue(decide(threads={}).startswith("SKIP"))

    def test_unpaginated_threads_block_merge(self):
        threads = copy.deepcopy(fixtures()["threads"])
        threads["data"]["repository"]["pullRequest"]["reviewThreads"]["pageInfo"]["hasNextPage"] = True
        self.assertTrue(decide(threads=threads).startswith("SKIP"))

    def test_validate_check_must_succeed(self):
        checks = {"check_runs": [{"name": "validate", "status": "completed", "conclusion": "failure"}]}
        self.assertTrue(decide(checks=checks).startswith("SKIP"))
        self.assertTrue(decide(checks={"check_runs": []}).startswith("SKIP"))

    def test_other_checks_must_be_green(self):
        checks = {"check_runs": [
            {"name": "validate", "status": "completed", "conclusion": "success"},
            {"name": "lint", "status": "in_progress", "conclusion": None},
        ]}
        self.assertTrue(decide(checks=checks).startswith("SKIP"))

    def test_disallowed_paths_require_review(self):
        self.assertTrue(decide(files=[{"filename": ".github/workflows/check.yml"}]).startswith("SKIP"))
        self.assertTrue(decide(files=[{"filename": "tools/build.py"}]).startswith("SKIP"))
        self.assertTrue(decide(files=[{"filename": "assets/js/app.js"}]).startswith("SKIP"))

    def test_rename_into_allowed_path_still_checked(self):
        files = [{"filename": "data/ok.json", "previous_filename": ".github/workflows/auto-merge.yml"}]
        self.assertTrue(decide(files=files).startswith("SKIP"))

    def test_cli_reads_payload_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, value in fixtures().items():
                if name == "head_sha":
                    continue
                (Path(tmp) / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(ROOT / "tools" / "automerge_check.py"), tmp],
                capture_output=True, text=True,
            )
            self.assertEqual(result.stdout.strip(), "MERGE")
            self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
