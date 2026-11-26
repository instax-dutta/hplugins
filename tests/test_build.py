"""Offline checks for the catalog schema and the generated index."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("hplugins_build", ROOT / "tools" / "build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)

with (ROOT / "data" / "catalog.json").open(encoding="utf-8") as handle:
    REAL = json.load(handle)


def entry(**overrides):
    base = {
        "slug": "test-entry", "name": "Test Entry", "type": "scorer",
        "author": "tester", "summary": "A short summary.", "description": "A description.",
        "status": "available", "tags": ["test"],
        "where_to_find": [{"label": "source", "url": "https://example.com"}],
        "how_to_use": ["Do the thing."], "config_snippet": "x = 1",
        "setup_prompt": "x" * 200,
    }
    base.update(overrides)
    return base


class CatalogTests(unittest.TestCase):
    def test_real_catalog_is_valid(self):
        self.assertEqual(build.validate(REAL), [])

    def test_empty_catalog_rejected(self):
        self.assertTrue(build.validate([]))

    def test_duplicate_slug_rejected(self):
        errors = build.validate([entry(), entry()])
        self.assertIn("test-entry: duplicate slug", errors)

    def test_bad_slug_rejected(self):
        self.assertTrue(build.validate([entry(slug="Bad Slug")]))

    def test_missing_field_rejected(self):
        broken = entry()
        del broken["setup_prompt"]
        self.assertTrue(any("missing required field 'setup_prompt'" in e for e in build.validate([broken])))

    def test_bad_type_and_status_rejected(self):
        errors = build.validate([entry(type="magic", status="maybe")])
        self.assertTrue(any("type must be one of" in e for e in errors))
        self.assertTrue(any("status must be one of" in e for e in errors))

    def test_empty_links_rejected(self):
        self.assertTrue(build.validate([entry(where_to_find=[])]))

    def test_bad_link_url_rejected(self):
        errors = build.validate([entry(where_to_find=[{"label": "x", "url": "http://example.com"}])])
        self.assertTrue(any("link url must be https" in e for e in errors))

    def test_missing_local_link_rejected(self):
        errors = build.validate([entry(where_to_find=[{"label": "x", "url": "no-such-file.py"}])])
        self.assertTrue(any("local link does not resolve" in e for e in errors))

    def test_short_prompt_rejected(self):
        self.assertTrue(build.validate([entry(setup_prompt="too short")]))

    def test_long_summary_rejected(self):
        self.assertTrue(build.validate([entry(summary="x" * 161)]))


class IndexTests(unittest.TestCase):
    def test_index_fields(self):
        index = build.build_index([entry()])
        self.assertEqual(index[0]["slug"], "test-entry")
        self.assertIn("test entry", index[0]["search"])
        self.assertEqual(set(index[0]), {"slug", "name", "type", "author", "summary", "status", "tags", "search"})

    def test_generated_index_matches_file(self):
        generated = json.dumps(build.build_index(REAL), ensure_ascii=False, indent=2) + "\n"
        self.assertEqual((ROOT / "data" / "search-index.json").read_text(encoding="utf-8"), generated)

    def test_real_catalog_copy_is_not_mutated(self):
        snapshot = copy.deepcopy(REAL)
        build.build_index(REAL)
        self.assertEqual(REAL, snapshot)


if __name__ == "__main__":
    unittest.main()
