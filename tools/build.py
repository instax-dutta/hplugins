#!/usr/bin/env python3
"""Validate hplugins catalog data and generate the browser search index.

This is the whole backend: GitHub Pages serves static files, so every
transformation of the source data happens here before publish.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
INDEX = ROOT / "data" / "search-index.json"
ENTRIES_DIR = ROOT / "entries"

REQUIRED = (
    "slug", "name", "type", "author", "summary", "description", "status",
    "tags", "where_to_find", "how_to_use", "config_snippet", "setup_prompt",
)
TYPES = {"scorer", "modifier", "config", "model", "fork", "tool"}
STATUSES = {"available", "community", "proof-of-concept", "planned"}
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate(entries: object) -> list[str]:
    """Return actionable validation errors. Empty list means the catalog is valid."""
    errors: list[str] = []
    if not isinstance(entries, list) or not entries:
        return ["catalog must be a non-empty array"]

    seen: set[str] = set()
    for i, entry in enumerate(entries):
        label = entry.get("slug", f"entry[{i}]") if isinstance(entry, dict) else f"entry[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: must be an object")
            continue

        for field in REQUIRED:
            if field not in entry:
                errors.append(f"{label}: missing required field '{field}'")

        slug = entry.get("slug")
        if not isinstance(slug, str) or not SLUG.match(slug):
            errors.append(f"{label}: slug must be lowercase words separated by hyphens")
        elif slug in seen:
            errors.append(f"{label}: duplicate slug")
        else:
            seen.add(slug)

        for field in ("name", "author", "summary", "description", "config_snippet"):
            if field in entry and not isinstance(entry[field], str):
                errors.append(f"{label}: '{field}' must be a string")
        for field in ("name", "author", "summary", "description"):
            if isinstance(entry.get(field), str) and not entry[field].strip():
                errors.append(f"{label}: '{field}' must not be empty")

        if isinstance(entry.get("summary"), str) and len(entry["summary"]) > 160:
            errors.append(f"{label}: summary must be under 160 characters")

        if entry.get("type") not in TYPES:
            errors.append(f"{label}: type must be one of {sorted(TYPES)}")
        if entry.get("status") not in STATUSES:
            errors.append(f"{label}: status must be one of {sorted(STATUSES)}")

        tags = entry.get("tags")
        if not isinstance(tags, list) or not tags:
            errors.append(f"{label}: tags must be a non-empty array")
        elif any(not isinstance(t, str) or not t.strip() or t != t.lower() for t in tags):
            errors.append(f"{label}: tags must be non-empty lowercase strings")

        links = entry.get("where_to_find")
        if not isinstance(links, list) or not links:
            errors.append(f"{label}: where_to_find must be a non-empty array")
        else:
            for link in links:
                if not isinstance(link, dict) or not link.get("label") or not link.get("url"):
                    errors.append(f"{label}: where_to_find entries need label and url")
                    continue
                url = link["url"]
                if not isinstance(url, str):
                    errors.append(f"{label}: link url must be a string")
                elif url.startswith("https://"):
                    continue
                elif url.startswith(("http://", "//", "/", ".")) or not url:
                    errors.append(f"{label}: link url must be https or a site-relative path: {url}")
                elif not (ROOT / url).is_file():
                    errors.append(f"{label}: local link does not resolve to a file: {url}")

        steps = entry.get("how_to_use")
        if not isinstance(steps, list) or not steps:
            errors.append(f"{label}: how_to_use must be a non-empty array of steps")
        elif any(not isinstance(s, str) or not s.strip() for s in steps):
            errors.append(f"{label}: how_to_use steps must be non-empty strings")

        prompt = entry.get("setup_prompt")
        if not isinstance(prompt, str) or len(prompt.strip()) < 120:
            errors.append(f"{label}: setup_prompt must be a usable prompt of at least 120 characters")

    return errors


def build_index(entries: list[dict]) -> list[dict]:
    """Compact display and search fields consumed by the browser."""
    index = []
    for entry in entries:
        haystack = " ".join(
            [entry["name"], entry["author"], entry["summary"], entry["description"],
             entry["type"], entry["status"], *entry["tags"]]
        ).lower()
        index.append({
            "slug": entry["slug"],
            "name": entry["name"],
            "type": entry["type"],
            "author": entry["author"],
            "summary": entry["summary"],
            "status": entry["status"],
            "tags": entry["tags"],
            "search": haystack,
        })
    return index


def load_catalog() -> object:
    with CATALOG.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_entry_pages(entries: list[dict]) -> dict[str, str]:
    """Derive one static page per slug from the root shell."""
    shell = (ROOT / "index.html").read_text(encoding="utf-8")
    pages = {}
    for entry in entries:
        page = shell.replace('href="assets/', 'href="../../assets/')
        page = page.replace('src="assets/', 'src="../../assets/')
        page = page.replace('href="docs/HERETIC_PLUGINS.md"', 'href="../../docs/HERETIC_PLUGINS.md"')
        meta = (
            '  <meta name="site-root" content="../../">\n'
            f'  <meta name="entry-slug" content="{entry["slug"]}">\n'
        )
        page = page.replace(
            '  <meta name="color-scheme" content="light dark">',
            meta + '  <meta name="color-scheme" content="light dark">',
        )
        pages[entry["slug"]] = page
    return pages


def legacy_entry_dirs() -> list[Path]:
    """Find entry pages left at the repo root by an older layout."""
    found = []
    for directory in ROOT.iterdir():
        page = directory / "index.html"
        if not directory.is_dir() or not page.is_file():
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        if 'name="entry-slug"' in text:
            found.append(directory)
    return found


def stale_entry_dirs(slugs: set[str]) -> list[Path]:
    """Find generated entry pages whose slug is no longer in the catalog."""
    stale = []
    if not ENTRIES_DIR.is_dir():
        return []
    for directory in ENTRIES_DIR.iterdir():
        page = directory / "index.html"
        if not directory.is_dir() or not page.is_file():
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'name="entry-slug" content="([^"]+)"', text)
        if match and match.group(1) not in slugs:
            stale.append(directory)
    return stale


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    try:
        entries = load_catalog()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return 1

    errors = validate(entries)
    if errors:
        for error in errors:
            print(f"INVALID {error}", file=sys.stderr)
        print(f"{len(errors)} validation error(s)", file=sys.stderr)
        return 1

    generated = json.dumps(build_index(entries), ensure_ascii=False, indent=2) + "\n"
    pages = build_entry_pages(entries)
    slugs = set(pages)

    if check_only:
        if not INDEX.exists() or INDEX.read_text(encoding="utf-8") != generated:
            print("INVALID search index is missing or stale; run tools/build.py", file=sys.stderr)
            return 1
        for slug, page in pages.items():
            target = ENTRIES_DIR / slug / "index.html"
            if not target.is_file() or target.read_text(encoding="utf-8") != page:
                print(f"INVALID entry page for {slug} is missing or stale; run tools/build.py", file=sys.stderr)
                return 1
        stale = stale_entry_dirs(slugs)
        if stale:
            print("INVALID stale generated pages: " + ", ".join(p.name for p in stale), file=sys.stderr)
            return 1
    else:
        INDEX.write_text(generated, encoding="utf-8")
        for slug, page in pages.items():
            target = ENTRIES_DIR / slug / "index.html"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(page, encoding="utf-8")
        for directory in stale_entry_dirs(slugs) + legacy_entry_dirs():
            shutil.rmtree(directory)

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["type"]] = counts.get(entry["type"], 0) + 1
    summary = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
    mode = "checked" if check_only else "built"
    print(
        f"{mode}: {len(entries)} entries ({summary}), index {len(generated.encode())} bytes, "
        f"{len(pages)} entry pages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
