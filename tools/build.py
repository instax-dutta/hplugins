#!/usr/bin/env python3
"""Validate the hplugins catalog and generate every published artifact.

GitHub Pages has no server runtime, so all transformation happens here before
publish: HTML pages, static content for crawlers, metadata, discovery files,
and the browser search index.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
INDEX = ROOT / "data" / "search-index.json"
ENTRIES_DIR = ROOT / "entries"
SHELL = ROOT / "tools" / "shell.html"
SITE_URL = "https://hplugins.sdad.pro"
OG_IMAGE = SITE_URL + "/assets/og-image.png"

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
            "slug": entry["slug"], "name": entry["name"], "type": entry["type"],
            "author": entry["author"], "summary": entry["summary"], "status": entry["status"],
            "tags": entry["tags"], "search": haystack,
        })
    return index


def load_catalog() -> object:
    with CATALOG.open(encoding="utf-8") as handle:
        return json.load(handle)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def site_href(url: str, root: str) -> str:
    return url if url.startswith("https://") else root + url


def json_script(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_head_meta(title: str, description: str, canonical: str, schema: dict) -> str:
    return "\n".join([
        f'  <link rel="canonical" href="{esc(canonical)}">',
        '  <meta property="og:type" content="website">',
        '  <meta property="og:site_name" content="hplugins">',
        f'  <meta property="og:title" content="{esc(title)}">',
        f'  <meta property="og:description" content="{esc(description)}">',
        f'  <meta property="og:url" content="{esc(canonical)}">',
        f'  <meta property="og:image" content="{esc(OG_IMAGE)}">',
        '  <meta name="twitter:card" content="summary_large_image">',
        f'  <meta name="twitter:title" content="{esc(title)}">',
        f'  <meta name="twitter:description" content="{esc(description)}">',
        f'  <meta name="twitter:image" content="{esc(OG_IMAGE)}">',
        f'  <script type="application/ld+json">{json_script(schema)}</script>',
    ])


def render_page(title: str, description: str, canonical: str, content: str, schema: dict,
                root: str, page_meta: str = "") -> str:
    template = SHELL.read_text(encoding="utf-8")
    return (template
        .replace("{{TITLE}}", esc(title))
        .replace("{{DESCRIPTION}}", esc(description))
        .replace("{{HEAD_META}}", render_head_meta(title, description, canonical, schema))
        .replace("{{PAGE_META}}", page_meta)
        .replace("{{STATIC_CONTENT}}", content)
        .replace("{{ROOT}}", root))


def render_static_catalog(entries: list[dict]) -> str:
    rows = []
    for i, entry in enumerate(entries, 1):
        rows.append(
            '<li><a class="entry-row" href="entries/' + esc(entry["slug"]) + '/">' +
            '<span class="entry-index">' + f"{i:02d}" + '</span>' +
            '<span class="entry-body"><span class="entry-name">' + esc(entry["name"]) + '</span>' +
            '<span class="entry-summary">' + esc(entry["summary"]) + '</span></span>' +
            '<span class="entry-meta">/' + esc(entry["type"]) + '</span></a></li>'
        )
    return (
        '<header class="catalog-head"><h1 class="catalog-title">Community-built plugins for Heretic.</h1>' +
        '<p class="catalog-intro">Scorers, modifiers, configs, models, and tools made by the Heretic community. ' +
        'Each entry has a source, a guide, and a setup prompt.</p></header>' +
        '<ol class="entry-list">' + "".join(rows) + "</ol>"
    )


def render_static_entry(entry: dict) -> str:
    links = "".join(
        '<li><a href="' + esc(site_href(link["url"], "../../")) + '">' +
        esc(link["label"]) + "</a></li>"
        for link in entry["where_to_find"]
    )
    steps = "".join("<li>" + esc(step) + "</li>" for step in entry["how_to_use"])
    config = (
        '<section class="article-section"><h2 class="section-title">Config fragment</h2>' +
        '<pre class="code-block">' + esc(entry["config_snippet"]) + "</pre></section>"
        if entry["config_snippet"] else ""
    )
    return (
        '<div class="article">' +
        '<a class="crumb" href="../../">&larr; back to index</a>' +
        '<h1 class="article-title">' + esc(entry["name"]) + "</h1>" +
        '<p class="article-meta">/' + esc(entry["type"]) + " - by " + esc(entry["author"]) +
        " - " + esc(entry["status"]) + "</p>" +
        '<p class="article-lead">' + esc(entry["summary"]) + "</p>" +
        '<section class="article-section"><h2 class="section-title">What it does</h2>' +
        '<p class="prose">' + esc(entry["description"]) + "</p></section>" +
        '<section class="article-section"><h2 class="section-title">Where to find it</h2>' +
        '<ul class="link-list">' + links + "</ul></section>" +
        '<section class="article-section"><h2 class="section-title">How to use it</h2>' +
        '<ol class="steps">' + steps + "</ol></section>" + config +
        '<section class="article-section"><h2 class="section-title">Setup prompt</h2>' +
        '<pre class="code-block">' + esc(entry["setup_prompt"]) + "</pre></section>" +
        "</div>"
    )


def home_schema(entries: list[dict]) -> dict:
    return {
        "@context": "https://schema.org", "@type": "WebSite", "name": "hplugins",
        "url": SITE_URL + "/",
        "description": "A curated registry of community-built Heretic plugins, scorers, configs, and models.",
        "inLanguage": "en",
        "publisher": {"@type": "Organization", "name": "hplugins community", "url": SITE_URL + "/"},
        "mainEntity": {"@type": "ItemList", "itemListElement": [
            {"@type": "ListItem", "position": i,
             "name": entry["name"], "url": SITE_URL + "/entries/" + entry["slug"] + "/"}
            for i, entry in enumerate(entries, 1)
        ]},
    }


def entry_schema(entry: dict) -> dict:
    return {
        "@context": "https://schema.org", "@type": "Article",
        "headline": entry["name"], "description": entry["summary"],
        "author": {"@type": "Person", "name": entry["author"]},
        "keywords": ", ".join(entry["tags"]), "inLanguage": "en",
        "mainEntityOfPage": SITE_URL + "/entries/" + entry["slug"] + "/",
        "isPartOf": {"@type": "WebSite", "name": "hplugins", "url": SITE_URL + "/"},
        "citation": [link["url"] for link in entry["where_to_find"] if link["url"].startswith("https://")],
    }


def build_pages(entries: list[dict]) -> tuple[str, dict[str, str]]:
    home = render_page(
        "hplugins - Community-built Heretic plugins and scorers",
        "A curated index of community-built Heretic scorers, modifiers, configs, and models. "
        "Each entry includes source links, a usage guide, and a copy-paste setup prompt.",
        SITE_URL + "/", render_static_catalog(entries), home_schema(entries), "",
    )
    pages = {
        entry["slug"]: render_page(
            entry["name"] + " - hplugins", entry["summary"],
            SITE_URL + "/entries/" + entry["slug"] + "/",
            render_static_entry(entry), entry_schema(entry), "../../",
            '  <meta name="site-root" content="../../">\n'
            f'  <meta name="entry-slug" content="{entry["slug"]}">',
        )
        for entry in entries
    }
    return home, pages


def build_discovery_files(entries: list[dict]) -> dict[str, str]:
    urls = [SITE_URL + "/"] + [SITE_URL + "/entries/" + e["slug"] + "/" for e in entries]
    sitemap = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    sitemap += "".join("  <url><loc>" + esc(url) + "</loc></url>\n" for url in urls)
    sitemap += "</urlset>\n"

    robots = "\n".join([
        "User-agent: *", "Allow: /", "",
        "User-agent: GPTBot", "Allow: /", "",
        "User-agent: ClaudeBot", "Allow: /", "",
        "User-agent: PerplexityBot", "Allow: /", "",
        "User-agent: Google-Extended", "Allow: /", "",
        "User-agent: CCBot", "Allow: /", "",
        "Sitemap: " + SITE_URL + "/sitemap.xml", "",
    ])

    llms = [
        "# hplugins", "",
        "> A curated registry of community-built Heretic plugins, scorers, configs, models, and tools.",
        "> Every entry has a source, a usage guide, and a setup prompt for a coding model.", "",
        "## Start here", "",
        "- [Catalog](https://hplugins.sdad.pro/) - every community entry",
        "- [Contribute an entry](https://hplugins.sdad.pro/docs/contributing.md) - required format and PR steps",
        "- [Full catalog text](https://hplugins.sdad.pro/llms-full.txt) - all entries with descriptions and sources", "",
        "## Entries", "",
    ]
    llms += ["- [" + e["name"] + "](" + SITE_URL + "/entries/" + e["slug"] + "/): " + e["summary"] for e in entries]
    llms.append("")

    full = ["# hplugins - full catalog", "",
            "Static registry of community Heretic work. Sources are attributed to their authors.", ""]
    for entry in entries:
        full += ["## " + entry["name"], "",
                 "Type: " + entry["type"] + ". Author: " + entry["author"] + ". Status: " + entry["status"] + ".", "",
                 entry["description"], "",
                 "Sources: " + "; ".join(link["label"] + " - " + link["url"] for link in entry["where_to_find"]), "",
                 "Setup prompt: " + entry["setup_prompt"], ""]
    return {
        "sitemap.xml": sitemap,
        "robots.txt": robots,
        "llms.txt": "\n".join(llms),
        "llms-full.txt": "\n".join(full) + "\n",
    }


def legacy_entry_dirs() -> list[Path]:
    found = []
    for directory in ROOT.iterdir():
        page = directory / "index.html"
        if not directory.is_dir() or not page.is_file():
            continue
        if 'name="entry-slug"' in page.read_text(encoding="utf-8", errors="replace"):
            found.append(directory)
    return found


def stale_entry_dirs(slugs: set[str]) -> list[Path]:
    if not ENTRIES_DIR.is_dir():
        return []
    stale = []
    for directory in ENTRIES_DIR.iterdir():
        page = directory / "index.html"
        if not directory.is_dir() or not page.is_file():
            continue
        match = re.search(r'name="entry-slug" content="([^"]+)"', page.read_text(encoding="utf-8", errors="replace"))
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
    home, pages = build_pages(entries)
    discovery = build_discovery_files(entries)
    artifacts = {INDEX: generated, ROOT / "index.html": home}
    artifacts.update({ENTRIES_DIR / slug / "index.html": page for slug, page in pages.items()})
    artifacts.update({ROOT / name: text for name, text in discovery.items()})

    if check_only:
        for path, text in artifacts.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(f"INVALID generated file is missing or stale: {path.relative_to(ROOT)}", file=sys.stderr)
                return 1
        stale = stale_entry_dirs(set(pages)) + legacy_entry_dirs()
        if stale:
            print("INVALID stale generated pages: " + ", ".join(p.name for p in stale), file=sys.stderr)
            return 1
        if not (ROOT / "assets" / "og-image.png").is_file():
            print("INVALID missing assets/og-image.png", file=sys.stderr)
            return 1
    else:
        for path, text in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        for directory in stale_entry_dirs(set(pages)) + legacy_entry_dirs():
            shutil.rmtree(directory)

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["type"]] = counts.get(entry["type"], 0) + 1
    summary = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
    mode = "checked" if check_only else "built"
    print(f"{mode}: {len(entries)} entries ({summary}), {len(pages)} entry pages, {len(discovery)} discovery files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
