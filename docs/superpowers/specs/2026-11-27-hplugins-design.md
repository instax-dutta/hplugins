# hplugins design spec

Date: 2026-11-27. Path: architectural. The user delegated stack and design choices and directed immediate implementation, so this spec records the decision rather than pausing for approval.

## Goal

A lightweight public listing of community-made and non-default Heretic plugins, scorers, configs, models, and related artifacts. Each entry has a name, a one-sentence description, a usage guide, a source link, and a prompt that can be pasted into any coding model to set the artifact up in that user's Heretic.

## Constraints

- GitHub Pages serves static files. There is no server runtime, no database, and no secrets.
- Runtime dependency budget is zero: no framework, no npm, no CDN.
- Build dependency budget is Python 3 standard library.
- Dark and light themes, keyboard access, visible focus, reduced motion.
- Source inventory is `HERETIC_PLUGINS.md`; no link, author, or benchmark may be invented.

## Architecture

1. `data/catalog.json` is authored by hand. It is the only source of truth.
2. `tools/build.py` validates every entry and emits `data/search-index.json`, a small array of search tokens and display fields. `--check` runs the same validation read-only for CI.
3. `index.html` is the static catalog shell. `tools/build.py` derives one `/<slug>/index.html` per entry from that shell, with `site-root` and `entry-slug` meta tags. `assets/js/app.js` fetches the index, routes on the clean path (with `#/p/<slug>` as a fallback), filters, sorts, renders the list or entry article, persists theme in `localStorage`, and copies prompts through the Clipboard API with a `document.execCommand` fallback.
4. `assets/css/site.css` owns all tokens and layout. Fonts are self-hosted woff2 in `assets/fonts/`.

## Data flow

Author entry -> `build.py` validates -> generated index and entry pages -> browser fetch -> route + filter -> list/article render -> copy prompt. There is no write path from the browser.

## Error handling

- Fetch failure: inline error panel with a retry button.
- Unknown slug: "entry not found" panel with a link back to the catalog.
- Empty filter result: "no entries match" with a clear-filters action.
- Clipboard denied: show the prompt selected with a short instruction to copy manually.

## Testing

- `unittest` suite covers schema rules, slug uniqueness, prompt presence, link shape, and generated index shape.
- `tools/build.py --check` runs in CI on every push and pull request.
- Browser verification covers list, entry page, filters, theme toggle, copy, clean URLs, and mobile width.

## Scope boundaries

Built-in default scorers are not catalog entries. The site does not host plugin files, run Heretic, or accept submissions at runtime. Contributions are pull requests.
