# hplugins

A lightweight registry of community-made and non-default Heretic plugins, scorers, configs, models, and related work. Each entry has a one-sentence description, a source link, a usage guide, and a prompt you can paste into a coding model to set it up in your own Heretic checkout.

## Why it is built this way

GitHub Pages serves static files. There is no server runtime, so the "backend" is a build-time data layer:

- `data/catalog.json` is the only source of truth.
- `tools/build.py` validates every entry, writes `data/search-index.json`, and generates one static `entries/<slug>/index.html` page per entry.
- The browser fetches the small index for the list and loads the full catalog only when an entry page is opened.
- CI runs the same validator in read-only mode on every push.

Runtime dependencies: none. No framework, no npm, no CDN, no external font request. The two self-hosted fonts total about 42 KB.

## Local use

```bash
python3 tools/build.py
python3 -m unittest discover -s tests -v
python3 -m http.server 8000
```

Open `http://localhost:8000/`. Entry URLs are clean paths such as `/entries/krea2-drift/`; the older `#/p/<slug>` links still work.

## Deploy

Push to GitHub, then in repository settings choose Pages, Deploy from a branch, `main`, `/ (root)`. No build command is needed because the generated index is committed.

## Add or update an entry

See [docs/contributing.md](docs/contributing.md). After editing `data/catalog.json`, run `python3 tools/build.py` and commit the catalog, the regenerated index, and the regenerated entry pages.

## Documentation

- `PRODUCT.md` and `DESIGN.md` - product and visual decisions
- `docs/superpowers/specs/` - design spec
- `docs/plans/` - implementation plan
- `docs/surface-briefs/` - direction contract for the catalog surface
- `HERETIC_PLUGINS.md` - extracted source inventory from the upstream repo, Discord export, and local Krea 2 files
- `AGENTS.md` plus child `AGENTS.md` files - DOX work contracts

DOX by [Agent Zero](https://www.agent-zero.ai/), [agent0ai/dox](https://github.com/agent0ai/dox), MIT.
