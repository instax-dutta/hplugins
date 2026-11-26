# hplugins

A static registry of community-made Heretic plugins, scorers, configs, models, and related work. Each entry has a one-sentence description, source links, a usage guide, and a setup prompt you can paste into a coding model to install it in your own Heretic checkout.

Live site: https://hplugins.sdad.pro

## How it works

GitHub Pages serves static files only, so the "backend" is a build-time data layer:

- `data/catalog.json` is the single source of truth.
- `tools/build.py` validates it, writes `data/search-index.json`, and derives one static page per entry.
- The browser fetches the small index for the catalog list and the full catalog only when an entry page opens.
- CI runs the same validation on every push and pull request. A guarded workflow auto-merges PRs that keep only catalog data, docs, and generated entry pages, provided every check passes on the exact PR commit.

No framework, no npm, no CDN, no external font request. Total browser code and data is about 130 KB; fonts add about 54 KB.

## Repository layout

| Path | Purpose |
| --- | --- |
| `index.html` | Static catalog shell served by Pages |
| `assets/` | CSS, JavaScript, fonts, favicon |
| `data/catalog.json` | Authored catalog entries |
| `data/search-index.json` | Generated search index; never hand-edit |
| `entries/<slug>/index.html` | Generated entry pages; never hand-edit |
| `plugins/` | Shipped Krea 2 scorer source files linked from the catalog |
| `docs/` | Contributor guide, spec, plan, surface brief, and extracted source inventory |
| `tools/` | Validator, entry-page generator, and auto-merge guard |
| `tests/` | Offline `unittest` checks |
| `.github/workflows/` | CI validation and auto-merge |

Root files are limited to what Pages and the project tooling require: `CNAME`, `AGENTS.md`, `PRODUCT.md`, `DESIGN.md`, and this README.

## Run locally

```bash
python3 tools/build.py
python3 -m unittest discover -s tests -v
python3 -m http.server 8000
```

Open `http://localhost:8000/` for the catalog and `http://localhost:8000/entries/krea2-drift/` for an entry. The older `#/p/<slug>` links still resolve.

## Contribute

See `docs/contributing.md`. Add one entry to `data/catalog.json`, run `python3 tools/build.py`, and commit the catalog, the regenerated search index, and the regenerated entry page. Pull requests that stay inside catalog data, docs, and generated entry pages are auto-merged once validation passes; code and workflow changes need a maintainer review.

## Deploy

Push to `main`. Pages is configured to deploy the root of `main` at `hplugins.sdad.pro`. No build command runs on Pages, so generated files must be committed.

## Verification

```bash
python3 tools/build.py --check
python3 -m unittest discover -s tests -v
```

The workflow `check catalog` runs the same checks in CI. `auto-merge verified PRs` merges only after every check is green on the verified head commit.
