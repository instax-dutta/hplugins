# assets

## Purpose

Browser-facing presentation assets: CSS, JavaScript, fonts, favicon, and the Open Graph image. The HTML pages are generated from `tools/shell.html` and the data lives in `data/`; this folder consumes both. No runtime dependencies beyond what GitHub Pages serves.

## Ownership

Static presentation and client-side behavior. `data/catalog.json` content is owned by `data/AGENTS.md`; this folder only consumes it.

## Local Contracts

- Dark and light are both first-class; colors live in CSS custom properties in `assets/css/site.css`, never inline in JS.
- One accent color only. No gradients on text, no pure black or pure white surfaces.
- Clean paths: `/` is the catalog and `/entries/<slug>/` is an entry page. The old `#/p/<slug>` hash form is a fallback only. Generated entry pages carry `site-root` and `entry-slug` meta tags so the same JS works at both depths.
- Motion must be transform/opacity only and disappear under `prefers-reduced-motion: reduce`.
- JS expects `data/search-index.json`; on fetch failure it shows an inline error with a retry, never a blank page.
- Self-hosted fonts in `assets/fonts/` with `font-display: swap`. Never link a font CDN.
- Bump the `?v=` query on the CSS and JS links in the root `index.html` when either file changes so Cloudflare serves the new asset.
- Generated pages reference these assets with `{{ROOT}}assets/` in the template; never write absolute URLs into it.
- Every generated page ships readable static content, canonical URL, Open Graph/Twitter metadata, and JSON-LD. JS enhances that content; it must not be the only source of text.

## Work Guidance

- Keep the first viewport useful: the catalog list starts above the fold, no marketing hero that pushes entries down.
- Keyboard focus must be visible on every interactive element. Copy actions give visible confirmation.

## Verification

- `python3 tools/build.py --check` validates data consumed by the UI.
- Open the root `index.html` through a local HTTP server (`python3 -m http.server`) and exercise `/`, `/entries/<slug>/`, search, filters, theme toggle, and copy. The `#/p/<slug>` form is a fallback check.

## Child DOX Index

None.
