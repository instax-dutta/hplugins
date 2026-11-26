# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated to the agent by the user ("choose the stack yourself"). Chosen: static HTML + CSS + vanilla JavaScript, no framework, no npm; `tools/build.py` uses Python 3 stdlib to validate the catalog and generate the search index. Deploy target is GitHub Pages, which sets the hard constraint that no server runtime is available.

## Users

Primary: people who modify language models with Heretic and want to find and install community-made scorers, modifiers, configs, and companion models. They arrive with a specific problem (refusal detection, quality preservation, image-generator conditioning, slop) and need a usable setup, not a discussion thread. Secondary: plugin authors who want a stable place to publish their work with attribution.

## Product Purpose

Make non-default Heretic work discoverable and installable. Success means a visitor can find an entry, understand what it does in one sentence, reach its source, and leave with a prompt they can paste into a coding model to set it up in their own Heretic installation.

## Positioning

Each entry carries a deterministic, copy-paste setup prompt written for a coding model with no Heretic context. The registry is therefore not a link list; it is a hand-off between documentation and the user's own coding agent.

## Operating Context

Entries come from upstream repository documentation, pull requests, and the Heretic community Discord. Links are sometimes discussions rather than released packages, so entry status must say what exists. Installation always happens locally in the user's Heretic working directory via `config.toml`.

## Capabilities and Constraints

- Static-only hosting on GitHub Pages; no API, no database, no user accounts.
- Catalog is a JSON file in the repository; contributions happen through pull requests.
- Dark and light presentation, keyboard navigation, and reduced-motion behavior are required.
- Inferred from the brief: the site name is `hplugins`, and listings cover community or non-default material only, not Heretic built-ins.

## Brand Commitments

Name: `hplugins`. No existing logo, palette, or font contract was supplied. Voice is factual and technical; no hype copy.

## Evidence on Hand

`docs/HERETIC_PLUGINS.md` contains the extracted inventory from `p-e-w/heretic` master, a 19,685-message Discord export (2025-12-09 to 2026-11-26), and two local Krea 2 scorer files. No visual assets, screenshots, or brand files were provided. Do not fabricate screenshots, testimonials, benchmarks, or licenses.

## Product Principles

- A one-sentence description beats a feature list.
- Every entry must lead to a source and to an actionable setup prompt.
- Attribution is part of the artifact, never an afterthought.
- The site works offline after first load and stays small enough to open on a slow connection.

## Accessibility & Inclusion

No product-specific standard was stated. Applied as a working floor: visible keyboard focus, semantic headings, sufficient contrast in both themes, `prefers-reduced-motion` respected. This is an inferred floor, not a confirmed user requirement.
