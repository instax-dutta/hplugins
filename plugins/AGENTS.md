# plugins

## Purpose

Source files for community plugins shipped with the registry so their catalog links resolve. Currently the two Krea 2 conditioning scorer files from zubzer0.

## Ownership

`data/catalog.json` owns all descriptive copy and links. This folder only stores the plugin source files those links point to.

## Local Contracts

- File names must match the `where_to_find` URLs in `data/catalog.json`; `tools/build.py` rejects local links that do not resolve to a file.
- These are contributed artifacts. Do not rewrite their contents to "improve" them; upstream attribution stays intact.
- Catalog links use the `plugins/` prefix. Setup prompts tell the user to copy the file into their own Heretic working directory, where the `./file.py:Class` plugin path applies.

## Work Guidance

- When adding a source file, add or update its catalog entry in the same change and regenerate with `python3 tools/build.py`.

## Verification

- `python3 tools/build.py --check` confirms every local catalog link resolves.

## Child DOX Index

None.
