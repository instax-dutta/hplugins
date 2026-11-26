# Contributing an entry

Entries live in `data/catalog.json`. One entry per plugin or artifact. Built-in default scorers are not listed.

1. Add an object with these fields:

   | Field | Rule |
   | --- | --- |
   | `slug` | lowercase words separated by hyphens, unique |
   | `name` | the name people use for it |
   | `type` | `scorer`, `modifier`, `config`, `model`, `fork`, or `tool` |
   | `author` | who made it, or `community` when unnamed |
   | `summary` | one sentence, under 160 characters |
   | `description` | a short paragraph of what it does and when to use it |
   | `status` | `available`, `community`, `proof-of-concept`, or `planned` |
   | `tags` | short lowercase search terms |
   | `where_to_find` | array of `{ "label": "...", "url": "https://..." }` |
   | `how_to_use` | array of ordered strings the visitor can follow |
   | `config_snippet` | the exact `config.toml` fragment, or empty string |
   | `setup_prompt` | a complete prompt for a coding model with no Heretic context |

2. Write the setup prompt as if the reader has a Heretic checkout but no knowledge of the plugin. Name the plugin, its path or model ID, the config keys to add, and one command or check that proves it loaded.
3. Run `python3 tools/build.py`. It validates the catalog and regenerates the search index.
4. Run `python3 -m unittest discover -s tests -v`.
5. Open the site locally with `python3 -m http.server 8000` and check `#/p/<your-slug>` renders.
6. Commit both `data/catalog.json` and the regenerated `data/search-index.json`.

Do not invent links or benchmark numbers. If the only source is a chat discussion, link it and set `status` to `community` or `proof-of-concept`.
