# Contribute a part to hplugins

Thanks for helping people find good Heretic work. The site is a static registry: you add one entry to a JSON file, and the build layer validates it, regenerates the search index, and publishes it. No coding framework is involved.

## What can be listed

Community-made or non-default Heretic work: scorers, modifiers, configs, companion models, forks, and tools. Built-in default scorers are documented in `HERETIC_PLUGINS.md` and are not listed as entries.

## Step 1: add one entry

Open `data/catalog.json` and add one object to the array. Use this exact shape:

| Field | Rule |
| --- | --- |
| `slug` | lowercase words separated by hyphens, unique, used in the page URL |
| `name` | the name people use for the plugin |
| `type` | one of `scorer`, `modifier`, `config`, `model`, `fork`, `tool` |
| `author` | who made it, or `community` when unnamed |
| `summary` | one sentence under 160 characters; this is what the list shows |
| `description` | a short paragraph: what it does and when to use it |
| `status` | one of `available`, `community`, `proof-of-concept`, `planned` |
| `tags` | non-empty array of lowercase search terms |
| `where_to_find` | non-empty array of `{ "label": "...", "url": "https://..." }` |
| `how_to_use` | non-empty array of ordered steps a visitor can follow |
| `config_snippet` | the exact `config.toml` fragment, or an empty string |
| `setup_prompt` | a complete prompt for a coding model with no Heretic context |

A minimal entry:

```json
{
  "slug": "example-scorer",
  "name": "Example Scorer",
  "type": "scorer",
  "author": "community",
  "summary": "One sentence describing what it measures.",
  "description": "A short paragraph about the problem it solves and when to use it.",
  "status": "community",
  "tags": ["scorer", "example"],
  "where_to_find": [
    { "label": "Source repository", "url": "https://github.com/owner/repo" }
  ],
  "how_to_use": [
    "Copy the plugin file into your Heretic working directory.",
    "Add the scorer entry to config.toml.",
    "Run a short trial and check the reported score."
  ],
  "config_snippet": "scorers = [\n    { plugin = \"./example.py:ExampleScorer\", optimization = \"minimize\" },\n]",
  "setup_prompt": "Install Example Scorer in my Heretic checkout. Copy example.py next to config.toml, add { plugin = \"./example.py:ExampleScorer\", optimization = \"minimize\" } to the scorers list, keep KLDivergence minimizing, and verify with a short trial that a score line is printed. Do not change other scorers."
}
```

### Writing the setup prompt

Write it for someone who has a Heretic checkout but has never seen this plugin. It must:

- name the plugin and its exact file path or model ID;
- state the config keys and optimization direction to add;
- include one command or log line that proves the setup loaded;
- avoid inventing benchmarks, licenses, or capabilities.

Do not invent links or numbers. If the only source is a chat discussion, link that discussion and use `community` or `proof-of-concept` as the status. A site-relative link must point to a file that exists in the repository; the validator rejects dead local links.

## Step 2: regenerate and verify locally

```bash
python3 tools/build.py
python3 -m unittest discover -s tests -v
python3 -m http.server 8000
```

Then visit `http://localhost:8000/#/p/<your-slug>` and confirm the entry renders with its guide, source links, and setup prompt.

## Step 3: open a pull request

Commit `data/catalog.json`, the regenerated `data/search-index.json`, and the regenerated `entries/<slug>/index.html` entry page. Do not hand-write entry pages; `tools/build.py` derives them from `index.html`, and the live URL becomes `https://hplugins.sdad.pro/entries/<slug>/`. Use this PR format:

**Title:** `Add <Name> (<type>)` or `Update <Name>`

**Body:**

```markdown
## What changed
One or two sentences about the entry.

## Source
Where the plugin comes from, with a link.

## Checklist
- [ ] I ran `python3 tools/build.py`
- [ ] I ran `python3 -m unittest discover -s tests -v`
- [ ] I checked `#/p/<slug>` locally
- [ ] I did not invent links, licenses, or benchmark numbers
```

### What CI checks

Every PR runs the `check catalog` workflow: catalog schema validation, generated-index freshness, all tests, JSON parsing, JavaScript syntax, and a check that `catalog.json` and `search-index.json` move together.

### Auto-merge policy

PRs that change only catalog data, docs, or Markdown are merged automatically once every check passes on the exact PR commit and there are no unresolved review comments. PRs that touch code, workflows, tests, or assets need a human review. If your PR is not merged automatically, that is expected for those files; a maintainer will take it from there.
