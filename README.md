# hplugins

**Community-built extensions for Heretic, collected in one place.**

hplugins is a public registry of scorers, modifiers, configs, models, and small tools made by people who use Heretic. Each listing gives you a plain-language description, the original source, a short usage guide, and a setup prompt you can hand to a coding model.

**Visit the site:** https://hplugins.sdad.pro

## What you will find

- **Scorers** that change what Heretic optimizes for
- **Modifiers** and abliteration variants for different model behavior
- **Configs** prepared for specific jobs, such as emotion steering or language-specific refusals
- **Models and tools** that work alongside Heretic

Every entry says what the thing does, who made it, and where to get it. Nothing is hidden behind an account or a signup form.

## Using the site

1. Browse the list, or search by name, author, or keyword.
2. Open a listing to read the guide and the source links.
3. Copy the setup prompt and paste it into a coding model in the Heretic folder you use.

The prompt names the files and config keys to add, and includes a check so you can confirm the plugin loaded. The site itself never installs anything; it helps you install the work in your own Heretic checkout.

## Add your work

If you built something useful with Heretic, it belongs here. Contributions are simple:

1. Add one entry to `data/catalog.json` with a short description, source link, usage steps, and setup prompt.
2. Run `python3 tools/build.py` to check the entry and prepare its page.
3. Open a pull request.

The automatic checks validate your entry. Pull requests that only add catalog entries or documentation are merged as soon as those checks pass. Changes to code or workflows are reviewed by a maintainer.

Step-by-step instructions, including the exact entry format, are in [docs/contributing.md](docs/contributing.md).

## About

hplugins is built with plain HTML, CSS, and JavaScript and hosted on GitHub Pages. Sources stay attributed to their authors, and the catalog is open for anyone to improve.

AI assistants that support `llms.txt` can read a compact index at https://hplugins.sdad.pro/llms.txt. The full text of every entry is at https://hplugins.sdad.pro/llms-full.txt.
