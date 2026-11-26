# hplugins implementation plan

1. Install DOX root contract and child AGENTS.md tree. Done before code.
2. Write `PRODUCT.md`, design spec, surface brief, and this plan.
3. Author `data/catalog.json` from `docs/HERETIC_PLUGINS.md` with non-default community entries only.
4. Write `tools/build.py` with validation and index generation; write `tests/test_build.py`.
5. Write `index.html`, `assets/css/site.css`, and `assets/js/app.js` for the exploded-manifest UI.
6. Self-host the two fonts and verify total transfer size.
7. Add `.github/workflows/check.yml` and `README.md` and `docs/contributing.md`.
8. Run tests, build check, and a local browser smoke test at desktop and mobile widths.
9. DOX closeout: update owning AGENTS.md files and write `DESIGN.md` from the built world.

Verification evidence for each step lives in the turn output: command results and file contents.
