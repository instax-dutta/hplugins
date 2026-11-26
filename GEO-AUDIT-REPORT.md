# GEO Audit Report: hplugins

**Audit Date:** 2026-11-27
**URL:** https://hplugins.sdad.pro
**Business Type:** Publisher / community registry (technical directory)
**Pages Analyzed:** 14 (homepage + 13 entry pages)

---

## Executive Summary

**Overall GEO Score: 66/100 (Fair)** after remediation.

Before this pass the site was effectively invisible to AI crawlers: every page was a JavaScript shell with no indexable text, and there was no sitemap, robots file, llms.txt, social metadata, or structured data. That has been fixed. The remaining ceiling is off-site: hplugins has no Wikipedia, Reddit, or press footprint yet, and an AI assistant is unlikely to recommend a brand it has never seen outside its own site.

### Score Breakdown

| Category | Score | Weight | Weighted Score |
|---|---|---|---|
| AI Citability | 85 | 25% | 21.25 |
| Brand Authority | 35 | 20% | 7.00 |
| Content E-E-A-T | 65 | 20% | 13.00 |
| Technical GEO | 85 | 15% | 12.75 |
| Schema & Structured Data | 80 | 10% | 8.00 |
| Platform Optimization | 40 | 10% | 4.00 |
| **Overall GEO Score** | | | **66/100** |

---

## Critical Issues (Fix Immediately)

1. **JavaScript-only content** - all pages rendered their text in the browser, so crawlers saw an empty `<div id="view">`. Fixed: `tools/build.py` now generates readable static HTML into every page before publish.
2. **No structured data** - no JSON-LD anywhere. Fixed: `WebSite` + `ItemList` on the homepage and `Article` with author, citations, and `isPartOf` on every entry page.
3. **No crawler discovery files** - `robots.txt`, `sitemap.xml`, and `llms.txt` all returned 404. Fixed: all three are generated, plus `llms-full.txt`.

## High Priority Issues

- **No Open Graph or Twitter metadata.** Fixed: canonical URL, OG title/description/image/URL, and Twitter large-card tags on every page. A 1200x630 `assets/og-image.png` was generated.
- **AI crawlers had no explicit permission.** Fixed: `robots.txt` allows GPTBot, ClaudeBot, PerplexityBot, Google-Extended, and CCBot by name.
- **No compact corpus for LLMs.** Fixed: `llms.txt` links the catalog and full text; `llms-full.txt` contains every entry's description, sources, and setup prompt.
- **Unresolved:** no third-party brand presence. This cannot be fixed on-site.

## Medium Priority Issues

- Article schema has no publication or modification dates. No dates were fabricated; add them only when real provenance exists.
- Authors are named but have no bios or credentials on the site.
- One site-wide OG image rather than per-entry images.
- External source links were not re-checked for liveness during this audit.

## Low Priority Issues

- No RSS feed or changelog for freshness signals.
- No Twitter account handle in card metadata.
- No FAQ section or FAQPage schema (the site has no FAQ content).

---

## Category Deep Dives

### AI Citability (85/100)
Every page now ships static, self-contained prose: a one-sentence summary, a longer description, ordered usage steps, source links, and the full setup prompt. These are short, declarative blocks that an assistant can quote. Entry pages still omit dates and author biographies, which limits trust scoring.

### Brand Authority (35/100)
The site is new and has no measured Wikipedia, Reddit, YouTube, or press mentions. This is the largest remaining gap and requires off-site work; no on-site edit can manufacture it.

### Content E-E-A-T (65/100)
Entries name the original author and cite real source URLs. Provenance is factual: proof-of-concept entries are labelled as such, and no benchmarks were invented. Author credentials and freshness dates are absent, so the score is capped.

### Technical GEO (85/100)
Static HTML is generated at build time; no rendering is required to read content. robots.txt, sitemap.xml, llms.txt, canonical URLs, and social cards are present. The site is fast and has no runtime dependencies.

### Schema & Structured Data (80/100)
Homepage uses `WebSite` with an `ItemList` of all entries. Entry pages use `Article` with `headline`, `description`, `author`, `keywords`, `citation`, and `isPartOf`. Schema is valid JSON-LD; dates were deliberately omitted rather than invented.

### Platform Optimization (40/100)
`llms.txt` and explicit AI-crawler directives cover ChatGPT, Claude, Perplexity, Gemini/Google-Extended, and Common Crawl. No external platform profiles exist yet.

---

## Quick Wins (Implement This Week)

1. Keep `llms.txt` and `llms-full.txt` regenerated with every catalog change.
2. Publish one real Reddit or Hacker News post linking to the registry and ask plugin authors to link back.
3. Add real author bios where contributors agree to provide them.
4. Add a changelog page with genuine dates once entries have update history.
5. Verify external source links monthly.

## 30-Day Action Plan

### Week 1: Discovery hygiene
- [ ] Confirm Google Search Console and Bing Webmaster Tools index the sitemap
- [ ] Check that `robots.txt` and `llms.txt` remain reachable through Cloudflare

### Week 2: Entity building
- [ ] Create a Wikipedia-neutral mention only if independent coverage exists
- [ ] Publish a short project announcement on a community channel

### Week 3: Author trust
- [ ] Invite three contributors to add a one-line bio and source profile
- [ ] Add real `dateModified` values when the catalog supports them

### Week 4: Measurement
- [ ] Ask assistants which entries they can recall by name
- [ ] Re-run this audit after the next catalog release

---

## Appendix: Pages Analyzed

| URL | Title | GEO Issues Before Remediation |
|---|---|---|
| https://hplugins.sdad.pro/ | hplugins - Community-built Heretic plugins and scorers | JS-only, no schema, no discovery files |
| https://hplugins.sdad.pro/entries/krea2-drift/ | Krea2Drift | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/krea2-good/ | Krea2Good | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/krea2-bad/ | Krea2Bad | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/gibberish-rate/ | GibberishRate | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/calibrated-piqa/ | Calibrated PIQA | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/llm-judge/ | LLM Judge scorer | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/refusal-direction-tagger/ | Refusal-direction scorer/tagger | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/china-censorship-config/ | Chinese censorship config fork | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/vantablack/ | vantablack refusal classifier | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/refusal-classifier-v3/ | Refusal Classifier V3 | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/mpoa/ | MPOA abliteration | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/soma-modifier/ | SOMA modifier | JS-only, no metadata |
| https://hplugins.sdad.pro/entries/emotions-config/ | Emotion steering config | JS-only, no metadata |
