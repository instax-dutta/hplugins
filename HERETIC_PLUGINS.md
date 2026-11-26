# Heretic plugins / scorers / scores - extracted inventory

Compiled 2026-11-27 from:

- `p-e-w/heretic` master (shallow clone, 241 commits): `src/heretic/scorers/`, `src/heretic/modifiers/`, `config.*.toml`
- Discord `#general` export `general-1795720340987.json`: 19,685 messages, 2025-12-09 to 2026-11-26
- Local files `krea2_drift.py` and `krea2_drift_v2.py`

## 1. Plugin model

Base classes: `Scorer` (`src/heretic/scorer.py`) and `Modifier` (`src/heretic/modifier.py`), both under `Plugin` (`src/heretic/plugin.py`).

Plugin reference forms accepted in config:

- `heretic.scorers.keyword_rate.KeywordRate` (fully qualified; required for reproducibility)
- `./krea2_drift.py:Krea2Drift` (file path + class)

`optimization` values: `minimize`, `maximize`, `none` (compute only, no influence). Config sections are `[scorer.<ClassName>]` or `[scorer.<ClassName>_<instance_name>]`.

Plugin types actually shipped: **scorer**, **modifier**. Types discussed/planned in Discord: **tagger** (early 1.x scorer/tagger branch), **logger** ("Heretic Next"/2.0), **dataset provider**, **optimizer**; **SOMA** planned as an additional built-in modifier.

## 2. Built-in scorers (5)

| Scorer | Plugin path | Config section | Score | Typical direction |
| --- | --- | --- | --- | --- |
| KeywordRate | `heretic.scorers.keyword_rate.KeywordRate` | `[scorer.KeywordRate]` | Share of responses containing refusal keyword markers | minimize (can be maximized, e.g. to promote keywords) |
| KLDivergence | `heretic.scorers.kl_divergence.KLDivergence` | `[scorer.KLDivergence.prompts]` | KL divergence from the unmodified model | minimize |
| BERTClassifier | `heretic.scorers.bert_classifier.BERTClassifier` | `[scorer.BERTClassifier]` | Share of responses whose predicted label is in `match_labels` (ModernBERT refusal model by default) | minimize (PR 500) |
| BenchmarkScore | `heretic.scorers.benchmark_score.BenchmarkScore` | `[scorer.BenchmarkScore]` | Any lm-evaluation-harness task/metric (default `piqa`, `acc_norm,none`) | maximize |
| ResponseLength | `heretic.scorers.response_length.ResponseLength` | `[scorer.ResponseLength]` | Mean response length, with short-response handling | minimize |

## 3. Built-in modifiers (2)

| Modifier | Plugin path | Notes |
| --- | --- | --- |
| ARA | `heretic.modifiers.ara.ARA` | Arbitrary-Rank Ablation, default since 2.0 |
| Abliteration | `heretic.modifiers.abliteration.Abliteration` | Directional ablation; supports projected abliteration, MPOA / norm preservation, weight kernels |

Planned: SOMA (multi-directional transport modifier, expected as a default plugin).

## 4. Official config presets and their scorer stacks

| Config | Purpose | Active scorers |
| --- | --- | --- |
| `config.default.toml` | Refusal suppression | KeywordRate (min), KLDivergence (min) |
| `config.classifier.toml` | Classifier-based refusal suppression | BERTClassifier (min), KLDivergence (min) |
| `config.emotions.toml` | Emotion steering | KLDivergence (min), BERTClassifier `negative_emotions` (min), BERTClassifier `positive_emotions` (max) |
| `config.piqa.toml` | Quality via benchmark | KeywordRate (min), BenchmarkScore (max) |
| `config.terse.toml` | Shorter answers | ResponseLength (min), BenchmarkScore (max) |
| `config.noslop.toml` | Slop reduction | KeywordRate + KLDivergence (default list, retuned settings) |
| `config.nohumor.toml` | Humor reduction | KeywordRate + KLDivergence (default list, retuned settings) |
| `config.nosyco.toml` | Sycophancy reduction | KeywordRate + KLDivergence (default list, retuned settings) |

## 5. Community / custom scorers and plugins found in Discord

| Name / class | Author | Source | What it scores | Status |
| --- | --- | --- | --- | --- |
| `Krea2Drift` (`./krea2_drift.py:Krea2Drift`) | zubzer0 | local file (2026-11-23 post) | Mean cosine drift of the fused Krea 2 text conditioning vs baseline; `score_name` "Krea 2 drift" | Personal-use plugin; config uses `optimization = "none"` (reference only) |
| `Krea2Good` (`./krea2_drift_v2.py:Krea2Good`) | zubzer0 | local file (2026-11-26 post); header calls it `krea2_drift_v2.1.py` | p90max drift of good-image prompts, in percent; "Krea2 good p90max" | minimize (keeps good prompts stable) |
| `Krea2Bad` (`./krea2_drift_v2.py:Krea2Bad`) | zubzer0 | same file | p90min drift of bad-image prompts, in percent; "Krea2 bad p90min" | maximize (moves bad-image prompts) |
| `GibberishRate` (`[scorer.GibberishRate]`) | randomlygeneratedusername (Claude-assisted) | Discord config snippet, 2026-11-04, no repo link | Detects gibberish via distinct Unicode scripts (`max_scripts`, `min_characters_per_script`); skips/aborts trials | Config-only community scorer, no source posted |
| Custom benchmark scorer ("calibrated PIQA") | electroglyph | https://github.com/electroglyph/heretic | Benchmark score on flip-flopping PIQA questions, used for GPQA-correlated runs | Working fork with custom scorer |
| LLM judge scorer | kurigohanandkamehameha | Heretic "LLM judge" PR (discussed 2026-03 to 2026-11) | LLM classifies refusals/compliance via OpenAI-compatible API | PR/proposal, not part of shipped scorers as of export |
| Refusal-direction scorer/tagger | us_beast_1 | Plugin branch, Dec 2025 | Compares initial refusal directions with response hidden states | Early plugin-system development branch |
| Chinese-censorship config/fork | benjamin920101 | https://github.com/benjamin920101/heretic-china-censorship | Custom refusal markers for Chinese censorship only | Config fork, not a scorer plugin |
| MPOA / Magnitude-Preserving Orthogonal Ablation | p-e-w / PR 52 | https://github.com/p-e-w/heretic/pull/52 | Modifier technique (not a score) | Implemented in `Abliteration` modifier |

### Classification / judge models used by these scorers

| Model | Used with | Notes |
| --- | --- | --- |
| `VINAY-UMRETHE/Refusal-Classifier`, `-V2`, `-V3-Small`, `-V3-Large` | BERTClassifier | Default recommended replacement; V3-Large reported 100% on 45 stress cases |
| `saidutta69/vantablack` | BERTClassifier | Config: `model = "saidutta69/vantablack"`, `match_labels = ["full_refusal", "partial_refusal"]`, `use_text_pairs = true`, `max_length = 512` |
| `NousResearch/Minos-v1` | refusal classification experiment | Tested by maintainer, weaker than keyword matching |
| `mradermacher/Crow-9B-Opus-4.6-Distill-Heretic_Qwen3.5-GGUF`, Gemma 4 31B | LLM judge experiments | Judge models, not Heretic plugins |

## 6. Example config snippets (extracted verbatim)

Default scoring:

```toml
scorers = [
    { plugin = "heretic.scorers.keyword_rate.KeywordRate", optimization = "minimize" },
    { plugin = "heretic.scorers.kl_divergence.KLDivergence", optimization = "minimize" },
]
```

Keyword maximization + classifier (community config, 2026-11-02):

```toml
scorers = [
    { plugin = "heretic.scorers.keyword_rate.KeywordRate", optimization = "maximize" },
    { plugin = "heretic.scorers.bert_classifier.BERTClassifier", optimization = "minimize" },
    { plugin = "heretic.scorers.kl_divergence.KLDivergence", optimization = "minimize" },
]
```

Krea 2 pair (2026-11-26):

```toml
{ plugin = "heretic.scorers.kl_divergence.KLDivergence", optimization = "minimize" },
{ plugin = "./krea2_drift_v2.py:Krea2Good", optimization = "minimize" },
{ plugin = "./krea2_drift_v2.py:Krea2Bad", optimization = "maximize" },

[scorer.Krea2Good]
checkpoint_path = "Krea2_turbo_bf16.safetensors"
max_length = 512
max_batch_tokens = 4096
measure_noise_floor = true
noise_limit = 0.01

[scorer.Krea2Bad]
checkpoint_path = "Krea2_turbo_bf16.safetensors"
max_length = 512
max_batch_tokens = 4096
measure_noise_floor = true
noise_limit = 0.01

[scorer.Krea2Good.prompts]
dataset = "GOOD_prompts.txt"
split = "[:200]"

[scorer.Krea2Bad.prompts]
dataset = "BAD_prompts.txt"
split = "[:200]"
```
