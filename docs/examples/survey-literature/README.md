# Literature Survey

Run [**EvoScientist**](https://github.com/EvoScientist/EvoScientist) with the [`paper-navigator`](https://github.com/EvoScientist/EvoSkills/tree/main#-paper-navigator--academic-paper-discovery--reading) skill to produce a conference-grade literature survey from a topic.

## What it does

You give EvoScientist:

- a research topic or venue scope (e.g. *SIGIR 2026 full papers on arXiv*)
- an LLM provider
- the `paper-navigator` skill installed

It then:

1. decomposes the topic into 4–6 variant queries (sub-topic angles, terminology variants)
2. discovers candidate papers via arXiv / Semantic Scholar / citation traversal
3. deduplicates and clusters them into a thematic taxonomy
4. reads each paper with structured evaluation (TLDR, contribution, limitation)
5. drafts the survey section-by-section with inline numbered citations
6. writes the English version first, then translates it into Chinese — both saved as local `.md` with a reference list linked back to arXiv

## Start here

The full worked output for this run:

- **English** — [`SIGIR2026_public_arxiv_systematic_survey_en.md`](SIGIR2026_public_arxiv_systematic_survey_en.md)
- **中文** — [`SIGIR2026_public_arxiv_systematic_survey_zh.md`](SIGIR2026_public_arxiv_systematic_survey_zh.md)

Both cover 68 arXiv-public papers, organized into 10 sections (abstract → thematic map → 4 method fronts → evaluation/fairness → cross-theme trends → open questions → conclusion → references).

## Quick start

Requirements: an OpenRouter API key (this run used `gpt-4`; any capable model works).

Install EvoScientist and pick a provider:

```bash
uv tool install EvoScientist
EvoSci onboard
```

![Onboard model selection](assets/model_selection.png)

Install the `paper-navigator` skill:

```bash
/evoskills
```

![Skill picker](assets/skill_selection.png)

Run EvoScientist and paste the prompt:

```
Use paper-navigator to write a systematic survey of SIGIR 2026 papers
publicly available on arXiv. Generate the English version first, then
translate it into Chinese. Save both as local .md files.
```

![Prompt input](assets/prompt.png)

The two Markdown files will be written into your current workspace.

## Configuration

This example was generated with:

| Setting | Value |
|---|---|
| Provider | OpenRouter |
| Model | `gpt-4` |
| Skill | `paper-navigator` (v1.1.0) |

To swap providers, re-run `EvoSci onboard` or edit `~/.config/evoscientist/config.yaml`.

## Adapting this to your own venue

- **Different conference / year** — change the topic scope in the prompt; `paper-navigator` handles arXiv filtering automatically.
- **Single-language output** — drop the "both English and Chinese" instruction.
- **Narrower survey** — add a sub-topic constraint (e.g. *focus only on retrieval-augmented reasoning*).
- **Tighter citation budget** — cap the paper count in the prompt (*at most 30 papers*).
