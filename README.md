# Latent Space Cartography Applied to Wikidata

**Website · [latent-space.emmaleonhart.com](https://latent-space.emmaleonhart.com)**

**Paper ID: 2604.01127** | **Claw4S Conference 2026**

This is the reproducibility artifact for "[Latent Space Cartography Applied to Wikidata: Relational Displacement Analysis Reveals a Silent Diacritic-Collapse Regression in the Ollama Runtime (mxbai-embed-large)](https://www.clawrxiv.io/abs/2604.01127)" by Emma Leonhart.

The paper (`paper.pdf`) and its markdown source (`paper.md`) are included in this repository.

🌐 **Website: <https://latent-space.emmaleonhart.com>** — interactive explainer with graphs.

## Claw4S Conference

Claw4S (Conference on Leveraging AI for Wikidata for Shinto Studies) is a peer-reviewed conference where both authors and reviewers are AI agents. Papers are submitted to clawRxiv and undergo automated peer review. This paper was submitted under Paper ID 2604.01127 for the 2026 proceedings.

The conference exists at the intersection of knowledge graph research, embedding space analysis, and Shinto studies — the dataset originates from Engishiki (Q1342448), a 10th-century Japanese text cataloguing Shinto shrines.

## What This Does

Applies standard TransE-style relational displacement analysis to **frozen** text embedding models using Wikidata knowledge graph triples as probes. Two findings:

1. **30 model-agnostic relational operations** — functional relations (flag, demographics, geography) encode as consistent vector displacements across mxbai-embed-large, nomic-embed-text, and all-minilm. Symmetric relations (sibling, spouse) do not. Self-diagnostic correlation r = 0.861 (95% CI [0.773, 0.926]).

2. **Silent diacritic-collapse regression in the Ollama runtime (serving mxbai-embed-large)** — 147,687 embedding pairs at cosine >= 0.95, caused by `[UNK]` token dominance on diacritical text. "Hokkaidō" has cosine 1.0 with "Éire" but 0.45 with "Hokkaido". The model weights are healthy: the byte-identical blob is clean on Ollama ≤ v0.13.4 and defective on ≥ v0.14.0 (bisected over 21 releases). **[Interactive explainer with graphs](https://latent-space.emmaleonhart.com/)**

## Quick Demo: Diacritic-Collapse Regression

See the `[UNK]` collapse for yourself in under a minute (requires Ollama ≥ v0.14.0; the defect does not reproduce on ≤ v0.13.4):

```bash
pip install -r requirements.txt
ollama pull mxbai-embed-large
python scripts/demo_collisions.py
```

This embeds pairs like "Hokkaidō" vs "Éire" and shows they have cosine 1.0 despite being completely unrelated. Pre-computed results are in `collisions.csv`.

## Full Pipeline

```bash
# Import 100 entities from Wikidata (10-15 min)
python scripts/random_walk.py Q1342448 --limit 100

# Discover relational operations (5-15 min)
python scripts/fol_discovery.py --min-triples 5

# Detect tokenizer collisions at scale
python scripts/analyze_collisions.py --threshold 0.95
```

Full reproducibility instructions with expected outputs: [SKILL.md](SKILL.md)

## Model

This repository no longer vendors model weights. The model is pulled via `ollama pull mxbai-embed-large` (HuggingFace: <https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1>). The `[UNK]` collapse analyzed in the paper is **not** a flaw in these weights — it is a regression in the Ollama serving runtime, introduced in v0.14.0 (2026-01-10); the same model blob is healthy on Ollama ≤ v0.13.4. To build the wrapped Ollama model used by the pipeline:

```bash
cd model/
ollama create mxbai-embed-large -f Modelfile
```

If the Ollama runtime fixes the regression in a future release, the defect may no longer reproduce on that release. See the Prerequisites section of [SKILL.md](SKILL.md) for a quick test to determine whether the defect is present in your Ollama version (clean on ≤ v0.13.4, reproduces on v0.14.0 through at least v0.24.0). The regression is reported upstream at <https://github.com/ollama/ollama/issues/15609>.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) with `mxbai-embed-large` model
- No GPU required

## Repository Structure

```
paper.pdf                 - Published paper
paper.md                  - Paper source (markdown)
SKILL.md                  - Full reproducibility instructions with expected outputs
collisions.csv            - Pre-computed collision data
model/
  Modelfile               - Ollama model definition (builds the ollama model)
scripts/
  demo_collisions.py      - Quick standalone demo of the runtime regression
  random_walk.py          - BFS entity import from Wikidata
  fol_discovery.py        - Core: discover relational displacement operations
  analyze_collisions.py   - Detect embedding collisions at scale
  analyze_collision_types.py - Classify collision types
  string_null_model.py    - String overlap baseline comparison
  compare_models.py       - Cross-model generalization analysis
  statistical_analysis.py - Bootstrap CIs, effect sizes, ablation
  generate_figures.py     - Publication figures
```

`paper.pdf` is built from `paper.md` by `.github/workflows/paper-pdf.yml`
(pandoc + latexmk with the NeurIPS 2026 style, `neurips_2026.sty` /
`paper.tex`) and committed back to the repo and `docs/paper.pdf`.

```
data/                     - Generated data (gitignored, ~1GB for full run)
```

## License

MIT
