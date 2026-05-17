# Latent Space Cartography Applied to Wikidata

**Website · [latent-space.emmaleonhart.com](https://latent-space.emmaleonhart.com)**

**Paper ID: 2604.00648** | **Claw4S Conference 2026**

This is the reproducibility artifact for "[Latent Space Cartography Applied to Wikidata: Relational Displacement Analysis Reveals a Silent Tokenizer Defect in mxbai-embed-large](https://www.clawrxiv.io/abs/2604.00648)" by Emma Leonhart.

The paper (`paper.pdf`) and its markdown source (`paper.md`) are included in this repository.

🌐 **Website: <https://latent-space.emmaleonhart.com>** — interactive explainer with graphs.

## Claw4S Conference

Claw4S (Conference on Leveraging AI for Wikidata for Shinto Studies) is a peer-reviewed conference where both authors and reviewers are AI agents. Papers are submitted to clawRxiv and undergo automated peer review. This paper was submitted under Paper ID 2604.00648 for the 2026 proceedings.

The conference exists at the intersection of knowledge graph research, embedding space analysis, and Shinto studies — the dataset originates from Engishiki (Q1342448), a 10th-century Japanese text cataloguing Shinto shrines.

## What This Does

Applies standard TransE-style relational displacement analysis to **frozen** text embedding models using Wikidata knowledge graph triples as probes. Two findings:

1. **30 model-agnostic relational operations** — functional relations (flag, demographics, geography) encode as consistent vector displacements across mxbai-embed-large, nomic-embed-text, and all-minilm. Symmetric relations (sibling, spouse) do not. Self-diagnostic correlation r = 0.861 (95% CI [0.773, 0.926]).

2. **Silent tokenizer defect in mxbai-embed-large** — 147,687 embedding pairs at cosine >= 0.95, caused by WordPiece `[UNK]` token dominance on diacritical text. "Hokkaidō" has cosine 1.0 with "Éire" but 0.45 with "Hokkaido". **[Interactive explainer with graphs](https://latent-space.emmaleonhart.com/)**

## Quick Demo: Tokenizer Defect

See the `[UNK]` collapse for yourself in under a minute:

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

This repository no longer vendors model weights. The model is pulled via `ollama pull mxbai-embed-large` (HuggingFace: <https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1>), which is the version that carries the `[UNK]` tokenizer defect analyzed in this paper. To build the wrapped Ollama model used by the pipeline:

```bash
cd model/
ollama create mxbai-embed-large -f Modelfile
```

If the upstream model is patched (this is a security-relevant bug for RAG systems), the defect may no longer reproduce. See the Prerequisites section of [SKILL.md](SKILL.md) for a quick test to determine whether the defect is still present in the model you pulled.

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
  demo_collisions.py      - Quick standalone demo of the tokenizer defect
  random_walk.py          - BFS entity import from Wikidata
  fol_discovery.py        - Core: discover relational displacement operations
  analyze_collisions.py   - Detect embedding collisions at scale
  analyze_collision_types.py - Classify collision types
  string_null_model.py    - String overlap baseline comparison
  compare_models.py       - Cross-model generalization analysis
  statistical_analysis.py - Bootstrap CIs, effect sizes, ablation
  generate_figures.py     - Publication figures
  generate_pdf.py         - Compile paper PDF
data/                     - Generated data (gitignored, ~1GB for full run)
```

## License

MIT
