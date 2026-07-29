# Contributing

This repository is the reproduction artifact for *Latent Space Cartography Applied to
Wikidata: Relational Displacement Analysis Reveals a Silent Diacritic-Collapse Regression in
the Ollama Runtime (mxbai-embed-large)*.

Reproduction instructions are in **[`SKILL.md`](SKILL.md)** — start there, not here. This file
covers what to do when reproduction *disagrees* with the paper, and what kinds of contribution
are useful.

## Read this before filing "I can't reproduce the collisions"

**The variable you pin is the Ollama version, not the model.** This is the single most common
way a reproduction attempt goes wrong.

The defect is a regression in the Ollama serving runtime, not in mxbai-embed-large's published
weights. The registry blob is content-addressed and byte-identical across the boundary:

- **Ollama ≤ v0.13.4** — clean. Diacritical collision rate ≈ 0%, statistically
  indistinguishable from an ASCII control. If you are on one of these, **seeing no collisions
  is the expected result**, and it is a confirmation of the paper's claim rather than a
  contradiction of it.
- **Ollama ≥ v0.14.0** (2026-01-10) — defective, 10.3–11.6% collision rate, through v0.24.0.

So a report of the form "I ran the scan and found nothing" needs the Ollama version to mean
anything. Pin it explicitly:

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="0.14.0" sh
```

`scripts/resolve_versions_for_date.py` resolves the correct Ollama + model versions for a
historical date, and `.github/workflows/collision-bisect.yml` is the harness that produced the
21-release bisection in Table 11.

The CI check is deliberately **two-sided**: it asserts the scan is clean on v0.13.4 *and*
reproduces on v0.14.0. A one-sided test would pass on a build that simply fails to embed
anything, which is why both halves matter — please keep that property in any test you add.

## Reporting a reproduction discrepancy

Include, in this order:

1. **Ollama version** (`ollama --version`) — see above.
2. **Model digest** (`ollama show mxbai-embed-large --modelfile`, or the registry digest). The
   claim rests on the blob being unchanged; a different digest is a different experiment.
3. The exact script and arguments you ran.
4. The numbers you got, next to the numbers you expected. Not "the correlation was different" —
   the actual values.
5. Corpus size, if you did not use the vendored one. Several results are scale-dependent.

For the geometric results, note that `r = 0.882` (mxbai-embed-large) reproduces at `r = 0.804`
in all-minilm but only `r = 0.430` in nomic-embed-text. Model-dependent strength is a stated
finding, not a bug — see §5 of `paper.md`.

## Useful contributions

- **Other runtimes.** Whether llama.cpp, vLLM, or sentence-transformers-direct show the same
  collapse for this model is **untested**, and the paper explicitly makes no claim about them.
  A clean measurement either way is the most valuable thing anyone could add.
- **The upstream commit.** The regression is localized to the v0.13.4 → v0.14.0 range by
  bisection, but the specific commit and its internal cause are not identified from Ollama
  source. See [ollama/ollama#15609](https://github.com/ollama/ollama/issues/15609).
- **Other models through the affected runtime.** nomic-embed-text and all-minilm show related
  but distinct failure signatures; a wider sweep would establish scope.
- **Additional seeds.** The Engishiki seed reached diacritic-rich terminology by accident of
  domain. Other domain seeds may reach other benchmark-invisible regions — that is the general
  method, and the defect was a byproduct of it.

## Claims discipline

This is a paper repository, so the bar for a claim in it is the bar for a claim in the paper:

- Numbers are **measured**, not estimated or remembered. If you change a number, say which
  script produced it.
- Negative and mixed results stay in. The weaker nomic-embed-text correlation is reported, not
  smoothed away.
- Do not describe the defect as a flaw in mxbai-embed-large. It is a runtime regression, and
  the distinction is the paper's third contribution.
- The paper does not quantify how many deployments are affected, and neither should any text
  added here — the population at risk is describable, its size is not.

## Repository layout

- `paper.md` — the paper. `paper.tex` + `neurips_2026.sty` wrap it for pdflatex.
- `SKILL.md` — reproduction instructions.
- `scripts/` — the scan, bisection, cross-model comparison, statistics, and figures.
- `verification/`, `collisions.csv` — verified sample data.
- `docs/arxiv.md` — how the arXiv source bundle is built and submitted.
- `.github/workflows/` — `collisions.yml` (scan), `collision-bisect.yml` (version bisection),
  `paper-pdf.yml` (PDF + arXiv bundle), `pages.yml` (site).

Editing `paper.md` triggers the paper CI. Keep edits substantive rather than incidental.
