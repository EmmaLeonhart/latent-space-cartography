# Pre-arXiv external AI reviews (2026-06)

Before submitting the paper to arXiv, Emma asked six AI assistants for a final
review. Their conversations are saved here (one markdown file each, extracted
from saved web pages): `claude.md`, `gemini.md`, `deepseek.md`, `grok.md`,
`meta-ai.md`, `mistral.md`. Overall verdict across all six: ready / accept with
minor revisions — no model challenged the methodology or the numbers.

## What was metabolized into the paper

Each reviewer worked from a rendered/pasted PDF, so several "blocking" issues
were **artifacts of their copy**, not the source. Verified against `paper.md`
before acting (real-numbers-only discipline):

**Fixed (real issues, confirmed in source):**
- `#### 5.4.1` rendered as "5.4.1 5.4.1" in the PDF — the build `sed` stripped
  manual numbers from `##`/`###` headings but not `####`, so LaTeX double-numbered
  it. Added a `####` strip rule to both `pages.yml` and `paper-pdf.yml`.
  (flagged by Claude, Gemini)
- "v0.13.5 → v0.14.0" (2 places) contradicted Table 11's tested boundary; the
  bisection only tested v0.13.4 (clean) and v0.14.0 (defect). Reworded to the
  v0.13.4 → v0.14.0 boundary. (DeepSeek)
- The repo URL appeared nowhere in the body (the abstract/conclusion only said
  "publicly available"). Added a **Data and Code Availability** section with the
  repository URL. (Claude, DeepSeek, Grok, Meta AI, Mistral — 5/6 consensus)
- Limitations list was misnumbered 1,2,3,4,5,4 → fixed the last item to 6.
  (found during verification)
- Table 3 referenced "supplementary" that doesn't exist for a standalone
  preprint → now points to the linked repository. (DeepSeek, Grok, Meta AI)

**Phantom (claimed by reviewers, NOT present in source — no change):**
- Stray `<fcel>` token in Table 10; Table 10 duplicated (DeepSeek) — source clean.
- Abstract truncates at "Filast…" (Meta AI) — `Filasṭīn` is complete; the ṭ is
  mapped in `paper.tex`, renders fine.
- "Englishki"/"Englishiki" misspelling (DeepSeek, Grok, Meta AI, Mistral) — the
  paper uses "Engishiki" consistently.
- Collision rate "≥ 0" vs "≈ 0" inconsistency (DeepSeek) — only "≈ 0" appears.

## Deferred — Emma's call (not changed unilaterally)

- **Responsible-disclosure note:** 4/6 asked whether the regression was reported
  to Ollama. Can't state it without knowing if it's true. Add a sentence if/when
  filed.
- **Title / abstract reframe to lead with the regression:** 4/6 suggested it;
  it's an authorial framing decision. Current title already names the regression.
- **Figures (t-SNE/UMAP of the [UNK] attractor, bisection timeline):** 3/6 asked
  for visualizations. A real generation task, not a copy edit.
- **Minor editorial:** soften "impact is likely substantial" (Claude); gloss
  "Jinmyōchō" for non-Japanese readers (3/6) — left to avoid terminology drift.
