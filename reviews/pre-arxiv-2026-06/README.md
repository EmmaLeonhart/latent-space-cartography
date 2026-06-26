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

## Decisions Emma made (2026-06-26) — now applied

- **Responsible-disclosure note (4/6 asked):** the regression *was* filed upstream
  — added a citation to `ollama/ollama#15609` in Section 5.4.1.
- **Abstract reframe to lead with the regression (4/6 suggested):** done — the
  abstract now opens with the Ollama defect + production impact, then frames the
  cartography method as the discovery vehicle. (Title kept as-is; it already names
  the regression. Change it on request — it ripples to `arxiv.html` and the
  clawRxiv supersedes chain.)
- **Figures (3/6 asked):** the four real figures already in `docs/figures/` (which
  had never been embedded) are now in Section 5.4 — the `[UNK]` attractor heatmap,
  the three-condition `[UNK]`-dominance histograms, the diacritic-vs-ASCII paradox,
  and the hard-collapse threshold curve. `graphicx` added to `paper.tex`; the PNGs
  are now shipped inside the arXiv source tarball.

## Still optional (not done — Emma's call)

- Soften "impact is likely substantial" (Claude); gloss "Jinmyōchō" for
  non-Japanese readers (3/6) — left to avoid terminology drift.
