# devlog

## 2026-07-02 — Fixes from reviews/arxiv-hold-analysis-2026-07-01.md

Worked through every fixable item in the arXiv-hold analysis report:

- **§2a (paper.md §4.6):** removed the "identical input" claim. The three
  cross-model runs used the same seed (Q1342448) and `--limit 500` but the BFS
  imports ran at different times against live Wikidata, so entity/alias sets and
  embedding counts differ; the text now says so and states that the comparison is
  made over shared predicates (matching `scripts/compare_models.py`).
- **§2b (collision denominators):** 16,067 + 74,760 = 90,827 is the combined
  two-seed embedding store (Engishiki BFS 41,725 embeddings + broad P31
  country-level sampling), confirmed by `old/UNIT_VECTOR_ANALYSIS.md` (90,827-row
  store) and `old/scripts/measure_collapse_geometry.py` (loads the full store);
  the collision examples themselves (Éire, România, Djazaïr) are country-level
  entities from the P31 seed. Fixed §3.2 (line 93), §5.4 "(of 41,725)" → "(of the
  90,827 in the combined two-seed store)", §5.4.1 geometry sentence, §5.6
  limitation 2, and aligned "16,067 entities" → "16,067 embedded labels" in §5.4
  and the conclusion. Residual: the definitive fix is a re-run of the collision
  scan on a frozen entity snapshot; the text edit makes the paper internally
  consistent with the evidence available.
- **§3 tone:** "silent and likely exploitable" → "silent and consequential"
  (no threat model was shown; also reduces the security-disclosure surface a
  moderator might react to).
- **§3 hygiene:** stripped the leftover CI HTML comment from the references.
- **§3 metadata:** README updated to the current runtime-regression title, paper
  ID 2604.00648 → 2604.01127, Model section reframed (weights healthy, regression
  is Ollama ≥ v0.14.0, upstream issue linked); GitHub repo description updated
  via `gh repo edit` to the same framing.
- **§1 (venue/endorsement):** not fixable by editing the paper — reported back to
  Emma as a decision she has to make (arXiv category/endorsement, independent-
  preprint framing vs. Claw4S provenance).
