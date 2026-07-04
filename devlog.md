# devlog

## 2026-07-03 — Paper numbers updated to the regenerated frozen snapshot (Emma's go-ahead)

Every statistic in paper.md now comes from the 2026-07-03 frozen
100,113-embedding snapshot. Supporting work done in the same pass:

- Ported `measure_collapse_geometry.py` into `scripts/` (vectorized —
  the old per-element Python loops were infeasible at 100k). Results:
  18,019 colliding embeddings, 2.9x density ratio, 77% in densest
  quartile, isolation ratio 0.98 — every old geometry claim replicates.
- String null model re-run with a new `LSC_NULL_MAX_TRIPLES` cap (10,
  seeded) because the uncapped run projected to days at the new triple
  counts; cap disclosed in the paper. Vector beats string 27/27, mean
  0.850 vs 0.019 (44x).
- Bootstrap stats re-run: r = 0.882 [0.803, 0.940] (was 0.861),
  Cohen's d = 2.906 (was 3.092).
- **New finding promoted into the paper:** colliding diacritical inputs
  receive byte-identical float vectors (verified by exact array
  equality), not merely cosine ~= 1.0. This directly answers the
  "exactly 1.000 is mathematically suspect" review criticism.
- Composition table, Tables 1-10, failure analysis (P31 now 0.202 over
  1,759 triples), cross-model section (all three models on byte-
  identical input; 33 universal ops; r = 0.901/0.436/0.623) updated.
- docs figures regenerated live on Ollama v0.17.1; heatmap title made
  computed instead of hardcoded (was "361 of 380", measured 342 of 380
  = exactly Sao Paulo's row+column, single-exception story intact).
- SKILL.md headline numbers + paper ID updated; Data Availability now
  calls collisions.csv a verified sample (it is 25 rows, not the full
  set).
- Cron prompts recreated without the stale "numbers update is blocked"
  clause (it had kept announcing a blocker Emma had already cleared).

Scope notes stated in the paper rather than hidden: prediction/overlap
statistics cover each model's top-50 operations; the null model uses
the 10-triple cap; the new collision scan ran on Ollama v0.17.1.

## 2026-07-03 — Productivity files brought to cleanvibe standards

Created `todo.md` (long-term horizon: arXiv transformation with its
sub-goals — venue decision, regenerated-numbers adoption, tautology
fix, actual-cosine reporting, nomic caveat, single-crawl §3.2
simplification — plus review-loop and infrastructure horizons) and
rewrote `queue.md` to the concrete/delete-on-done convention. Emma's
2026-07-02 conversational queue prose is preserved as intent in
todo.md; her CI/CD-revival ask was already satisfied (Actions green,
including the daily two-sided collision check). This completes the
"bring productivity stuff up to cleanvibe standards" queue item.

## 2026-07-03 — Regeneration step 4: full analysis re-run complete (frozen snapshot)

All analyses ran cleanly (exit 0) on the one frozen 100,113-embedding
snapshot with byte-identical input across the three models. Headline
numbers, old paper value in parentheses:

- Predicates analyzed: 268 per model (159)
- Strong operations, alignment > 0.7: mxbai 54 (32), nomic 80, minilm 68
- Universal operations found by all 3 models: **33** (30); 16 by two, 19 by one
- Consistency↔MRR correlation: mxbai **0.779** (0.861), nomic 0.430, minilm 0.804
- Cross-model alignment correlations: mxbai↔minilm r = 0.901 (0.779),
  mxbai↔nomic 0.436 (0.554), nomic↔minilm 0.623 (0.358)
- Collision scan (mxbai, Ollama v0.17.1, threshold 0.95):
  **969,622 cross-entity colliding pairs** (147,687), 16,684 same-entity;
  mean k-NN cosine distance 0.2296

Qualitative read: every published finding replicates directionally —
a universal-operation core exists (and grew), consistency predicts
accuracy in all three models, and the diacritic collapse reproduces at
~6.6x the old pair count because the deeper crawl reaches far more
diacritic-rich entities. The nomic correlation (0.43) is the weakest
link and worth a caveat sentence if the paper is updated. Results live
in data/fol_results.json, data-nomic/, data-minilm/,
data/cross_model_comparison.json, data/analysis_results.json (all
gitignored, regenerable).

Remaining queue: paper.md numbers update — BLOCKED-ON-USER-ACTION
(Emma's explicit go-ahead), plus per-entity collision participation
count and collapse-geometry stats to compute at update time.

## 2026-07-03 — Regeneration step 3: frozen re-embeds complete

`reembed_frozen.py` embedded the identical 100,113-text frozen set with
nomic-embed-text (768-dim) and all-minilm (384-dim); outputs renamed to
the established `data-nomic/` and `data-minilm/` convention that
`compare_models.py` expects. All three stores verified by shape and
index row count: (100113, 1024) / (100113, 768) / (100113, 384), index
100,113 in each. This is the first run in this project where all three
models see byte-identical input — the original cross-model runs
re-crawled per model, which is what produced the "identical input"
error the arXiv-hold report flagged. Analysis chain (fol_discovery x3,
compare_models, analyze_collisions) launched.

## 2026-07-03 — Regeneration step 2: P31 country seed folded in

`import_wikidata.py --instances Q6256 --limit 300` found 209 direct
country instances; 209 of the 217 requested QIDs were already fully
imported by the deep Engishiki crawl, so the second seed added only 8
new entities + 45 linked labels. Final frozen store: **37,893 items,
100,113 embeddings**, index row-count verified. The process was killed
externally during the Step 5 RDF rebuild (triples.nt/trajectories.ttl),
which the analysis pipeline does not read — core store saved and
consistent. One transient WDQS ReadTimeout on the first attempt;
straight retry succeeded (query itself runs in 0.4 s). Takeaway for the
paper: at --limit 1000 the Engishiki BFS subsumes the country-level
seed; the two-seed design is now effectively one deep crawl.

## 2026-07-03 — Regeneration step 1: fresh Engishiki crawl complete

`random_walk.py Q1342448` finished cleanly (exit 0): **37,840 items,
100,006 mxbai-embed-large embeddings** in `data/` (gitignored). Notes:
the run resumed from a checkpoint after being externally killed at 215
entities, and the resume path picked up the script's default `--limit
1000` rather than the original 500 — kept deliberately (denser map;
Emma said to barrel through). The fresh Engishiki-only store already
exceeds the original paper's combined two-seed store (90,827
embeddings), consistent with Wikidata growth since the original crawl.
`scripts/reembed_frozen.py` was smoke-tested mid-crawl (16,645 texts,
all-minilm, row-count assertion passed). Next: P31 country seed
(`import_wikidata.py --instances Q6256 --limit 300`, now running), then
frozen re-embeds with nomic-embed-text and all-minilm, then the
analysis re-runs. The paper.md numbers update stays blocked on Emma's
go-ahead.

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
