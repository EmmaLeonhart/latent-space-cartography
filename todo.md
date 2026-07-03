# todo — long-term horizon

Abstract, multi-session goals. Items here are destinations, not steps; when work
begins on one, it gets decomposed into concrete items in `queue.md` (see
CLAUDE.md "Queue and longer-horizon work" for the flow).

## Turn the paper from a Claw4S entry into an arXiv paper

The big one, from Emma's 2026-07-02 framing: the competition is over; the goal
is a paper that can clear arXiv moderation and stand as an independent preprint.
Sub-horizons, roughly in dependency order:

- **Venue/endorsement decision (Emma).** arXiv category + endorsement, framing
  as an independent preprint rather than Claw4S proceedings, how much to
  de-emphasize the AI-agent-conference provenance. Analysis in
  `reviews/arxiv-hold-analysis-2026-07-01.md` §1. Everything below is shaped by
  this.
- **Adopt the regenerated frozen-snapshot numbers.** The 2026-07-03 re-run
  (devlog) replicates every finding on byte-identical cross-model input:
  100,113 embeddings, 268 predicates, 33 universal ops, r = 0.779/0.804/0.430,
  969,622 colliding pairs on Ollama v0.17.1. Blocked on Emma's explicit
  go-ahead; at update time also compute per-entity collision participation and
  collapse-geometry stats (analogs of the old 16,067 figure and §5.4.1
  density numbers), and port `old/scripts/measure_collapse_geometry.py` into
  `scripts/`.
- **Kill the tautology criticism properly.** Train/test split for the
  consistency↔accuracy correlation per `planning/tautology-fix.md` — the r=0.861
  hedge did not stop it resurfacing (see review_20260702_235309). The
  regenerated data is the natural base for this.
- **Report actual cosine values instead of rounded 1.000.** Reviewers read
  "exactly 1.000" as fabricated; state the measured floats (e.g. 0.9997) in
  Table 10 and the abstract examples.
- **Explain or caveat the nomic outlier.** nomic-embed-text shows the weakest
  consistency↔accuracy correlation (0.43 vs 0.78/0.80) on the frozen snapshot;
  needs either a mechanism investigation or a plain one-sentence caveat.
- **Single-crawl simplification of §3.2.** The deep 1000-entity Engishiki BFS
  subsumes the P31 country seed (209/217 countries already reached); the
  two-seed story can be retired when numbers are refreshed.

## Keep the review-feedback loop useful

- The clawRxiv auto-submit pipeline exists purely for review signal now.
  Reviewer knowledge-cutoff artifacts (e.g. the 2026-dates "hallucination"
  Strong Reject) are noise; decide whether to keep auto-submitting every push
  or gate it.

## Productivity/infrastructure

- Keep this repo at cleanvibe standards: `queue.md` (concrete, delete-on-done) /
  `todo.md` (this file) / `devlog.md` (dated history), autonomous-loop crons
  for extensive sessions, forward flow only.
