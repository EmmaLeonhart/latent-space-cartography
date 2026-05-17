# Latent Space Cartography → Hyperdimensional Computing Paper

## Three Pillars (the core contributions)

1. **HDC framework for logical inference on frozen embeddings** — Chains of logical inference implemented as vector operations on existing general-purpose embeddings. Massively cheaper than full model inference. The displacement vectors are binding/unbinding operators, and composition (multi-hop) is sequential binding.

2. **Emergent VSA in frozen embeddings** — General-purpose text embeddings (mxbai-embed-large, nomic-embed-text, all-minilm) spontaneously implement Vector Symbolic Architecture binding/unbinding operations without being trained for them. 30 universal relations manifest across all three models. This is an emergent property, not a trained one.

3. **Critical production defect in mxbai-embed-large** — The [UNK] token dominance defect causes 147,687 cross-entity embedding collisions. Strings with diacritical marks collapse into identical vectors (cosine 1.0 between unrelated entities like "Hokkaidō" and "Éire"). Makes the model nearly useless for any domain with non-ASCII text. Missed by standard benchmarks like MTEB.

## Paper Strategy
- **Quality over quantity** — This is the one paper we're focusing on.
- **clawRxiv competition is OVER** — extended deadline (~April 20, 2026) has
  passed. The $5,000 prize / Claw4S 2026 framing is no longer the goal.
- **Why the pipeline is still on:** auto-submit is re-enabled purely to keep
  getting AI peer-review *feedback* on each iteration. Superseding the
  Strong-Accept post is fine now — there is no competition left to protect it
  for; the value is the review signal, not the rating.
- **Current status:** Post 1127, paper_id 2604.01127, Strong Accept (final
  competition result). Earlier "Post 859 / v15" notes were stale.
- **Repo:** EmmaLeonhart/latent-space-cartography

## Key Criticisms to Address
- **"It's just TransE"** (10/15 reviews) — Fix by reframing as VSA/HDC, which it actually is
- **"Tautology" in consistency-accuracy correlation** (9/15 reviews) — Fix with proper train/test split
- **"Grandiose framing"** — Don't overclaim. Let the empirical results speak.
- See `planning/` directory for detailed analysis

## Workflow
- Push changes to paper.md or SKILL.md → CI auto-submits to clawRxiv → fetches AI review → commits review to reviews/
- Auto-submit (`.github/workflows/publish.yml`) is enabled **for the peer-review feedback loop only** — the competition is concluded (see Paper Strategy).
- Planning docs in `planning/`
- All 15 historical reviews in `reviews/`

## Queue and longer-horizon work

(Clarity model adopted from the `cleanvibe` scaffold — the bar for "clear project docs." This repo had no documented `queue.md`/`todo.md` convention; this is it.)

- **`queue.md`** — what is being worked on *right now*: concrete, executable steps. Deleted in the same commit that completes them — no checkmarks, no "done" markers, no status narration. If a line is still there, it is not done. Not in `queue.md` = not in scope this session.
- **`todo.md`** — the long-term horizon: abstract, multi-session goals (a destination, not a step). The *basis for* `queue.md`; parked / deferred / reference material lives here, never in `queue.md`.
- **Forward flow only:** `todo.md` → `queue.md` → task tool → `git log`. Items only move forward; done work is deleted, not annotated. Create these files when work begins (cleanvibe-bootstrap style); a stale `queue.md` is worse than none.

## Technical Notes
- Scripts in `scripts/`, main analysis is `fol_discovery.py`
- Data regenerable from Wikidata + Ollama (mxbai-embed-large, 1024-dim)
- Frozen model weights in `model/` (Git LFS)
- Use `python` not `python3` on this system
