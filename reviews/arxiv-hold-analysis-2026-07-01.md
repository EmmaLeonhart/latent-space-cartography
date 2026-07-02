# arXiv-hold analysis — Latent Space Cartography (2026-07-01)

Review of `paper.md` (current local version, corrected title) done while vendoring
this repo as a submodule of `research_library`. Goal: identify what might be behind
the arXiv hold and any errors worth fixing before submitting a new version.

## TL;DR

- The **most likely cause of the hold is not a content error** — it's the
  AI-authored / AI-agent-venue signal (Claw4S / clawRxiv openly state authors and
  reviewers are AI agents). arXiv moderation holds/flags suspected LLM-generated
  submissions and non-standard venues; that is the probable trigger, and no paper
  edit fixes it. See §1.
- The paper's **big scientific risk is already fixed**: the title + abstract now
  attribute the defect to the **Ollama runtime (v0.14.0 regression)**, not to
  mxbai-embed-large's weights. That was the dangerous framing (naming a model as
  "defective" for a third-party runtime bug). Good.
- Two **concrete numeric inconsistencies** remain and a reviewer will catch them
  (§2). Fix before resubmission.
- A few **tone/hygiene fixes** (§3): "likely exploitable" overclaims; the GitHub
  repo description + README still carry the old *model-defect* framing; a leftover
  CI HTML comment sits in the references.

## 1. Why it's probably on hold (and what you can/can't do)

- **AI authorship / venue (most likely, not fixable by editing the paper).**
  Claw4S is described as a venue where "both authors and reviewers are AI agents,"
  and the preprint lives on clawRxiv. arXiv has been actively holding/removing
  submissions it flags as LLM-generated or from non-standard conferences. The
  paper's own "AI Disclosure" is reasonable, but the venue signal is the thing
  arXiv moderators react to. **Action:** if you want it on real arXiv, it likely
  needs (a) a recognized category + endorsement, (b) framing as an independent
  preprint rather than a Claw4S proceedings paper, and (c) possibly de-emphasizing
  the AI-agent-conference provenance. This is a venue/endorsement problem, not a
  content bug — don't burn effort "fixing" the science to clear it.
- **Security-disclosure tone (possible secondary trigger).** The paper names a
  specific OSS product (Ollama), a specific version (v0.14.0), links a GitHub
  issue, and calls the defect "silent and **likely exploitable**." A moderator
  scanning for vulnerability-disclosure content could hold on that. Softening the
  "exploitable" claim (§3) reduces this surface.

## 2. Concrete numeric inconsistencies (fix these)

**2a. "Identical input" vs. different embedding counts.**
§4.6 (line 279): "All three models were given **identical input**: the same
Wikidata entities seeded from Engishiki (Q1342448) with --limit 500."
But Table 8 (lines 283–285) reports different embedding counts per model:

| model | embeddings |
|---|---|
| mxbai-embed-large | 41,725 |
| nomic-embed-text | 69,111 |
| all-minilm | 54,375 |

If the input entity+alias set is identical, the number of embeddings must be
identical across models (embedding count is a property of the input, not the
model). Three different counts contradict "identical input." Almost certainly the
three runs used **different Wikidata snapshots** (BFS re-run at different times, so
the graph had grown). **Fix:** either re-run all three on one frozen entity set so
the counts match, or state plainly that the runs used different snapshots and give
each run's date — and drop the word "identical."

**2b. Collision totals don't add up.**
§5.4 (line 323): "16,067 entities (**of 41,725**) participating in at least one
collision." That leaves 41,725 − 16,067 = **25,658** non-colliding.
But §5.4.1 (line 367): "16,067 colliding embeddings (vs. **74,760** non-colliding)."
16,067 + 74,760 = **90,827 ≠ 41,725**. The "74,760 non-colliding" figure is
inconsistent with the "of 41,725" base (it looks like the geometry analysis ran on
a larger/multi-seed set of ~90,827). **Fix:** reconcile to one denominator — either
state the geometry analysis used a different, larger set (and name its size), or
correct 74,760 → 25,658.

## 3. Tone / hygiene

- **"likely exploitable" (line 379)** overclaims. There's no threat model or
  attacker-controlled path shown — it's a silent *degradation*, not an exploit.
  A reviewer will call this unsupported, and it reads as security-disclosure
  language to a moderator. Suggest: "silent and consequential" / "silently
  corrupts downstream retrieval."
- **GitHub repo description is stale and re-introduces the wrong framing.** It
  still reads "…frozen mxbai-embed-large-v1 model weights with documented [UNK]
  tokenizer defect" — i.e. blames the model, the exact thing the paper corrected.
  The README's linked clawRxiv title is also the **old** one ("…Silent Tokenizer
  **Defect in mxbai-embed-large**"). Update both to the runtime-regression framing
  so the public metadata matches the paper.
- **Leftover CI marker in the references (line 463):** an HTML comment
  `<!-- ci: pipeline submission check 2026-05-19 (invisible marker, no content change) -->`
  is embedded after the Wang et al. (2014) entry. Harmless in markdown, but strip
  it from any submitted source.

## 4. What's already solid (don't touch)

- Attribution is correct and repeatedly guarded ("not the model," byte-identical
  blob, clean on ≤ v0.13.4). The version bisection (Table 11) is a clean controlled
  experiment and the strongest part of the paper.
- The r = 0.861 consistency↔accuracy correlation is **honestly hedged** — the paper
  itself notes it's "partly a geometric property… not purely empirical," which
  defuses the obvious "tautology" reviewer objection.
- Limitations §5.6 are candid (training-data overlap, single-seed geometry,
  label-only embeddings, mechanism localized empirically not from source). Good.
- The string-overlap null model (§4.4) with its own stated limitation pre-empts the
  "it's just string matching" objection.

## 5. Recommended order of operations

1. Decide the venue question first (§1) — that governs whether an arXiv resubmission
   can even clear moderation. If arXiv is the goal, sort category/endorsement and
   de-emphasize the AI-agent-conference provenance.
2. Fix the two numeric inconsistencies (§2) — cheap, and they're the only things a
   competent reviewer will flag as *errors* rather than choices.
3. Soften "exploitable" and fix the GitHub description + README title (§3).
4. Strip the CI HTML comment.
