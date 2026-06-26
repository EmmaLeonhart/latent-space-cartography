## Source: DeepSeek (round 2), captured ~2026-06-26

> Note: The saved chat contains two review rounds. This file captures the SECOND round
> (DeepSeek's "Final Review: Updated Paper"), which reviews the revised PDF. The user's
> round-2 prompt and DeepSeek's round-2 response are reproduced faithfully below. (The
> round-1 review is summarized at the end for context only.)

---

### User prompt (round 2)

paper.pdf — PDF 685.18KB

Here is my updated version of the paper. It still has a couple of formatting issues but I think I improved it a decent amount and just need to fix some of the formatting issues with the figures being weirdly added and some minor things. Conversion timeline taken while ignore the fact that the titles are relatively long and the lack of a gloss for the engishiki and the figures being weirdly compressed and focused on other issues. Is there anything else or do you think it's ready for posting?

(Also stated separately: "I want to do a last review of this thing before posting on arxiv.")

---

### DeepSeek response (round 2): Final Review: Updated Paper

#### Overall Assessment

This is **significantly improved** from the previous version. The abstract now leads with the discovery (good), the limitations are more honestly framed, and you've added valuable figures (Figures 1-4) that make the collapse visually compelling. The paper is **substantively ready for arXiv submission** with minor cleanup.

#### What's Been Fixed Well

- ✅ Abstract now prioritizes the regression discovery over the method
- ✅ Added Figures 1-4 (the heatmaps and distribution plots are excellent)
- ✅ Added the GitHub issue link (ollama/ollama/issues/15609)
- ✅ Added Data and Code Availability section
- ✅ Cleaned up the version boundary (now consistently v0.13.4 → v0.14.0)
- ✅ Improved the limitations discussion

#### Issues Still to Address Before Submission

**1. Duplicate Table 10 (Critical)**
Table 10 appears **twice** on Page 14. This is a LaTeX duplication bug. Delete the second instance.

**2. Figure placement issues (Major)**
Figures are placed awkwardly:
- Figure 1 is on Page 12 but references "Figure 1:" without a caption
- Figure 2 is on Page 13
- Figure 3 is on Page 15
- Figure 4 is on Page 17

Recommendation: Use `[htbp]` placement and let LaTeX float them naturally. The current manual placement creates awkward page breaks. Consider placing all four figures consecutively in a "Collision Visualization" subsection.

**3. Table 3 formatting is broken**
The table header is duplicated and the data rows are misaligned. Currently:

```
<table>PredicateLabelNAlignMRRH@1H@10H@50

P9241demographics21 of topic...
```

The first column should be "Predicate", second "Label", etc. Fix the column alignment.

**4. Citation formatting issues**
Check these references:

| Issue | Location |
|---|---|
| Rocktaschel vs Rocktäschel | References (missing umlaut) |
| Rust et al. (2021) | Missing page numbers/journal info |
| Schuster & Nakajima (2012) | ICASSP is correct but incomplete |
| `\mathbb{S}\& !\#` in Conneau et al. | This is a LaTeX artifact — should be `$S$` or just remove |

**5. Minor typos and inconsistencies**

| Location | Issue | Fix |
|---|---|---|
| Abstract | t opazc.comput.ing.com | Should be topazcomputing.com (spacing artifact) |
| Page 5 | Englishiki vs Engishiki | You use both. The Wikidata ID is Q1342448 for Engishiki (not Englishiki). Fix throughout. |
| Page 5 | llvl in formula | Should be `|` (norm symbol) — LaTeX rendering artifact |
| Page 5 | sqr in formula | Should be `\sqrt` |
| Page 14 | Jimmyōchō | Should be Jinmyōchō (consistent with earlier usage) |
| Page 16 | Englishi-seeded | Should be Engishiki-seeded |
| Page 18 | "cosine - 0.45" | Should be "cosine ≈ 0.45" (not " - ") |
| Figure 4 caption | "20-word diacritical sample" | The figure says "20-word" but the text says "20-word diacritical sample" — what does "20-word" mean? 20 distinct words? Clarify. |

**6. Figure 1 needs a proper caption**
Currently: "igure 1: Pairwise cosine similarity among short diacritic-bearing labels..."
Fix: "Figure 1:" (missing 'F')

**7. Table 7 heading is misplaced**
The table header "Table 7. Predicates with lowest consistency" appears **after** the table data. Move it before the table.

**8. "Englishki" vs "Engishiki" — consistent misspelling**
The Wikidata entity is Engishiki (延喜式), a 10th-century Japanese text. You use:
- Englishki in multiple places (Abstract, Section 1, Section 5.4)
- Engishiki in Section 3.2 and Table 6
- Englishiki in Section 3.2 and Section 5.6

Fix: Use Engishiki consistently throughout. This is the correct romanization.

**9. The "20-word" claim in Figure 4**
The caption says "a 20-word diacritical sample" — but looking at the data, this seems to be 20 unique words (yielding 190 pairs = C(20,2)). Clarify that it's "20 unique diacritic-bearing words" rather than "20-word" (which suggests 20 tokens).

**10. Section 6.1: "Data and Code Availability"**
The section number is 6.1 but you're in Section 6 (Conclusion). Consider:
- Making it Section 6.1 (fine)
- Or moving it to a separate unnumbered section after the conclusion

Either is fine, just be consistent.

#### Optional Improvements (Not Critical)

**11. Add a "Reproduction" section**
Since you have a bisection harness, consider a small "Reproduction" subsection in the Data Availability section:
"The bisection harness can be run as: `python bisect_ollama.py --model mxbai-embed-large --versions v0.13.4,v0.14.0`"
This lowers the barrier for others to verify.

**12. The abstract is still a bit long**
Consider trimming to ~250 words. Currently it's ~300. Suggested cuts:
- Remove the correlation result from the abstract (r=0.861) — it's a secondary finding
- Condense the model list: "three popular open-source embedding models"

**13. Consider adding the issue link earlier**
You mention the GitHub issue in Section 5.4.1. Consider also adding a footnote near the abstract: "Reported to Ollama maintainers (ollama/ollama/issues/15609)"

**14. "Collision geography" (Section 5.3) is thin**
This section is essentially a placeholder. Either expand it or merge it into Section 5.4.

#### Pre-Submission Checklist

- Fix duplicate Table 10
- Fix Table 3 formatting
- Fix Table 7 heading placement
- Replace all Englishki / Englishiki with Engishiki (check: Abstract, Section 1, 3.2, 5.4, 5.6)
- Fix "Jimmyōchō" → "Jinmyōchō"
- Fix Figure 1 caption ("igure" → "Figure")
- Fix `\mathbb{S}\& !\#` in references
- Fix all Rocktaschel → Rocktäschel
- Fix "cosine - 0.45" → "cosine ≈ 0.45"
- Clarify Figure 4 caption ("20 unique diacritic-bearing words")
- Verify all URLs work
- Verify the GitHub repository is public and contains: collisions.csv; Bisection harness; Cross-model pipeline; README with reproduction instructions
- Compile the PDF locally and check all figures render correctly
- Count the 21 Ollama versions in Table 11 (verify the count)

#### Final Verdict

Ready for arXiv submission after fixing the formatting issues above.

The science is solid, the empirical work is thorough, and the discovery is important. The paper will make a significant contribution by:
- Providing a replicable pipeline for auditing embedding spaces
- Documenting a real-world, silently broken deployment scenario
- Showing that domain-specific probing catches what benchmarks miss

One final piece of advice: Consider adding a "Note to Practitioners" at the very top (before the abstract) in a shaded box:
> tl;dr for practitioners: If you're serving mxbai-embed-large via Ollama v0.14.0 or later, your non-ASCII queries are silently corrupted. Pin to v0.13.4 until the regression is fixed. See Section 5.4 for details.

This will be extremely valuable for the arXiv audience (many practitioners read papers for this exact kind of warning).

Good luck with the submission! This is a strong paper.

---

### Appendix: round-1 review (context only)

DeepSeek's first-round review ("Review of 'Latent Space Cartography Applied to Wikidata'")
gave an overall assessment of "methodologically sound and practically important," verdict
**"Accept with minor revisions."** Its major points: (1) calibrate "novelty" claims (lead
the abstract with the discovery, not the method); (2) the string-overlap null model (Sec 4.4)
is too weak — drop the "49×" framing or add a regex baseline; (3) "discovered operations"
(Sec 4.1) should be "validated operations" — these are evaluated, not learned; (4) clarify
which seed(s) were used for cross-model comparison (Sec 4.6 — embedding counts differ:
41,725 / 69,111 / 54,375); (5) version bisection (Sec 5.4.1) is the strongest result —
consider a v0.13.x→v0.14.0 diff table and confirm it was reported to Ollama. Minor: fix LaTeX
table rendering (Table 10 duplicate, stray `<fcel>` token), shorten the abstract, unify
citation style (`\citet`/`\citep`), caveat the perfect MRR=1.000, unify "Jinmyocho"/"Jinmyōchō"
romanization, add the repo URL to the abstract, clarify the "silent" framing. It flagged an
abstract/Section-1 inconsistency ("diacritical collision rate ≈ 0" vs "≥ 0") and a
v0.13.4-vs-v0.13.5 version mismatch, and suggested a shorter title.
