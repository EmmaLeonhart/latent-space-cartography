## Source: DeepSeek — "Review of latent space cartography paper", captured ~2026-06-26

---

**Emma (prompt, attached `paper.pdf`, 212.79 KB):**

> I want to do a last review of this thing before posting on arxiv

---

**DeepSeek:**

# Review of "Latent Space Cartography Applied to Wikidata"

## Overall Assessment

This is a methodologically sound and practically important paper that makes two distinct contributions: (1) a systematic application of relational displacement analysis to frozen embedding models, and (2) discovery of a serious serving-stack regression. The paper is well-structured and the empirical work appears thorough. Below I note issues to address before arXiv submission.

## Major Issues

**1. Claims about "novelty" need careful calibration**

The abstract says the method is "not methodologically novel" but the introduction claims "systematic relational mapping across all predicates... remains underexplored." This is fine, but be precise: the contribution is the systematic pipeline + the discovery, not the method itself. The current framing could read as overselling.

Recommendation: In the abstract, lead with the discovery, not the method. Something like: "We apply standard relational displacement analysis as a systematic auditing tool and discover..." This accurately positions the work.

**2. The string overlap null model (Section 4.4) is too weak**

You acknowledge this in the limitations paragraph, but the framing in Table 7 ("vector MRR is 49× higher") could be misleading. You correctly note that a regex baseline would close the gap for formulaic predicates.

Consider either:
- Removing the "49×" framing entirely, or
- Adding a stronger baseline (e.g., regex patterns for the most formulaic predicates like "Demographics of [X]") and reporting those results.

The current version feels like you're setting up a straw man to knock down, even though you qualify it.

**3. The "discovered operations" framing in Section 4.1**

Table 2 lists "discovered operations" with alignment scores. But these are not discovered in the sense of being learned — they're evaluated. The term is used throughout, but it conflates "we found these relations work" with "we discovered new relations."

Recommendation: Use "validated operations" or "consistently encoded relations" throughout. Save "discovered" for the regression finding.

**4. Cross-model generalization (Section 4.6)**

You report 30 universal operations across all three models. But Table 8 shows vastly different embedding counts (41,725 vs 69,111 vs 54,375) — were these all seeded with the same BFS parameters? If not, the overlap comparison is not apples-to-apples.

Check: The method section says "both seeds contribute to the relational displacement analysis" and "the collision analysis focuses on the Englishiki seed." Clarify exactly which seed(s) were used for cross-model comparison.

**5. The version bisection (Section 5.4.1)**

This is the strongest empirical result. However, you state "identifying the precise upstream commit within that release is left to Ollama maintainers." This is fine, but consider adding a small table of the v0.13.5 → v0.14.0 diff (or at least summarizing what changed in the embedding path) if you can infer it from release notes or source. If not, at least note that you've reported this to the Ollama team (have you? If so, mention it).

## Minor Issues

**6. Formatting errors**

The paper has several table rendering issues in the PDF:
- Table 10 appears twice (duplicate heading)
- The `<fcel>` token appears in Table 10 ("Hokkaido" `<fcel>` 0.500) — this looks like a LaTeX rendering artifact
- Tables 1, 5, 7, 9 have inconsistent formatting

Fix: Clean up the LaTeX table code. Use `\begin{tabular}` with proper column specifications and avoid markdown-style table syntax in the raw text.

**7. Overly long abstract**

The abstract is dense and tries to cover too much. Consider:
- Moving the technical details (147,687 pairs, 95% CI, etc.) to the main text
- Keeping the abstract focused on: (1) what you did, (2) what you found, (3) why it matters

**8. Citation style inconsistencies**

Some citations are in parentheses (Bordes et al., 2013) and some are textual TransE (Bordes et al., 2013) — pick one style. `\citet` vs `\citep` in LaTeX would fix this.

**9. Section 4.2: Perfect MRR = 1.000**

You explain this well (functional predicates + naming conventions), but add a caveat: perfect MRR means the model perfectly predicts which entity label should be the object, not that it understands the semantic relation in a deep sense. The string overlap null model partially addresses this, but make it explicit.

**10. Section 5.4: "Jinmyocho" vs "Jinmyōchō"**

Inconsistent romanization throughout. Pick one (preferably with diacritics, since that's the point) and stick with it.

**11. Missing code/data repository link in the abstract**

You mention "All code and data are publicly available" but the link appears only in the author affiliation (github.com/EmmaLeonhart/latent-space-cartography). Add the repository URL explicitly in the abstract or a dedicated "Data Availability" section.

**12. The "silent" framing**

You use "silent" repeatedly (silent regression, silent failure, silently fails). This is appropriate, but be careful: the failure is not silent in the sense of being undetectable — you detected it. It's silent in the sense that it raises no error and returns plausible-looking vectors. Clarify this once and then use the term consistently.

## Suggestions for Strengthening

**13. Add a "Practical Recommendations" section**

You mention practical recommendations in Section 5.5, but they're buried. Consider a dedicated subsection or a boxed summary:

For practitioners:
- If using mxbai-embed-large via Ollama, pin to v0.13.4 or earlier until the regression is fixed
- Test deployments with diacritic-rich input ("naïve", "Zürich", "Hokkaidō") as part of your CI
- Record runtime version alongside model version in embedding provenance

**14. The "broader lesson" (Section 5.5) could be expanded**

This is actually one of the most important points: generic benchmarks miss domain-specific failures. Consider adding a brief discussion of benchmark design — what would a better benchmark look like? You don't need to design one, but noting the gap is valuable.

**15. Limitations section needs more specificity**

- "Potential training data overlap" — you note this, but quantify it: what fraction of Wikidata entities have Wikipedia articles? Is there a way to test this (e.g., on entities with vs without Wikipedia coverage)?
- "Mechanism localized empirically" — this is honest, but consider adding a small experiment: does the same collapse occur when using the same model via sentence-transformers directly (without Ollama)? If not, that further confirms it's an Ollama-specific regression.

## Typos and Language

| Location | Issue | Fix |
|---|---|---|
| Abstract | "diacritical collision rate ≈ 0" | Should be "≈ 0" (\approx 0) not "≥ 0" (\geq 0) (Section 2 says ≈ 0 but Section 1 says ≥ 0) |
| Section 1 | "diacritical collision rate ≥ 0" | Should be "≈ 0" (typo in the PDF) |
| Section 4.4 | Table 7 formatting | Fix the table layout |
| Section 5.4.1 | "v0.13.5 \rightarrow v0.14.0" | The table says v0.13.4 → v0.14.0, but the text says v0.13.5 → v0.14.0. Unify. |
| Throughout | "Englishki" vs "Englishiki" | Check consistency — Section 5.4 uses "Englishki" but earlier sections use "Englishiki" (Q1342448) |

## Verification Checklist

Before submission, verify:
- All URLs work (especially the GitHub repo)
- The GitHub repo contains exactly the code/data described
- All tables render correctly in the arXiv PDF
- The Ollama version numbers are consistent throughout (v0.13.4 vs v0.13.5)
- The collision count (147,687) is clearly defined as pairwise collisions
- The 95% CI for the correlation (0.773, 0.926) is calculated correctly
- The "21 Ollama releases" claim matches the number of versions in Table 11 (count them)

## Final Recommendation

Accept with minor revisions.

This is a solid, reproducible piece of work with a genuinely important empirical finding. The methodological contribution (systematic relational probing) is valuable, and the regression discovery is actionable for practitioners.

The main things to address are:
- Clean up the tables (rendering issues)
- Calibrate the "novelty" claims
- Strengthen or contextualize the string overlap baseline
- Clarify the cross-model comparison setup
- Add the practical recommendations more prominently

## Optional: Title Suggestion

Current title is descriptive but verbose. Consider a shorter alternative:

"Relational Displacement Analysis Reveals a Runtime Regression in Ollama's mxbai-embed-large"

Or keep the current title but consider moving the method name to a subtitle:

"Latent Space Cartography Applied to Wikidata: A Silent Diacritic-Collapse Regression in the Ollama Runtime"
