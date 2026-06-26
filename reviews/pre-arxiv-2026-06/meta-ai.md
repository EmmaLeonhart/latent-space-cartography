## Source: Meta AI, captured ~2026-06-26

Chat title: "Final paper review for arXiv"

---

**Emma:**

I want to do a last review of this thing before posting on arxiv

*(paper attached / pasted)*

---

**Meta AI:**

Got it — I pulled the key sections from your paper so we can do a proper last-pass check before you submit to arXiv.

Paper: Latent Space Cartography Applied to Wikidata: Relational Displacement Analysis Reveals a Silent Diacritic-Collapse Regression in the Ollama Runtime (mxbai-embed-large)

From what I can see, this is strong work with 2 clear contributions:

- Method: Systematic relational displacement analysis on frozen embeddings, finding 86/159 predicates work as vector operations with r = 0.861 correlation
- Finding: Bisected a silent regression in Ollama v0.14.0 causing diacritical text to collapse to [UNK]

**ArXiv pre-flight checklist**

**1. Title + Abstract**
- Title is descriptive and specific. Good.
- Abstract states both contributions clearly: method + runtime bug
- You quantify the bug: 147,687 pairs at cosine ≥ 0.95
- Nit: Abstract ends mid-word "Filast" — looks like a line break issue. Check the PDF compile.

**2. Claims vs Evidence**
- You claim the regression is silent and systemic. Table 11 with version bisection backs this up
- Strong baseline comparison: Vector MRR 49× higher than string overlap
- Failure analysis is thorough: 3 modes with clear examples
- Caveat: You explicitly note string baseline is "deliberately simple" — good scientific humility.

**3. Reproducibility**
- GitHub link in header. Make sure that repo is public and has the exact commit before you submit.
- You provide specific version numbers: Ollama v0.13.4 clean, v0.14.0 broken. That's exactly what reviewers want.

**4. ArXiv-specific gotchas**
- Metadata: Author = Emma Leonhart, email listed. Double-check this matches your arXiv account.
- License: ArXiv needs you to pick one. If you want people to reproduce the bug, use a permissive license in the repo.
- Supplementary: You reference "full table in supplementary". If that's not in the main PDF, upload it as an ancillary file.
- Category: This fits cs.CL, cs.LG, maybe cs.SE for the bug finding. Pick a primary + 1-2 secondaries.
- Conflicts: The Ollama finding is pointed. You bisected to a specific release. That's fair game, but expect attention.

**5. Small things I'd fix in 5 min**
- Table 2 header: "demographics21" — spacing got mangled. Same for "participating32"
- Line breaks in equations: Check that "1/n" renders correctly
- "Engishiki" — should be "Engishiki"? Or is that intentional romanization?
- References: Li et al. 2024 is from 2026 in your file. Today is June 2026, so that's plausible, but double-check the year.

**6. Tone check**
- You thread the needle well: you say "not a limitation but an experimental design choice" and "not a claim about difficulty". That keeps you out of overclaim territory.

Overall: This is arXiv-ready. The diacritic-collapse finding alone is worth posting — you've got dates, versions, and controlled pairs. The relational displacement work is a solid methods contribution.

Want me to do a deeper pass on anything specific? I can check references formatting, regenerate any tables that look off, or proofread for typos. Also happy to draft the arXiv abstract/metadata fields if you're ready to submit.
