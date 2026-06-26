## Source: Claude (round 2), captured ~2026-06-26

_Chat title: "Final review before arxiv submission". Model: Sonnet 4.6 (Low). This is a second review round — round 1 reviewed an earlier draft, round 2 reviews the revised draft. Both rounds are preserved below as captured._

---

### You said:
I want to do a last review of this thing before posting on arxiv
[pdf attachment]
I want to do a last review of this thing before posting on arxiv

### Claude responded: This is solid work — the bisection methodology is particularly compelling as evidence.

This is solid work — the bisection methodology is particularly compelling as evidence. Here are the things worth reviewing before you post:

**Strongest parts (don't touch)**

The version bisection is the paper's crown jewel. The controlled pairs table (Table 10) is immediately legible and damning. The string overlap null model is a clean, preemptive rebuttal.

**Issues worth fixing**

_Structural/framing_

The abstract front-loads the cartography framing heavily, but the actual novelty that will get this paper cited is the Ollama regression. Consider reordering the abstract so the bisection finding leads, with cartography as the method that discovered it. Right now someone skimming abstracts might mentally file this as "another embedding analysis paper" and miss it.

Section 5.4 header appears duplicated in 5.4.1: "5.4.1 5.4.1 Provenance..." — fix the numbering.

_Claims that need tightening_

"The impact is likely substantial" (p. 13) — this is the one place where you're making a claim you can't support with your data. Either cut it or hedge more explicitly ("we cannot estimate the scale of affected deployments, but...").

The v0.14.0 changelog observation about the NaN/−Inf embedding-path change is suggestive but you flag it without following through. Either explicitly say "this is a candidate mechanism we cannot confirm from source" or cut it — as written it reads like you're gesturing at a cause you haven't verified.

_Minor_

Table 6 footnote explaining that all two-hop examples involve Tadahira is good — but the note at the end ("The composition mechanism itself is general") feels slightly defensive. You could just state the dataset composition fact neutrally and let readers draw their own conclusions.

The phrase "The method, all code, and all data are publicly available" appears in the abstract but the paper never gives the actual URL/repo link anywhere except the author header. Make sure the repo link is explicitly cited in Section 6 or a dedicated availability statement.

"Jinmyochō" appears in Section 5.4 before it's formally introduced — the first mention references the "Jinmyochō collapse" as if the reader already knows what it is. One sentence of context would help.

**One thing to consider**

The limitations section (5.6, point 5) is appropriately honest that you haven't identified the exact commit. But you might consider adding a sentence about whether you've filed an issue with Ollama maintainers or plan to — this is the kind of responsible disclosure note that reviewers and readers will look for, and its absence might read as an oversight even if you're posting to arXiv preprint.

Overall the paper is in good shape. The empirical core is airtight.

---

### You said:
Here is my updated version of the paper.
[pdf attachment]
Here is my updated version of the paper. It still has a couple of formatting issues but I think I improved it a decent amount and just need to fix some of the formatting issues with the figures being weirdly added and some minor things. Conversion timeline taken while ignore the fact that the titles are relatively long and the lack of a gloss for the engishiki and the figures being weirdly compressed and focused on other issues. Is there anything else or do you think it's ready for posting?

### Claude responded: The revision addresses everything substantive from my previous feedback well.

The revision addresses everything substantive from my previous feedback well. The abstract reordering is a real improvement — the regression leads now and the cartography framing follows naturally as "how we found it." The GitHub issue link and the data availability section both land exactly where they should. The duplicate section numbering is fixed.

The remaining issues you've already identified (figure rendering, long title wrapping, Engishiki gloss) are real but minor and purely cosmetic. One small thing you may not have caught: the "likely substantial" line is still in there (p. 16, in the "defect is silent and likely exploitable" paragraph). Given that you can't actually quantify the affected deployment population, that phrase is still the one place where you're slightly overreaching. You could swap it for something like "the prevalence of diacritical marks in non-English text suggests the impact is non-trivial, though we cannot estimate the scale of affected deployments" — or just cut the sentence entirely since the paragraph makes the point without it.

Otherwise I think it's ready. The figures compress fine on arXiv's renderer even when they look odd in a PDF preview, and the title length is not unusual for the venue. Post it.
