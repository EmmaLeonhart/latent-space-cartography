# Pre-arXiv external AI reviews — round 2 (2026-06-26)

A second pass from the same six assistants (Claude, Gemini, DeepSeek, Grok,
Meta AI, Mistral) after the round-1 revisions. Verdict across all six: **ready /
post it** — no model raised a scientific or claim-level objection.

Important: most reviewers were looking at the **pre-fix PDF** (Meta AI explicitly
said it never received the updated file). So their headline items are things
already fixed, or artifacts of their PDF paste — not new work.

## Already fixed before this round (verified live)

- **Distorted/compressed figures** (all six flagged) — fixed: the build stripped
  pandoc's `height=\textheight` cap so figures keep their aspect ratio. Live PDF
  figure ratios now match the sources exactly.
- **Long title** (Mistral, Grok) — retitled to lead with the regression.
- **"v0.13.5 → v0.14.0"** (Mistral, Gemini) — unified to v0.13.4 → v0.14.0 (0 left).
- **"likely substantial"** (Claude) — softened to a non-quantitative scoping (0 left).
- **Availability statement + Ollama disclosure link** (most) — §6.1 + `ollama#15609`.
- **Abstract reframe**, **5.4.1 double-numbering**, **Engishiki inline gloss** — done.

## Phantom — claimed but NOT in the source (PDF-extraction artifacts, no change)

- Duplicate Table 10; Table 3 "broken"; Table 7 heading after the data;
  "demographic31"/"demographics21"/"participating32"; "igure 1:"; `<fcel>` token;
  abstract truncating at "Filast"; "t opazc.comput.ing.com" URL spacing;
  "Englishki" spelling; dropped umlaut in "Rocktäschel". All verified clean /
  compile correctly; these are how the reviewers' copy/paste mangled the rendered
  PDF, not bugs in the LaTeX source.

## Genuinely optional ideas (none blocking — Emma's call)

- `pdffonts paper.pdf` check for Type 3 fonts (arXiv hygiene; `lmodern` is already
  installed in CI).
- Add the commit hash next to the repo URL in §6.1.
- A short "Note to Practitioners" / Key-Contributions box after the abstract.
- Move the availability section directly under the abstract.
- Richer figure captions / a one-line Docker quickstart in the README.

Do not add Mistral's invented references (`Nostic et al. 2023`) or invented
paths (`emmaleonhart/latent-space-cartography:latest`, `scripts/bisection.py`).
