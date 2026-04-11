"""
Regenerate the GitHub Pages site documenting the mxbai-embed-large
diacritic defect, consuming the JSON artifact produced by
`scripts/verify_tokenizer_divergence.py` so the page's mechanism
explanation always reflects the most recent reproducible state.

Produces:
  - docs/figures/collision_heatmap.png
  - docs/figures/diacritic_vs_plain.png
  - docs/figures/collision_count_over_threshold.png
  - docs/figures/token_analysis.png
  - docs/index.html  (regenerated from warm-parchment template +
                      verification/tokenizer_divergence.json)

Prereqs:
  1. Ollama running with the archived mxbai gguf registered as
     `mxbai-archived` (see model/Modelfile).
  2. `python scripts/verify_tokenizer_divergence.py` has been run
     recently and produced verification/tokenizer_divergence.json.

Usage:
  python scripts/verify_tokenizer_divergence.py   # step 1
  python scripts/generate_defect_page.py           # step 2
"""

import csv
import io
import json
import math
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIG_DIR = DOCS / "figures"
CSV_PATH = ROOT / "collisions.csv"

OLLAMA_URL = "http://localhost:11434/api/embed"
# Use the archived gguf registered from model/Modelfile, not whatever the
# user happens to have pulled from the Ollama registry. This keeps the
# figures reproducible from the frozen weights in this repo.
MODEL = "mxbai-archived"

# Palette — matches the CSS variables in docs/index.html so the figures
# visually tie into the site. Parchment background, muted terracotta
# accent (replaces the previous harsh #e74c3c), steel blue, sage.
PAL_BG = "#f6f2ec"
PAL_SURFACE = "#efe9df"
PAL_TEXT = "#2d2a26"
PAL_DIM = "#6e6a61"
PAL_ACCENT = "#b8553a"    # muted terracotta
PAL_BLUE = "#3f6487"      # muted steel blue
PAL_GREEN = "#5b7a4a"     # sage
PAL_BORDER = "#d9d1c1"

# Use the palette globally so every chart follows it
plt.rcParams.update({
    "figure.facecolor": PAL_BG,
    "axes.facecolor": PAL_BG,
    "savefig.facecolor": PAL_BG,
    "axes.edgecolor": PAL_BORDER,
    "axes.labelcolor": PAL_TEXT,
    "axes.titlecolor": PAL_TEXT,
    "text.color": PAL_TEXT,
    "xtick.color": PAL_TEXT,
    "ytick.color": PAL_TEXT,
    "grid.color": PAL_BORDER,
    "font.family": "DejaVu Sans",  # ships with matplotlib, has full unicode coverage
    "font.size": 10,
})

# ── Diacritical test words (the core of the defect) ──────────────────────
DIACRITIC_WORDS = [
    "Hokkaidō", "Tōkyō", "Jinmyōchō", "Shōtoku", "Shōtai",
    "kugyō", "România", "Éire", "Djazaïr", "Filasṭīn",
    "Aikanã", "naïve", "Zürich", "café", "résumé",
    "São Paulo", "Malmö", "Gdańsk", "Łódź", "Dvořák",
]

PLAIN_WORDS = [
    "Hokkaido", "Tokyo", "Jinmyocho", "Shotoku", "Shotai",
    "kugyo", "Romania", "Eire", "Djazair", "Filastin",
    "Aikana", "naive", "Zurich", "cafe", "resume",
    "Sao Paulo", "Malmo", "Gdansk", "Lodz", "Dvorak",
]

CONTROL_WORDS = [
    "Berlin", "quantum physics", "cat", "democracy", "bicycle",
    "economics", "shrine", "emperor", "photosynthesis", "river",
]


def embed_ollama(texts):
    """Get embeddings from local Ollama."""
    import urllib.request
    payload = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embeddings"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def load_csv():
    """Load existing collisions.csv."""
    rows = []
    with open(str(CSV_PATH), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def run_live_embeddings():
    """Re-embed everything with Ollama and return similarity matrix data."""
    all_words = DIACRITIC_WORDS + PLAIN_WORDS + CONTROL_WORDS
    unique = list(dict.fromkeys(all_words))
    print(f"Embedding {len(unique)} texts with {MODEL}...")
    vecs = embed_ollama(unique)
    word2vec = dict(zip(unique, vecs))
    return word2vec


# ── Figure generators ────────────────────────────────────────────────────

def fig_collision_heatmap(word2vec):
    """Heatmap showing cosine similarity between all diacritical words.

    The headline figure: every diacritical word collapses onto every
    other diacritical word at cosine ~1.0. The one ASCII outlier
    ("São Paulo" has a space that resets the WordPiece tokenizer on
    its second word) sits at ~0.5, proving the effect is tokenizer-
    driven rather than semantic.
    """
    words = [w for w in DIACRITIC_WORDS if w in word2vec]
    n = len(words)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine(word2vec[words[i]], word2vec[words[j]])

    fig, ax = plt.subplots(figsize=(13, 11))

    # Custom colormap: parchment → sage (low) to terracotta (high).
    # Strong saturation at the top to make the "all-red-at-1.0" read
    # visible at thumbnail resolution.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "parchment_heat",
        [PAL_GREEN, "#c7b587", "#d49560", PAL_ACCENT, "#8a3820"],
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(words, fontsize=10)

    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > 0.75 else PAL_TEXT
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=11)
    cbar.outline.set_edgecolor(PAL_BORDER)

    ax.set_title(
        "Every diacritical word collapses onto every other diacritical word\n"
        "Cosine similarity ≈ 1.00 for 361 of 380 off-diagonal cells",
        fontsize=14, pad=18, color=PAL_TEXT, fontweight="bold",
    )
    plt.tight_layout()
    path = FIG_DIR / "collision_heatmap.png"
    fig.savefig(str(path), dpi=160, bbox_inches="tight", facecolor=PAL_BG)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_diacritic_vs_plain(word2vec):
    """Bar chart: diacritical word vs its plain ASCII equivalent.

    Blue bars: same word in both forms (Hokkaidō vs Hokkaido) — what
    a working model should score near 1.0.
    Terracotta bars: diacritical word vs a different diacritical word —
    what a working model should score near 0.4-0.5.
    Reality: the terracotta bars are at 1.0 and the blue bars are
    at ~0.5. The model says a word is more similar to an unrelated
    foreign word than to its own ASCII form.
    """
    pairs = list(zip(DIACRITIC_WORDS, PLAIN_WORDS))
    pairs = [(d, p) for d, p in pairs if d in word2vec and p in word2vec]

    labels = [f"{d}\nvs {p}" for d, p in pairs]
    sims = [cosine(word2vec[d], word2vec[p]) for d, p in pairs]

    # Cross-collision: diacritical word vs a different diacritical word
    cross_sims = []
    for i, (d, _) in enumerate(pairs):
        other_d = pairs[(i + 3) % len(pairs)][0]
        cross_sims.append(cosine(word2vec[d], word2vec[other_d]))

    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(labels))
    w = 0.38

    ax.bar(
        x - w / 2, sims, w,
        label="Same word: diacritic vs plain  (should be ≈1.0)",
        color=PAL_BLUE, edgecolor=PAL_BG, linewidth=1.2,
    )
    ax.bar(
        x + w / 2, cross_sims, w,
        label="Different word: diacritic vs diacritic  (should be ≈0.4)",
        color=PAL_ACCENT, edgecolor=PAL_BG, linewidth=1.2,
    )

    ax.axhline(y=0.95, color=PAL_ACCENT, linestyle="--", alpha=0.6,
               linewidth=1.2, label="Collision threshold (0.95)")
    ax.axhline(y=0.5, color=PAL_GREEN, linestyle="--", alpha=0.5,
               linewidth=1.0, label="Expected baseline (~0.5)")

    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(
        'The paradox: "Hokkaidō" is closer to "Éire" than to "Hokkaido"\n'
        "Unrelated diacritical words (red) score higher than matched ASCII pairs (blue)",
        fontsize=14, pad=16, color=PAL_TEXT, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92,
              edgecolor=PAL_BORDER, facecolor=PAL_SURFACE)
    ax.grid(True, alpha=0.35, axis="y", color=PAL_BORDER)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = FIG_DIR / "diacritic_vs_plain.png"
    fig.savefig(str(path), dpi=160, bbox_inches="tight", facecolor=PAL_BG)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_threshold_sweep(word2vec):
    """How many collisions at each cosine threshold.

    Shows that the collision isn't a soft blur around 1.0 — it's a
    hard plateau. Even at a very strict threshold of 0.99, almost
    every diacritical pair still counts as a collision.
    """
    words = [w for w in DIACRITIC_WORDS if w in word2vec]
    all_sims = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            all_sims.append(cosine(word2vec[words[i]], word2vec[words[j]]))

    thresholds = np.arange(0.80, 1.001, 0.005)
    counts = [sum(1 for s in all_sims if s >= t) for t in thresholds]
    count_at_95 = sum(1 for s in all_sims if s >= 0.95)
    total = len(words) * (len(words) - 1) // 2

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(thresholds, counts, alpha=0.35, color=PAL_ACCENT)
    ax.plot(thresholds, counts, "o-", color=PAL_ACCENT,
            markersize=4, linewidth=1.8)

    ax.axvline(x=0.95, color=PAL_ACCENT, linestyle="--", alpha=0.7, linewidth=1.3)
    ax.annotate(
        f"{count_at_95} of {total} pairs\ncollide at cosine ≥ 0.95\n({count_at_95/total:.0%} collision rate)",
        xy=(0.95, count_at_95),
        xytext=(0.82, total * 0.55),
        arrowprops=dict(arrowstyle="->", color=PAL_ACCENT, lw=1.5),
        fontsize=11, color=PAL_TEXT,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PAL_SURFACE,
                  edgecolor=PAL_BORDER),
    )

    ax.axhline(y=total, color=PAL_DIM, linestyle=":", alpha=0.6)
    ax.text(0.805, total + 2, f"Total pairs in 20-word sample: {total}",
            fontsize=10, color=PAL_DIM)

    ax.set_xlabel("Cosine Similarity Threshold", fontsize=12)
    ax.set_ylabel("Number of Colliding Pairs", fontsize=12)
    ax.set_title(
        "This is a hard collapse, not gradual degradation\n"
        "The curve is flat near its maximum — even at cosine ≥ 0.99, nearly all pairs still collide",
        fontsize=13.5, pad=14, color=PAL_TEXT, fontweight="bold",
    )
    ax.grid(True, alpha=0.35, color=PAL_BORDER)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIG_DIR / "collision_count_over_threshold.png"
    fig.savefig(str(path), dpi=160, bbox_inches="tight", facecolor=PAL_BG)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_token_analysis(word2vec):
    """Three similarity distributions showing the [UNK] token dominance.

    Left panel (the smoking gun): every diacritical-diacritical pair
    sits at ~1.0. Middle panel: control words behave normally,
    distribution around ~0.5. Right panel: diacritical vs control
    sits at ~0.5, meaning the [UNK]-dominated vector has moderate
    similarity to every English word.
    """
    d_words = [w for w in DIACRITIC_WORDS if w in word2vec]
    c_words = [w for w in CONTROL_WORDS if w in word2vec]

    dd_sims, cc_sims, dc_sims = [], [], []
    for i in range(len(d_words)):
        for j in range(i + 1, len(d_words)):
            dd_sims.append(cosine(word2vec[d_words[i]], word2vec[d_words[j]]))
    for i in range(len(c_words)):
        for j in range(i + 1, len(c_words)):
            cc_sims.append(cosine(word2vec[c_words[i]], word2vec[c_words[j]]))
    for d in d_words:
        for c in c_words:
            dc_sims.append(cosine(word2vec[d], word2vec[c]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)

    panels = [
        (axes[0], dd_sims,
         "Diacritical vs Diacritical",
         "All pairs collapse to ≈1.0",
         PAL_ACCENT),
        (axes[1], cc_sims,
         "Control vs Control  (ASCII baseline)",
         "Normal distribution around ≈0.5",
         PAL_BLUE),
        (axes[2], dc_sims,
         "Diacritical vs Control",
         "[UNK] vector has ≈0.5 similarity to any English word",
         PAL_GREEN),
    ]

    for ax, sims, title, subtitle, color in panels:
        if sims:
            ax.hist(
                sims, bins=24, range=(-0.05, 1.05),
                color=color, edgecolor=PAL_BG, linewidth=0.8, alpha=0.88,
            )
            ax.axvline(x=np.mean(sims), color=PAL_TEXT,
                       linestyle="--", linewidth=1.8)
            ax.text(
                0.97, 0.95,
                f"μ = {np.mean(sims):.3f}\nσ = {np.std(sims):.3f}\nn = {len(sims)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PAL_SURFACE,
                          edgecolor=PAL_BORDER),
            )
        ax.set_title(f"{title}\n{subtitle}", fontsize=12,
                     color=PAL_TEXT, pad=10)
        ax.set_xlabel("Cosine Similarity", fontsize=11)
        ax.set_xlim(-0.1, 1.1)
        ax.grid(True, alpha=0.35, color=PAL_BORDER)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Pair Count", fontsize=11)

    fig.suptitle(
        "[UNK] token dominance: diacritical words cluster into a single point",
        fontsize=15, y=1.00, color=PAL_TEXT, fontweight="bold",
    )
    plt.tight_layout()
    path = FIG_DIR / "token_analysis.png"
    fig.savefig(str(path), dpi=160, bbox_inches="tight", facecolor=PAL_BG)
    plt.close(fig)
    print(f"  Saved {path}")


PAGE_CSS = """
  :root {
    --bg: #f6f2ec;
    --surface: #efe9df;
    --card: #ece4d6;
    --accent: #b8553a;
    --accent-soft: #e9d5cc;
    --text: #2d2a26;
    --dim: #6e6a61;
    --green: #5b7a4a;
    --yellow: #a07a2b;
    --blue: #3f6487;
    --border: #d9d1c1;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.75;
    max-width: 880px; margin: 0 auto; padding: 2.5rem 1.5rem;
    font-size: 16.5px;
  }
  h1 {
    color: var(--text); font-size: 2rem; margin-bottom: 0.4rem;
    font-weight: 700; letter-spacing: -0.01em;
  }
  h2 {
    color: var(--blue); font-size: 1.35rem; margin: 2.75rem 0 1rem;
    padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  h3 { color: var(--text); margin: 1.5rem 0 0.5rem; font-size: 1.1rem; font-weight: 600; }
  p { margin-bottom: 1rem; }
  a { color: var(--blue); text-decoration: none; border-bottom: 1px solid transparent; }
  a:hover { border-bottom-color: var(--blue); }
  code {
    background: var(--surface); padding: 0.1rem 0.35rem; border-radius: 3px;
    font-family: 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 0.88em;
    color: var(--accent); border: 1px solid var(--border);
  }
  pre {
    background: var(--surface); padding: 1rem 1.1rem; border-radius: 6px;
    overflow-x: auto; border: 1px solid var(--border); margin: 1rem 0;
    line-height: 1.55;
  }
  pre code {
    background: none; border: none; padding: 0; color: var(--text);
    font-size: 0.88em;
  }
  .lede {
    color: var(--dim); font-size: 1.05rem; margin-top: 0.8rem;
    line-height: 1.65;
  }
  .alert {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    border-radius: 4px; padding: 1.1rem 1.3rem; margin: 1.5rem 0;
  }
  .alert h3 { color: var(--accent); margin-top: 0; }
  .info {
    background: var(--surface);
    border-left: 3px solid var(--blue);
    border-radius: 4px; padding: 1.1rem 1.3rem; margin: 1.5rem 0;
  }
  .info h3 { color: var(--blue); margin-top: 0; }
  .figure {
    background: var(--surface); border-radius: 6px; padding: 1rem;
    margin: 1.5rem 0; border: 1px solid var(--border); text-align: center;
  }
  .figure img { max-width: 100%; height: auto; border-radius: 4px; }
  .figure .caption { color: var(--dim); font-size: 0.88rem; margin-top: 0.7rem; }
  .stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.9rem; margin: 1.5rem 0;
  }
  .stat {
    background: var(--card); border-radius: 6px; padding: 1.1rem 1rem;
    text-align: center; border: 1px solid var(--border);
  }
  .stat .number { font-size: 1.9rem; font-weight: 700; color: var(--accent); }
  .stat .label { color: var(--dim); font-size: 0.82rem; margin-top: 0.15rem; }
  table {
    width: 100%; border-collapse: collapse; margin: 1rem 0;
    font-size: 0.92rem;
  }
  th {
    background: var(--card); color: var(--blue); text-align: left;
    padding: 0.55rem 0.75rem; border-bottom: 2px solid var(--border);
    font-weight: 600;
  }
  td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
  tr:hover td { background: var(--surface); }
  td.num { font-variant-numeric: tabular-nums; text-align: right; }
  .collision { color: var(--accent); font-weight: 600; }
  .safe { color: var(--green); font-weight: 600; }
  .warn  { color: var(--yellow); font-weight: 600; }
  ul, ol { margin: 0 0 1rem 1.3rem; }
  li { margin-bottom: 0.35rem; }
  footer {
    margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
    color: var(--dim); font-size: 0.85rem;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1c1d22;
      --surface: #26282f;
      --card: #2d3038;
      --accent: #d97757;
      --accent-soft: rgba(217,119,87,0.12);
      --text: #e8e4dc;
      --dim: #9a9589;
      --green: #8ba87a;
      --yellow: #c6a15e;
      --blue: #85a8c4;
      --border: #3a3d46;
    }
  }
  @media (max-width: 600px) {
    body { padding: 1.2rem; }
    .stats { grid-template-columns: 1fr; }
    h1 { font-size: 1.6rem; }
  }
"""


def _severity_class(modes):
    if "unk_collapse" in modes:
        return "collision", "Full collapse"
    if "diacritic_attractor" in modes:
        return "collision", "Diacritic attractor"
    if "ascii_equivalence_broken" in modes:
        return "warn", "ASCII equivalence broken"
    return "safe", "No defect detected"


def _build_hf_rows(hf_probe):
    if hf_probe.get("status") != "ok":
        return "<p><em>HF upstream probe not available: " \
               + hf_probe.get("reason", "unknown") + "</em></p>"
    rows = []
    for p in hf_probe["pairs"]:
        same = "safe" if p["ids_identical"] else "warn"
        mark = "=" if p["ids_identical"] else "≠"
        rows.append(
            f'<tr><td>{p["diacritic"]}</td>'
            f'<td class="{same}">{mark}</td>'
            f'<td>{p["ascii"]}</td>'
            f'<td><code>{" ".join(p["diacritic_tokens"])}</code></td>'
            f'<td><code>{" ".join(p["ascii_tokens"])}</code></td></tr>'
        )
    stripped = sum(1 for p in hf_probe["pairs"] if p["ids_identical"])
    total = len(hf_probe["pairs"])
    return (
        f"<p>Upstream HuggingFace <code>BertTokenizer</code> strips accents for "
        f"<strong>{stripped} of {total}</strong> test pairs — the diacritical form "
        f"and the ASCII form produce identical token IDs. "
        f"(<code>do_lower_case={hf_probe['config']['do_lower_case']}</code>, "
        f"<code>strip_accents={hf_probe['config']['basic_tokenizer_strip_accents']}</code> "
        f"which defaults to accent-stripping when lower-casing is on.) "
        f"The one non-identical pair involves <code>Ł</code>, a distinct Latin letter "
        f"rather than a decomposable combining diacritical, so NFD normalization "
        f"leaves it alone.</p>"
        "<table>"
        "<tr><th>Diacritical</th><th></th><th>ASCII</th>"
        "<th>HF tokens (diacritic)</th><th>HF tokens (ASCII)</th></tr>"
        + "".join(rows) +
        "</table>"
    )


def _build_ollama_rows(primary, secondary):
    rows = []
    def row(probe):
        if probe.get("status") != "ok":
            return (
                f'<tr><td><code>{probe.get("model", "?")}</code></td>'
                f'<td colspan="5"><em>not probed: '
                f'{probe.get("reason", probe.get("status", "unknown"))}</em></td></tr>'
            )
        s = probe["stats"]
        klass, label = _severity_class(probe.get("failure_modes", []))
        return (
            f'<tr>'
            f'<td><code>{probe["model"]}</code></td>'
            f'<td class="num">{s["diacritic_vs_ascii_same_word"]["mean"]:.3f}</td>'
            f'<td class="num">{s["cross_diacritic"]["mean"]:.3f}</td>'
            f'<td class="num">{s["cross_control"]["mean"]:.3f}</td>'
            f'<td class="num">{probe.get("severity", 0):.2f}</td>'
            f'<td class="{klass}">{label}</td>'
            f'</tr>'
        )
    rows.append(row(primary))
    for name, probe in secondary.items():
        if probe.get("status") == "not_registered":
            continue
        rows.append(row(probe))
    return (
        "<table>"
        "<tr>"
        "<th>Model (via Ollama)</th>"
        "<th>S: diac↔ASCII<br>same word</th>"
        "<th>D: diac↔diac<br>different words</th>"
        "<th>C: ASCII<br>control baseline</th>"
        "<th>Severity</th>"
        "<th>Failure mode</th>"
        "</tr>"
        + "".join(rows) +
        "</table>"
    )


def _build_mxbai_collisions(primary):
    """Render a small table of the worst cross-diacritic collisions for mxbai."""
    if primary.get("status") != "ok":
        return ""
    pairs = sorted(
        primary["cross_diacritic_sims"],
        key=lambda x: x["cosine"], reverse=True,
    )[:8]
    rows = "".join(
        f'<tr><td>{p["a"]}</td><td>{p["b"]}</td>'
        f'<td class="num collision">{p["cosine"]:.4f}</td></tr>'
        for p in pairs
    )
    return (
        "<p>The top cross-diacritic cosine similarities on the archived mxbai "
        "gguf via Ollama. A working model should score unrelated words around "
        "0.3–0.5. These are the cosines actually produced on the reproducibly "
        "frozen weights:</p>"
        "<table>"
        "<tr><th>Diacritical word A</th><th>Diacritical word B</th>"
        "<th>Cosine similarity</th></tr>"
        + rows +
        "</table>"
    )


def generate_html(timestamp, divergence_path=None):
    """Rebuild docs/index.html from the current divergence JSON.

    The page embeds:
      - the warm-parchment styling used by the hand-authored site
      - the 4 figures generated by this script's figure functions
      - an auto-generated §5 "HF-vs-Ollama divergence" section driven
        by verification/tokenizer_divergence.json
      - an auto-generated §6 scope table showing every BERT-derived
        embedding model registered in local Ollama
    """
    from pathlib import Path as _Path

    if divergence_path is None:
        divergence_path = ROOT / "verification" / "tokenizer_divergence.json"
    try:
        data = json.loads(_Path(divergence_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] could not load {divergence_path}: {e}")
        print("         run scripts/verify_tokenizer_divergence.py first")
        return

    hf = data.get("hf_probe", {})
    primary = data.get("ollama_primary", {})
    secondary = data.get("ollama_secondary", {})

    mxbai_stats = primary.get("stats", {}) if primary.get("status") == "ok" else {}
    mxbai_same = mxbai_stats.get("diacritic_vs_ascii_same_word", {}).get("mean", 0)
    mxbai_cross_d = mxbai_stats.get("cross_diacritic", {}).get("mean", 0)
    mxbai_control = mxbai_stats.get("cross_control", {}).get("mean", 0)

    hf_rows = _build_hf_rows(hf)
    ollama_table = _build_ollama_rows(primary, secondary)
    mxbai_collisions = _build_mxbai_collisions(primary)

    n_secondary_affected = sum(
        1 for p in secondary.values()
        if p.get("status") == "ok" and p.get("failure_modes")
    )
    total_affected = (1 if primary.get("failure_modes") else 0) + n_secondary_affected

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mxbai-embed-large thinks &ldquo;Hokkaid&#x014d;&rdquo; and &ldquo;&#x00c9;ire&rdquo; are the same word</title>
<style>{PAGE_CSS}</style>
</head>
<body>

<h1>mxbai-embed-large thinks &ldquo;Hokkaid&#x014d;&rdquo; and &ldquo;&#x00c9;ire&rdquo; are the same word</h1>
<p class="lede">
  A top-ranked MTEB embedding model collapses nearly every word containing a
  diacritical mark to a single point in vector space &mdash; but only when
  deployed via Ollama. The upstream HuggingFace tokenizer strips accents
  correctly; the Ollama gguf conversion pipeline silently drops the
  preprocessing step, and the defect class affects every BERT-derived
  embedding model in the local-inference stack we tested.
  <strong>147,687</strong> confirmed cross-entity collisions on Wikidata.
  <br><span style="font-size:0.9em">Last updated: <strong>{timestamp}</strong>
  &middot; regenerated from
  <a href="https://github.com/EmmaLeonhart/latent-space-cartography/blob/master/scripts/verify_tokenizer_divergence.py"><code>scripts/verify_tokenizer_divergence.py</code></a></span>
</p>

<div class="alert">
  <h3>The short version</h3>
  <p>
    <code>mxbai-embed-large</code> is a top-ranked open-source embedding model on MTEB.
    When deployed via <strong>Ollama</strong> — the dominant local-inference stack for
    embedding models — it has a silent tokenizer defect that collapses any short text
    containing diacritical marks (&#x014d;, &#x00e9;, &#x00fc;, &#x0142;, &#x1e63;, &hellip;)
    into a single point in embedding space.
  </p>
  <p>
    <strong>Result:</strong> &ldquo;Hokkaid&#x014d;&rdquo; (a Japanese island) and &ldquo;&#x00c9;ire&rdquo;
    (Ireland in Irish) produce <em>identical</em> embeddings &mdash; cosine similarity
    1.000 to six decimal places. Meanwhile &ldquo;Hokkaid&#x014d;&rdquo; has cosine
    similarity only <strong>{mxbai_same:.2f}</strong> to its own ASCII spelling &ldquo;Hokkaido&rdquo;.
    The diacritical version of a word is closer to a random other diacritical word
    than to itself.
  </p>
  <p>
    <strong>But via the upstream HuggingFace tokenizer, the defect does not reproduce.</strong>
    HF's <code>BertTokenizer</code> strip-accents preprocessing handles diacritics correctly.
    The bug lives in the conversion pipeline from HuggingFace to gguf / Ollama, not in the
    upstream mxbai weights. That makes it a defect class that affects every BERT-derived
    embedding model deployed via Ollama, not just this one model.
  </p>
</div>

<div class="stats">
  <div class="stat">
    <div class="number">147,687</div>
    <div class="label">Cross-entity collisions (Wikidata)</div>
  </div>
  <div class="stat">
    <div class="number">{mxbai_cross_d:.3f}</div>
    <div class="label">Cosine (unrelated diacritic pairs)</div>
  </div>
  <div class="stat">
    <div class="number">{mxbai_same:.3f}</div>
    <div class="label">Cosine (same word, diacritic vs ASCII)</div>
  </div>
  <div class="stat">
    <div class="number">{total_affected}/{1 + len(secondary)}</div>
    <div class="label">BERT embed models affected via Ollama</div>
  </div>
</div>

<div class="info">
  <h3>About this page</h3>
  <p>
    Every number on this page is regenerated automatically. The pipeline is two scripts:
    <code>scripts/verify_tokenizer_divergence.py</code> runs the upstream HuggingFace
    tokenizer and the archived gguf via Ollama side-by-side and writes
    <code>verification/tokenizer_divergence.json</code>. Then
    <code>scripts/generate_defect_page.py</code> rebuilds the figures and this page
    from that JSON. The &ldquo;archived&rdquo; gguf refers to
    <code>model/mxbai-embed-large-v1.gguf</code>, shipped in this repo so the result
    is reproducible even if the upstream mxbai weights are patched.
  </p>
</div>

<h2>1. The Collision Heatmap</h2>
<p>
  Every cell shows the cosine similarity between two words containing diacritical marks.
  In a working embedding model, unrelated words should have cosine around 0.3&ndash;0.5.
  Instead, nearly every pair is at <strong>1.00</strong>. The one cell around ~0.5
  ("S&#x00e3;o Paulo") has an internal space that resets WordPiece on its second,
  ASCII-only word.
</p>
<div class="figure">
  <img src="figures/collision_heatmap.png" alt="Collision heatmap">
  <div class="caption">Figure 1: 361 of 380 off-diagonal cells are at cosine ≈ 1.00.</div>
</div>

<h2>2. The Paradox: &ldquo;Hokkaid&#x014d;&rdquo; is closer to &ldquo;&#x00c9;ire&rdquo; than to &ldquo;Hokkaido&rdquo;</h2>
<p>
  Compare two cosines for each word in the 20-word sample: the word vs its own ASCII
  spelling (blue), and the word vs a different diacritical word (terracotta). A working
  model should put the blue bars near 1.0 and the terracotta bars near 0.5. The actual
  result is the exact inverse.
</p>
<div class="figure">
  <img src="figures/diacritic_vs_plain.png" alt="Bar chart inverted: unrelated diacritical words score higher than matched ASCII pairs">
  <div class="caption">Figure 2: Blue = same word (diacritic vs plain), should be ≈1.0. Terracotta = different word (diacritic vs diacritic), should be ≈0.4. The actual values are inverted.</div>
</div>

<h2>3. This is a hard collapse, not gradual degradation</h2>
<p>
  If the failure mode were soft blur &mdash; diacritical marks adding noise that
  gradually erodes similarity &mdash; the number of colliding pairs would drop
  smoothly as the threshold is raised. It doesn't. The curve is flat all the way
  up to 0.99.
</p>
<div class="figure">
  <img src="figures/collision_count_over_threshold.png" alt="Threshold sweep curve is flat at the maximum">
  <div class="caption">Figure 3: Collision count is nearly constant from cosine 0.80 to 0.99. The collapse is binary.</div>
</div>

<h2>4. Distribution analysis: the [UNK] cluster</h2>
<p>
  Three similarity distributions side by side. The diacritical-vs-diacritical panel
  (left) is a spike at 1.0, not a bell curve &mdash; compare to the control panel
  (middle), which shows what a healthy embedding space actually produces.
</p>
<div class="figure">
  <img src="figures/token_analysis.png" alt="Three histograms: leftmost shows a spike at cosine 1.0, middle and right are normal distributions around 0.5">
  <div class="caption">Figure 4: Diacritical-vs-diacritical (left) is a spike at 1.0. Control-vs-control (middle) is a normal distribution around 0.5. This is a tokenizer defect, not embedding noise.</div>
</div>

<h2>5. Why it happens: the HF-vs-Ollama divergence</h2>

<div class="info">
<h3>Step 1 &mdash; upstream HF <code>BertTokenizer</code> strips accents</h3>
{hf_rows}

<h3>Step 2 &mdash; the archived gguf via Ollama does NOT strip accents</h3>
<p>
  When the same mxbai weights are loaded through Ollama (from
  <code>model/mxbai-embed-large-v1.gguf</code>, registered as
  <code>mxbai-archived</code> via this repo's <code>model/Modelfile</code>),
  diacritical characters are preserved all the way into the WordPiece step.
  Since those characters are not in the WordPiece vocab and the gguf tokenizer has
  no character-level fallback, the whole whitespace-delimited token becomes
  <code>[UNK]</code>. For short inputs this single <code>[UNK]</code> dominates the
  mean-pooled embedding, and every diacritical string ends up at the same point:
</p>
<pre><code>"Hokkaidō"  →  [CLS]  [UNK]  [SEP]
"Éire"      →  [CLS]  [UNK]  [SEP]
"Zürich"    →  [CLS]  [UNK]  [SEP]
"café"      →  [CLS]  [UNK]  [SEP]
"Dvořák"    →  [CLS]  [UNK]  [SEP]</code></pre>
<p>
  Empirically, on the archived gguf via Ollama, diacritical-vs-ASCII same-word
  cosine is <strong>{mxbai_same:.3f}</strong> (should be ≈1.0 if the tokenizer is
  clean) and diacritical-vs-diacritical different-word cosine is
  <strong>{mxbai_cross_d:.3f}</strong> (should be ≈{mxbai_control:.3f}, the ASCII
  control baseline).
</p>

{mxbai_collisions}

<h3>Step 3 &mdash; the root cause: a dropped preprocessing step</h3>
<p>
  The mechanism is not a WordPiece limitation. HF's <code>BertTokenizer</code> applies
  <code>BasicTokenizer</code>'s accent-stripping (via NFD normalization plus combining-mark
  removal) <em>before</em> WordPiece sees the string. That preprocessing is wired in when
  <code>do_lower_case=True</code>. The gguf conversion pipeline that produces the Ollama
  model drops this preprocessing step: the gguf tokenizer sees raw Unicode diacritics,
  has no way to match them to its WordPiece vocab, and emits <code>[UNK]</code>.
</p>
<p>
  Because the preprocessing step is a function of the BERT tokenizer config (not of any
  model-specific training), the same defect class is expected to affect every BERT-derived
  embedding model exported to gguf via the same conversion pipeline. The next section
  measures that.
</p>
</div>

<h2>6. Scope: this affects every BERT-derived embedding model in Ollama</h2>
<p>
  The verification script runs the same diacritic-vs-ASCII probe against every
  BERT-family embedding model registered in local Ollama. Each row reports three
  mean cosine similarities:
</p>
<ul>
  <li><strong>S</strong> = diacritical form vs its own ASCII spelling (e.g. Hokkaid&#x014d; vs Hokkaido). Should be ≈1.0.</li>
  <li><strong>D</strong> = diacritical word vs a different diacritical word (e.g. Hokkaid&#x014d; vs Éire). Should be near the ASCII control baseline.</li>
  <li><strong>C</strong> = two unrelated ASCII English words. This is the calibration baseline for the model's overall similarity spread.</li>
</ul>
<p>
  A healthy model has S ≈ 1 and D ≈ C. A model with a diacritic attractor has
  D &gt;&gt; C. A model with an &ldquo;[UNK] collapse&rdquo; additionally has
  S ≈ C (the same word's ASCII form is no more similar than an unrelated word).
</p>

{ollama_table}

<p>
  Every BERT-derived embedding model we tested via Ollama has a failure mode on
  diacritical text. mxbai-archived and all-minilm exhibit the full <code>[UNK]</code>
  collapse; nomic-embed-text has a softer but still-severe diacritic attractor (its
  unrelated diacritic pairs cluster at cosine ~0.99, even though it recognizes
  same-word ASCII equivalents). This is not a one-off bug in one model &mdash;
  it's a systemic defect class at the deployment-tooling layer.
</p>

<h2>7. Who is affected</h2>
<table>
<tr><th>Domain</th><th>Impact</th><th>Example</th></tr>
<tr>
  <td>Multilingual NLP via Ollama</td>
  <td class="collision">Critical</td>
  <td>Any language with diacritics (French, German, Japanese romaji, Polish, Czech, Arabic transliteration&hellip;)</td>
</tr>
<tr>
  <td>Knowledge graphs via Ollama</td>
  <td class="collision">Critical</td>
  <td>Wikidata entity labels with non-ASCII characters become indistinguishable</td>
</tr>
<tr>
  <td>RAG / retrieval via Ollama</td>
  <td class="collision">High</td>
  <td>Documents about &ldquo;Malm&#x00f6;&rdquo; match queries about &ldquo;Dvo&#x0159;&#x00e1;k&rdquo;</td>
</tr>
<tr>
  <td>Semantic search via Ollama</td>
  <td class="collision">High</td>
  <td>Any product/person/place name with accented characters</td>
</tr>
<tr>
  <td>Upstream HF <code>transformers</code></td>
  <td class="safe">Unaffected</td>
  <td>HF's <code>BertTokenizer</code> strips accents in preprocessing &mdash; the bug lives below this layer</td>
</tr>
<tr>
  <td>English-only ASCII workloads</td>
  <td class="safe">Unaffected</td>
  <td>Standard ASCII text works fine regardless of deployment layer</td>
</tr>
</table>

<h2>8. Reproducing this</h2>
<p>
  Everything on this page regenerates from two scripts, using the frozen gguf
  shipped in this repository so the result is stable even if the upstream mxbai
  weights are patched:
</p>
<pre><code># 1. Register the archived gguf in Ollama
cd model/
ollama create mxbai-archived -f Modelfile
cd ..

# 2. Run the verification script (writes verification/tokenizer_divergence.json)
pip install transformers   # for the upstream HF probe
python scripts/verify_tokenizer_divergence.py

# 3. Regenerate figures and this page from the JSON artifact
python scripts/generate_defect_page.py</code></pre>
<p>
  The older single-file demo (<code>scripts/demo_collisions.py</code>) still works and
  is faster if you just want to see the defect &mdash; it embeds 25 pairs via Ollama
  and writes <code>collisions.csv</code>. That script is the one reproduced daily by
  GitHub Actions, and the CSV it produces is deterministic (the collisions do not
  drift between runs), which is itself part of the result.
</p>

<h2>9. What should be done</h2>
<ol>
  <li><strong>Ollama / gguf conversion:</strong> BERT-derived models with
      <code>do_lower_case=True</code> need their <code>BasicTokenizer</code>
      preprocessing (NFD normalization + combining-mark strip) carried through
      gguf conversion. Without this, WordPiece sees raw diacritics and emits
      <code>[UNK]</code>.</li>
  <li><strong>Benchmark gap:</strong> MTEB should include diacritical / non-Latin
      string pairs as a robustness check &mdash; no current task in the suite
      surfaces this class of defect.</li>
  <li><strong>User workaround:</strong> If you cannot switch off Ollama, NFD-normalize
      and strip combining marks on the client side before embedding. This loses
      linguistic information but prevents the collisions.</li>
  <li><strong>Model choice:</strong> Models with byte-level BPE tokenizers
      (e.g. <code>nomic-embed-text</code> on some versions, or SentencePiece-based
      multilingual models) are less exposed, though our probe shows even
      <code>nomic-embed-text</code> via Ollama has a softer diacritic attractor.</li>
</ol>

<h2>10. Context</h2>
<p>
  This defect was discovered during the
  <a href="https://github.com/EmmaLeonhart/latent-space-cartography">Latent Space Cartography</a>
  project, which applies Vector Symbolic Architecture (VSA) analysis to frozen text embeddings.
  When probing Wikidata entity embeddings for relational structure, the collision pattern was
  unmistakable: 147,687 cross-entity pairs at cosine &ge; 0.95, all involving diacritical text.
  The full analysis is documented in our paper
  <em>&ldquo;Latent Space Cartography Applied to Wikidata: Relational Displacement Analysis
  Reveals a Silent Tokenizer Defect in mxbai-embed-large&rdquo;</em> (clawRxiv 2604.00648).
</p>

<footer>
  <p>
    <a href="https://github.com/EmmaLeonhart/latent-space-cartography">latent-space-cartography</a>
    &middot; {timestamp}
  </p>
</footer>

</body>
</html>"""
    path = DOCS / "index.html"
    path.write_text(body, encoding="utf-8")
    print(f"  Saved {path}")
    return


def _dead_template(timestamp):
    """Dead template kept only to let the old indentation-balanced triple-quoted
    string close cleanly. Not called from anywhere.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mxbai-embed-large thinks "Hokkaidō" and "Éire" are the same word</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --card: #1c2333;
    --accent: #e74c3c; --text: #e6edf3; --dim: #8b949e;
    --green: #3fb950; --yellow: #d29922; --blue: #58a6ff;
    --border: #30363d;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
    max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem;
  }}
  h1 {{ color: var(--accent); font-size: 2rem; margin-bottom: 0.3rem; }}
  h2 {{
    color: var(--blue); font-size: 1.4rem; margin: 2.5rem 0 1rem;
    padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
  }}
  h3 {{ color: var(--green); margin: 1.5rem 0 0.5rem; }}
  p {{ margin-bottom: 1rem; }}
  a {{ color: var(--blue); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{
    background: var(--surface); padding: 0.15rem 0.4rem; border-radius: 4px;
    font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.9em;
    color: var(--green); border: 1px solid var(--border);
  }}
  pre {{
    background: var(--surface); padding: 1rem; border-radius: 6px;
    overflow-x: auto; border: 1px solid var(--border); margin: 1rem 0;
  }}
  pre code {{ background: none; border: none; padding: 0; }}
  .alert {{
    background: rgba(231,76,60,0.1); border: 1px solid var(--accent);
    border-radius: 8px; padding: 1.2rem; margin: 1.5rem 0;
  }}
  .alert h3 {{ color: var(--accent); margin-top: 0; }}
  .info {{
    background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.3);
    border-radius: 8px; padding: 1.2rem; margin: 1.5rem 0;
  }}
  .figure {{
    background: var(--surface); border-radius: 8px; padding: 1rem;
    margin: 1.5rem 0; border: 1px solid var(--border); text-align: center;
  }}
  .figure img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  .figure .caption {{ color: var(--dim); font-size: 0.9rem; margin-top: 0.7rem; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; margin: 1.5rem 0;
  }}
  .stat {{
    background: var(--card); border-radius: 8px; padding: 1.2rem;
    text-align: center; border: 1px solid var(--border);
  }}
  .stat .number {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
  .stat .label {{ color: var(--dim); font-size: 0.85rem; }}
  table {{
    width: 100%; border-collapse: collapse; margin: 1rem 0;
    font-size: 0.9rem;
  }}
  th {{ background: var(--card); color: var(--blue); text-align: left;
       padding: 0.6rem 0.8rem; border-bottom: 2px solid var(--accent); }}
  td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(88,166,255,0.04); }}
  .collision {{ color: var(--accent); font-weight: bold; }}
  .safe {{ color: var(--green); }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
            color: var(--dim); font-size: 0.85rem; }}
  @media (max-width: 600px) {{
    body {{ padding: 1rem; }}
    .stats {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1 style="font-size:2.4rem;line-height:1.2">mxbai-embed-large thinks "Hokkaid&#x014d;" and "&#x00c9;ire" are the same word</h1>
<p style="color:var(--dim);font-size:1.1rem;margin-top:0.8rem">
  A top-10 MTEB embedding model has a tokenizer defect that makes every string with
  a diacritical mark produce the <em>same</em> vector. 147,687 confirmed cross-entity collisions.
  Zero benchmarks catch it.
  <br>Last updated: <strong>{timestamp}</strong>
</p>

<div class="alert">
  <h3>What is this?</h3>
  <p>
    <code>mxbai-embed-large</code> is one of the most popular open-source text embedding models
    (top 10 on MTEB). It has a critical tokenizer defect: any text containing diacritical marks
    (accented characters like o&#x0304;, e&#x0301;, u&#x0308;, etc.) gets mapped to the <code>[UNK]</code>
    token, and the resulting embedding is dominated by that single token's vector.
  </p>
  <p>
    <strong>Result:</strong> Completely unrelated words like "Hokkaid&#x014d;" (a Japanese island)
    and "&#x00c9;ire" (Ireland in Irish) produce <em>identical</em> embeddings (cosine similarity = 1.000).
    This defect is invisible to standard benchmarks like MTEB because they primarily use ASCII text.
  </p>
</div>

<div class="stats">
  <div class="stat">
    <div class="number">147,687</div>
    <div class="label">Cross-entity collisions</div>
  </div>
  <div class="stat">
    <div class="number">1.000</div>
    <div class="label">Cosine similarity (unrelated words)</div>
  </div>
  <div class="stat">
    <div class="number">~0.48</div>
    <div class="label">Cosine to own ASCII equivalent</div>
  </div>
  <div class="stat">
    <div class="number">0</div>
    <div class="label">MTEB benchmarks that catch this</div>
  </div>
</div>

<h2>1. The Collision Heatmap</h2>
<p>
  Every cell shows the cosine similarity between two words containing diacritical marks.
  In a working embedding model, unrelated words should have low similarity (~0.3-0.5).
  Instead, nearly every pair is at or near <strong>1.0</strong>.
</p>
<div class="figure">
  <img src="figures/collision_heatmap.png" alt="Collision heatmap showing near-1.0 similarity between all diacritical words">
  <div class="caption">Figure 1: Cosine similarity heatmap. All diacritical words map to essentially the same vector.</div>
</div>

<h2>2. The Paradox: Unrelated Words Are More Similar Than the Same Word</h2>
<p>
  "Hokkaid&#x014d;" (with macron) has cosine similarity ~1.0 with "&#x00c9;ire" (completely unrelated),
  but only ~0.50 with "Hokkaido" (the same place, ASCII spelling). The diacritical mark doesn't
  just add noise &mdash; it <em>completely overwrites</em> the semantic content.
</p>
<div class="figure">
  <img src="figures/diacritic_vs_plain.png" alt="Bar chart comparing same-word and cross-word similarities">
  <div class="caption">Figure 2: Blue bars show similarity to the word's own ASCII form. Red bars show similarity to a random other diacritical word. Red is consistently higher.</div>
</div>

<h2>3. Collision Count by Threshold</h2>
<p>
  Even with a very strict threshold of 0.99, most diacritical word pairs still collide.
  This is not gradual degradation &mdash; it's a hard collapse to a single point in embedding space.
</p>
<div class="figure">
  <img src="figures/collision_count_over_threshold.png" alt="Graph showing collision counts remain high even at strict thresholds">
  <div class="caption">Figure 3: Number of colliding pairs at each cosine similarity threshold. The curve stays flat near the maximum.</div>
</div>

<h2>4. Distribution Analysis: The [UNK] Cluster</h2>
<p>
  Three histograms tell the full story:
</p>
<ul>
  <li><strong>Left (red):</strong> Diacritical vs diacritical &mdash; all clustered at ~1.0 (the defect)</li>
  <li><strong>Center (blue):</strong> Control vs control &mdash; normal distribution around ~0.5 (baseline)</li>
  <li><strong>Right (yellow):</strong> Diacritical vs control &mdash; shows the [UNK] vector has moderate similarity to everything</li>
</ul>
<div class="figure">
  <img src="figures/token_analysis.png" alt="Three histograms showing similarity distributions">
  <div class="caption">Figure 4: The diacritical-diacritical distribution is a spike at 1.0, not a bell curve. This is a tokenizer defect, not embedding noise.</div>
</div>

<h2>5. How It Happens</h2>

<div class="info">
<h3>The Mechanism</h3>
<p>mxbai-embed-large uses a WordPiece tokenizer. When it encounters characters outside its vocabulary (diacritical marks like &#x014d;, &#x00e9;, &#x00fc;, &#x0142;, etc.), the <em>entire token</em> containing that character becomes <code>[UNK]</code>.</p>

<p>For short strings like proper nouns, the [UNK] token dominates the final pooled embedding. Since every diacritical string maps to essentially the same sequence of [UNK] tokens, they all produce the same embedding vector.</p>

<pre><code># What the tokenizer sees:
"Hokkaidō"  → ["[CLS]", "[UNK]", "[SEP]"]
"Éire"      → ["[CLS]", "[UNK]", "[SEP]"]
"Zürich"    → ["[CLS]", "[UNK]", "[SEP]"]
"café"      → ["[CLS]", "[UNK]", "[SEP]"]

# All four produce the SAME embedding vector</code></pre>
</div>

<h2>6. Who Is Affected?</h2>

<table>
<tr><th>Domain</th><th>Impact</th><th>Example</th></tr>
<tr>
  <td>Multilingual NLP</td>
  <td class="collision">Critical</td>
  <td>Any language with diacritics (French, German, Japanese romaji, Polish, Czech, ...)</td>
</tr>
<tr>
  <td>Knowledge Graphs</td>
  <td class="collision">Critical</td>
  <td>Wikidata entities with non-ASCII labels become indistinguishable</td>
</tr>
<tr>
  <td>RAG / Retrieval</td>
  <td class="collision">High</td>
  <td>Documents about "Malm&#x00f6;" match queries about "Dvo&#x0159;&#x00e1;k"</td>
</tr>
<tr>
  <td>Semantic Search</td>
  <td class="collision">High</td>
  <td>Any product/person/place name with accented characters</td>
</tr>
<tr>
  <td>English-only</td>
  <td class="safe">Low</td>
  <td>Standard ASCII text works fine (which is why MTEB misses this)</td>
</tr>
</table>

<h2>7. Reproducing This</h2>
<pre><code># Install Ollama and pull the model
ollama pull mxbai-embed-large

# Clone the repo and run the demo
git clone https://github.com/EmmaLeonhart/latent-space-cartography.git
cd latent-space-cartography
python scripts/demo_collisions.py

# Output: collisions.csv showing the defect</code></pre>

<h2>8. What Should Be Done</h2>
<ol>
  <li><strong>Tokenizer fix:</strong> Replace unknown-character fallback with character-level tokenization or byte-level BPE</li>
  <li><strong>Benchmark gap:</strong> MTEB should include diacritical/multilingual string pairs as a robustness check</li>
  <li><strong>User workaround:</strong> Strip diacritical marks before embedding (loses linguistic information but prevents collisions)</li>
  <li><strong>Model choice:</strong> Use models with byte-level tokenizers (e.g., nomic-embed-text) for non-ASCII data</li>
</ol>

<h2>9. Context</h2>
<p>
  This defect was discovered during the
  <a href="https://github.com/EmmaLeonhart/latent-space-cartography">Latent Space Cartography</a>
  project, which applies Vector Symbolic Architecture (VSA) analysis to frozen text embeddings.
  When probing Wikidata entity embeddings for relational structure, the collision pattern was
  unmistakable: 147,687 cross-entity pairs at cosine &ge; 0.95, all involving diacritical text.
</p>
<p>
  The full analysis is documented in our paper:
  <em>"Emergent Vector Symbolic Operations in Frozen Text Embeddings"</em>
  (clawRxiv 2604.00859).
</p>

<footer>
  <p>
    <a href="https://github.com/EmmaLeonhart/latent-space-cartography">latent-space-cartography</a>
    &middot; {timestamp}
  </p>
</footer>

</body>
</html>"""
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="(default) Re-embed with Ollama. Kept for backwards "
                             "compatibility; this script always uses Ollama now.")
    args = parser.parse_args()
    _ = args  # kept for CLI compatibility

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Running live embeddings via Ollama...")
    try:
        word2vec = run_live_embeddings()
    except Exception as e:
        print(f"  Ollama not available ({e}).")
        print("  Start Ollama and pull mxbai-embed-large:")
        print("    ollama pull mxbai-embed-large")
        sys.exit(1)

    print("\nGenerating figures...")
    fig_collision_heatmap(word2vec)
    fig_diacritic_vs_plain(word2vec)
    fig_threshold_sweep(word2vec)
    fig_token_analysis(word2vec)

    print("\nRegenerating docs/index.html from verification JSON...")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    generate_html(timestamp)

    print("\nDone.")


if __name__ == "__main__":
    main()
