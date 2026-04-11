"""
Wikidata Collision Scan — end-to-end reproduction of the mxbai defect
=====================================================================

Pulls a stratified sample of ~800 Wikidata entities with native-language
labels (which reliably contain diacritics), plus ~200 English-labelled
entities as an ASCII control. Embeds everything via mxbai-embed-large
on local Ollama and computes pairwise collision statistics.

This is the full reproducibility experiment that the daily GitHub
Actions cron runs. It is NOT the full Engishiki BFS from the paper
(which walks ~34k entities via random_walk.py); it is a directly
targeted sample, which is cheaper to run and easier to interpret.

Output:
  - wikidata_collision_summary.json   summary stats for the CI log
  - prints human-readable report to stdout
  - exits non-zero if the defect appears to have been fixed

Requires:
  - Ollama running with mxbai-embed-large pulled
  - Network access to https://query.wikidata.org/sparql
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "wikidata_collision_summary.json"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "mxbai-embed-large"
USER_AGENT = (
    "latent-space-cartography/1.0 "
    "(https://github.com/EmmaLeonhart/latent-space-cartography; "
    "mxbai defect reproduction)"
)
COLLISION_THRESHOLD = 0.95

# Diacritical sample: human settlements per country, taking labels in
# the local language. Settlements-per-country is a small enough
# namespace to query without hitting the Wikidata 60s timeout, and
# native-language city/village names almost always contain the diacritic
# characters we care about. Humans-per-country (the obvious first
# choice) hits millions of results and 504s on anything large.
DIACRITIC_QUERIES = [
    # (description, country QID, label language tag)
    ("Czech settlements",       "Q213", "cs"),
    ("Polish settlements",      "Q36",  "pl"),
    ("Vietnamese settlements",  "Q881", "vi"),
    ("German settlements",      "Q183", "de"),
    ("French settlements",      "Q142", "fr"),
    ("Romanian settlements",    "Q218", "ro"),
    ("Hungarian settlements",   "Q28",  "hu"),
    ("Turkish settlements",     "Q43",  "tr"),
    ("Slovak settlements",      "Q214", "sk"),
    ("Portuguese settlements",  "Q45",  "pt"),
    ("Spanish settlements",     "Q29",  "es"),
    ("Icelandic settlements",   "Q189", "is"),
]

# ASCII control: English-labelled settlements in English-speaking
# countries, filtered to pure ASCII. These should all behave normally.
ASCII_CONTROL = ("UK / US / AU / NZ settlements (ASCII-only en labels)",
                 ["Q145", "Q30", "Q408", "Q664"], "en")

PER_QUERY_LIMIT = 120   # ~12 * 120 = ~1440 candidates before ASCII filter
CONTROL_FETCH = 300     # over-fetch, then filter to ASCII-only
BATCH_SIZE = 32         # Ollama embedding batch size
SPARQL_RETRIES = 2      # per-query retries on transient failures


# ── SPARQL ────────────────────────────────────────────────────────────────

def sparql(query: str) -> list[dict]:
    """POST a SPARQL query to Wikidata with retries on transient failures."""
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    last_err: Exception | None = None
    for attempt in range(SPARQL_RETRIES + 1):
        try:
            req = urllib.request.Request(
                SPARQL_ENDPOINT,
                data=data,
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["results"]["bindings"]
        except Exception as exc:
            last_err = exc
            if attempt < SPARQL_RETRIES:
                time.sleep(2.0 * (attempt + 1))
    raise last_err if last_err else RuntimeError("SPARQL failed")


def fetch_settlements_from_country(country_qid: str, lang: str, limit: int) -> list[tuple[str, str]]:
    """Fetch human settlements in a country, taking labels in the given language.

    No ORDER BY — on large namespaces Wikidata's query planner forces a
    full enumeration before sorting and times out at 60s. The collision
    statistics are robust to which specific settlements get sampled, so
    we accept whatever set the query service returns first.
    """
    query = f"""
    SELECT DISTINCT ?item ?label WHERE {{
      ?item wdt:P31 wd:Q486972 .
      ?item wdt:P17 wd:{country_qid} .
      ?item rdfs:label ?label .
      FILTER(LANG(?label) = "{lang}")
    }}
    LIMIT {limit}
    """
    rows = sparql(query)
    return [
        (b["item"]["value"].rsplit("/", 1)[1], b["label"]["value"])
        for b in rows
    ]


def fetch_ascii_control(countries: list[str], lang: str, limit: int) -> list[tuple[str, str]]:
    """Fetch English-labelled settlements from English-speaking countries."""
    values = " ".join(f"wd:{c}" for c in countries)
    query = f"""
    SELECT DISTINCT ?item ?label WHERE {{
      VALUES ?country {{ {values} }}
      ?item wdt:P31 wd:Q486972 .
      ?item wdt:P17 ?country .
      ?item rdfs:label ?label .
      FILTER(LANG(?label) = "{lang}")
    }}
    LIMIT {limit}
    """
    rows = sparql(query)
    return [
        (b["item"]["value"].rsplit("/", 1)[1], b["label"]["value"])
        for b in rows
    ]


# ── Ollama ────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama."""
    payload = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["embeddings"]


def embed_all(texts: list[str], label: str) -> np.ndarray:
    """Embed every text, batching for Ollama. Returns (N, 1024) float32."""
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        all_vecs.extend(embed_batch(chunk))
        done = min(i + BATCH_SIZE, len(texts))
        print(f"  [{label}] embedded {done}/{len(texts)}", flush=True)
    return np.asarray(all_vecs, dtype=np.float32)


# ── Collision stats ───────────────────────────────────────────────────────

def is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def collision_stats(
    vecs: np.ndarray,
    qids: list[str],
    labels: list[str],
    name: str,
    threshold: float = COLLISION_THRESHOLD,
) -> dict:
    """Compute cross-entity collision statistics for one group.

    A "collision" is a pair (i, j) with i < j, qids[i] != qids[j], and
    cosine similarity >= threshold. Distinct QIDs is the important
    constraint — we do NOT count a label aliased to itself.
    """
    n = len(qids)
    if n < 2:
        return {
            "group": name,
            "n_entities": n,
            "total_cross_entity_pairs": 0,
            "collision_count": 0,
            "collision_rate": 0.0,
            "mean_cosine": 0.0,
            "top_collisions": [],
        }

    # Normalise for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.clip(norms, 1e-9, None)

    # Full similarity matrix (n up to ~1000, so 1000x1000 float32 = 4 MB, fine)
    sim = unit @ unit.T

    iu, ju = np.triu_indices(n, k=1)
    pair_sims = sim[iu, ju]

    # Mask out same-QID pairs (aliases for one entity)
    qid_arr = np.asarray(qids)
    cross_mask = qid_arr[iu] != qid_arr[ju]
    cross_sims = pair_sims[cross_mask]

    total_cross = int(cross_mask.sum())
    collision_mask = cross_sims >= threshold
    collision_count = int(collision_mask.sum())

    mean_cosine = float(cross_sims.mean()) if total_cross else 0.0

    # Pick up to 10 example colliding pairs for the log
    top: list[dict] = []
    if collision_count:
        idx_hits = np.where(collision_mask)[0]
        # Get the top 10 by highest similarity
        order = np.argsort(-cross_sims[idx_hits])[:10]
        cross_iu = iu[cross_mask]
        cross_ju = ju[cross_mask]
        for k in order:
            ii = int(cross_iu[idx_hits[k]])
            jj = int(cross_ju[idx_hits[k]])
            top.append(
                {
                    "qid_a": qids[ii],
                    "label_a": labels[ii],
                    "qid_b": qids[jj],
                    "label_b": labels[jj],
                    "cosine": round(float(sim[ii, jj]), 4),
                }
            )

    return {
        "group": name,
        "n_entities": n,
        "total_cross_entity_pairs": total_cross,
        "collision_count": collision_count,
        "collision_rate": round(collision_count / total_cross, 4) if total_cross else 0.0,
        "mean_cosine": round(mean_cosine, 4),
        "top_collisions": top,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Wikidata Collision Scan — mxbai-embed-large [UNK] defect")
    print("=" * 72)

    # 1. Pull diacritical sample
    print("\n[1/4] Fetching diacritical sample from Wikidata SPARQL")
    diacritical: list[tuple[str, str]] = []
    for desc, country, lang in DIACRITIC_QUERIES:
        print(f"  {desc}…", end=" ", flush=True)
        try:
            rows = fetch_settlements_from_country(country, lang, PER_QUERY_LIMIT)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue
        # Only keep rows whose label actually contains non-ASCII
        rows = [(q, l) for q, l in rows if not is_ascii(l)]
        diacritical.extend(rows)
        print(f"{len(rows)} non-ASCII labels")
        time.sleep(1.0)  # polite to the query service

    print(f"\n  → {len(diacritical)} diacritical entities collected")

    # 2. Pull ASCII control
    print("\n[2/4] Fetching ASCII control from Wikidata SPARQL")
    desc, countries, lang = ASCII_CONTROL
    print(f"  {desc}…", end=" ", flush=True)
    try:
        raw_control = fetch_ascii_control(countries, lang, CONTROL_FETCH)
    except Exception as exc:
        print(f"FAILED ({exc})")
        raw_control = []
    control = [(q, l) for q, l in raw_control if is_ascii(l)]
    print(f"{len(control)} ASCII-only labels (filtered from {len(raw_control)})")

    if len(diacritical) < 200:
        print(f"\nFAIL: only {len(diacritical)} diacritical entities collected; need ≥200")
        sys.exit(1)
    if len(control) < 50:
        print(f"\nFAIL: only {len(control)} control entities collected; need ≥50")
        sys.exit(1)

    # 3. Embed both groups
    print("\n[3/4] Embedding labels via Ollama + mxbai-embed-large")

    d_qids = [q for q, _ in diacritical]
    d_labels = [l for _, l in diacritical]
    d_vecs = embed_all(d_labels, "diacritical")

    c_qids = [q for q, _ in control]
    c_labels = [l for _, l in control]
    c_vecs = embed_all(c_labels, "control")

    # 4. Collision stats
    print("\n[4/4] Computing pairwise collision statistics")
    d_stats = collision_stats(d_vecs, d_qids, d_labels, "diacritical")
    c_stats = collision_stats(c_vecs, c_qids, c_labels, "ascii_control")

    # Cross-group: every diacritical vs every ASCII control. Different
    # QID sets so all pairs are cross-entity by construction.
    print("  cross-group (diacritical × ASCII control)…")
    d_norm = d_vecs / np.clip(np.linalg.norm(d_vecs, axis=1, keepdims=True), 1e-9, None)
    c_norm = c_vecs / np.clip(np.linalg.norm(c_vecs, axis=1, keepdims=True), 1e-9, None)
    cross_sim = d_norm @ c_norm.T
    cross_total = cross_sim.size
    cross_hits = int((cross_sim >= COLLISION_THRESHOLD).sum())
    cross_mean = float(cross_sim.mean())

    summary = {
        "model": MODEL,
        "threshold": COLLISION_THRESHOLD,
        "diacritical": d_stats,
        "ascii_control": c_stats,
        "cross_group": {
            "pairs": cross_total,
            "collisions": cross_hits,
            "collision_rate": round(cross_hits / cross_total, 4) if cross_total else 0.0,
            "mean_cosine": round(cross_mean, 4),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── Report ────────────────────────────────────────────────────────────
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"\nDiacritical group  ({d_stats['n_entities']} entities)")
    print(f"  cross-entity pairs:  {d_stats['total_cross_entity_pairs']:,}")
    print(f"  collisions ≥ {COLLISION_THRESHOLD}: {d_stats['collision_count']:,}")
    print(f"  collision rate:      {pct(d_stats['collision_rate'])}")
    print(f"  mean pairwise cos:   {d_stats['mean_cosine']:.3f}")

    print(f"\nASCII control group ({c_stats['n_entities']} entities)")
    print(f"  cross-entity pairs:  {c_stats['total_cross_entity_pairs']:,}")
    print(f"  collisions ≥ {COLLISION_THRESHOLD}: {c_stats['collision_count']:,}")
    print(f"  collision rate:      {pct(c_stats['collision_rate'])}")
    print(f"  mean pairwise cos:   {c_stats['mean_cosine']:.3f}")

    print(f"\nCross-group (diacritical × control)")
    print(f"  pairs:               {cross_total:,}")
    print(f"  collisions ≥ {COLLISION_THRESHOLD}: {cross_hits:,}")
    print(f"  mean pairwise cos:   {cross_mean:.3f}")

    if d_stats["top_collisions"]:
        print("\nExample colliding pairs (diacritical group, top by cosine):")
        for hit in d_stats["top_collisions"][:10]:
            print(
                f"  {hit['cosine']:.4f}  "
                f"{hit['qid_a']:>10} {hit['label_a'][:30]:<32}"
                f"{hit['qid_b']:>10} {hit['label_b'][:30]}"
            )

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {SUMMARY_PATH}")

    # ── CI assertions ─────────────────────────────────────────────────────
    # The defect is "present" if diacritical collisions greatly exceed the
    # ASCII-control collision rate. If this ever stops being true on a daily
    # run, something has changed — either the model was fixed or Ollama
    # swapped in a differently-tokenised build.
    print("\n" + "=" * 72)
    if d_stats["collision_rate"] < 0.05:
        print(
            f"FAIL: diacritical collision rate {pct(d_stats['collision_rate'])} "
            f"is implausibly low. The mxbai [UNK] defect may have been fixed."
        )
        sys.exit(1)
    if c_stats["collision_rate"] > 0.02:
        print(
            f"FAIL: ASCII control collision rate {pct(c_stats['collision_rate'])} "
            f"is suspiciously high. Something other than the [UNK] defect "
            f"is firing."
        )
        sys.exit(1)

    # Ratio report — use raw (unrounded) counts so the figure is meaningful
    # when the control collision count is tiny. Fall back to an explicit
    # "vs <n> controls" description when the control has zero collisions.
    d_n = d_stats["collision_count"]
    c_n = c_stats["collision_count"]
    d_pairs = d_stats["total_cross_entity_pairs"] or 1
    c_pairs = c_stats["total_cross_entity_pairs"] or 1
    d_raw_rate = d_n / d_pairs
    c_raw_rate = c_n / c_pairs
    if c_n == 0:
        comparison = f"vs {c_n} collisions in {c_pairs:,} ASCII-control pairs"
    else:
        comparison = (
            f"vs {pct(c_raw_rate)} of ASCII controls "
            f"(ratio {d_raw_rate / c_raw_rate:.0f}×)"
        )
    print(
        f"✓ Defect reproduced: "
        f"{d_n:,} collisions in {d_pairs:,} diacritical pairs ({pct(d_raw_rate)}) "
        f"{comparison}"
    )


if __name__ == "__main__":
    main()
