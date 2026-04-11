"""
Verify the HF-vs-Ollama tokenizer divergence for mxbai-embed-large (and
other BERT-derived embedding models registered in local Ollama).

The claim the earlier page-text made — that mxbai-embed-large produces
[CLS] [UNK] [SEP] for any diacritical word via its WordPiece tokenizer —
does NOT reproduce when you run the upstream HuggingFace tokenizer
directly. HF's BertTokenizer applies BasicTokenizer's accent-stripping
preprocessing (because do_lower_case=True implies strip_accents), so
"Hokkaidō" normalizes to "hokkaido" and tokenizes cleanly.

But Emma's empirical collisions via Ollama ARE real. This script proves
the divergence side-by-side:

  1. Upstream HF tokenizer: run AutoTokenizer on the same test strings;
     show that diacritical forms and ASCII forms produce IDENTICAL
     token IDs (accent-stripping active).
  2. Ollama HTTP API: embed the same strings via the ARCHIVED gguf
     (registered under the name `mxbai-archived` from model/Modelfile,
     not pulled live from the registry); show that diacritical forms
     and ASCII forms produce DIFFERENT embeddings (cosine < 0.5), and
     that different diacritical forms produce IDENTICAL embeddings
     (cosine ≈ 1.0) — i.e., [UNK] collapse is live in Ollama's path.
  3. Control: also run the same test on `nomic-embed-text` and
     `all-minilm` if they are registered, to see whether the defect
     class is Ollama-wide for BERT-derived embedding models or
     specific to mxbai.

Output: `verification/tokenizer_divergence.json` at the repo root,
containing the full HF tokenization, Ollama embedding similarities,
and a verdict flag per model.

This script and its output are consumed by `scripts/generate_defect_page.py`
so the public page always reflects the most recent reproducible state
of the divergence.

Usage:
  python scripts/verify_tokenizer_divergence.py

Prereqs:
  1. `ollama serve` running locally (default port 11434)
  2. Archived model registered:
       cd model/
       ollama create mxbai-archived -f Modelfile
  3. `pip install transformers` for upstream HF tokenizer comparison
"""

import io
import json
import math
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
VERIFICATION_DIR = ROOT / "verification"
JSON_PATH = VERIFICATION_DIR / "tokenizer_divergence.json"
GGUF_PATH = ROOT / "model" / "mxbai-embed-large-v1.gguf"

OLLAMA_URL = "http://localhost:11434"
EMBED_ENDPOINT = f"{OLLAMA_URL}/api/embed"
SHOW_ENDPOINT = f"{OLLAMA_URL}/api/show"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"

# The primary model under test: the archived gguf registered via
# model/Modelfile, NOT mxbai-embed-large:latest from the registry.
PRIMARY_MODEL = "mxbai-archived"

# Secondary models — BERT-derived embedding models available via Ollama.
# If registered, they get tested with the same divergence probe so we
# can see whether the defect class is Ollama-wide.
SECONDARY_MODELS = ["nomic-embed-text", "all-minilm"]

# HuggingFace model id for upstream tokenizer comparison.
HF_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"

# Test pairs: (diacritical form, ASCII form). Same meaning, different
# encoding. A working tokenizer should produce either identical tokens
# (accent-stripping path) or different-but-related tokens that produce
# high cosine similarity. mxbai-via-Ollama produces neither — the
# diacritical forms all collapse to a single UNK vector.
TEST_PAIRS = [
    ("Hokkaidō", "Hokkaido"),
    ("Éire", "Eire"),
    ("Zürich", "Zurich"),
    ("café", "cafe"),
    ("Dvořák", "Dvorak"),
    ("naïve", "naive"),
    ("São Paulo", "Sao Paulo"),
    ("Malmö", "Malmo"),
    ("Gdańsk", "Gdansk"),
    ("Łódź", "Lodz"),
]

# English ASCII controls — a working embedding model should produce
# low cosine similarity between unrelated English words, typically 0.3-0.5.
CONTROL_WORDS = [
    "Berlin", "quantum physics", "cat", "democracy", "bicycle",
]


# ── Ollama helpers ───────────────────────────────────────────────────

def ollama_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def ollama_post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def list_ollama_models():
    """Return a list of model name strings currently registered with Ollama."""
    try:
        tags = ollama_get(TAGS_ENDPOINT)
        return [m["name"].split(":")[0] for m in tags.get("models", [])]
    except Exception as e:
        print(f"  [error] Could not reach Ollama at {TAGS_ENDPOINT}: {e}")
        return []


def ollama_show(model_name):
    """Fetch model details (digest, family, parameters) for traceability."""
    try:
        return ollama_post(SHOW_ENDPOINT, {"name": model_name})
    except Exception as e:
        return {"error": str(e)}


def ollama_embed(model_name, texts):
    """Embed a batch of texts via Ollama HTTP API. Returns list of vectors."""
    resp = ollama_post(EMBED_ENDPOINT, {"model": model_name, "input": texts})
    return resp["embeddings"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── HuggingFace upstream tokenizer probe ─────────────────────────────

def run_hf_tokenizer_probe():
    """Run the upstream HF BertTokenizer on the test strings.

    Returns a dict with per-string tokens and ids, and a flag showing
    whether accent-stripping is active (i.e. diacritical and ASCII
    forms produce identical token IDs).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {
            "status": "skipped",
            "reason": (
                "transformers not installed. "
                "Run: pip install transformers"
            ),
        }

    try:
        tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    bt = getattr(tok, "basic_tokenizer", None)
    config = {
        "tokenizer_class": type(tok).__name__,
        "do_lower_case": getattr(tok, "do_lower_case", None),
        "strip_accents_attr": getattr(tok, "strip_accents", None),
        "basic_tokenizer_do_lower_case": getattr(bt, "do_lower_case", None) if bt else None,
        "basic_tokenizer_strip_accents": getattr(bt, "strip_accents", None) if bt else None,
    }

    pair_results = []
    for diacritic, ascii_form in TEST_PAIRS:
        d_tokens = tok.tokenize(diacritic)
        a_tokens = tok.tokenize(ascii_form)
        d_ids = tok.convert_tokens_to_ids(d_tokens)
        a_ids = tok.convert_tokens_to_ids(a_tokens)
        pair_results.append({
            "diacritic": diacritic,
            "ascii": ascii_form,
            "diacritic_tokens": d_tokens,
            "ascii_tokens": a_tokens,
            "diacritic_ids": d_ids,
            "ascii_ids": a_ids,
            "ids_identical": d_ids == a_ids,
        })

    all_identical = all(p["ids_identical"] for p in pair_results)
    return {
        "status": "ok",
        "hf_model_id": HF_MODEL_ID,
        "config": config,
        "pairs": pair_results,
        "accent_stripping_active": all_identical,
    }


# ── Ollama embedding divergence probe ────────────────────────────────

def run_ollama_probe(model_name):
    """Embed all test pairs + controls via Ollama. Returns divergence stats."""
    show = ollama_show(model_name)
    if "error" in show:
        return {
            "status": "error",
            "model": model_name,
            "reason": show["error"],
        }

    diacritic_words = [d for d, _ in TEST_PAIRS]
    ascii_words = [a for _, a in TEST_PAIRS]
    all_texts = diacritic_words + ascii_words + CONTROL_WORDS

    try:
        vecs = ollama_embed(model_name, all_texts)
    except Exception as e:
        return {
            "status": "error",
            "model": model_name,
            "reason": f"embed call failed: {e}",
        }

    if len(vecs) != len(all_texts):
        return {
            "status": "error",
            "model": model_name,
            "reason": f"got {len(vecs)} vectors for {len(all_texts)} inputs",
        }

    word2vec = dict(zip(all_texts, vecs))

    # For each pair: cosine(diacritic, ascii) — should be ≈1.0 if the
    # tokenizer path strips accents or tokenizes cleanly; will be much
    # lower if the diacritic collapses to UNK.
    same_word_sims = []
    for diacritic, ascii_form in TEST_PAIRS:
        s = cosine(word2vec[diacritic], word2vec[ascii_form])
        same_word_sims.append({
            "diacritic": diacritic,
            "ascii": ascii_form,
            "cosine": s,
        })

    # Cross-pair diacritic-vs-diacritic: should be low (≈0.5) if the
    # model is doing real work. If the UNK cluster is live, this spikes
    # to ≈1.0 for unrelated words.
    cross_diacritic_sims = []
    for i, (d1, _) in enumerate(TEST_PAIRS):
        for j, (d2, _) in enumerate(TEST_PAIRS):
            if i < j:
                cross_diacritic_sims.append({
                    "a": d1,
                    "b": d2,
                    "cosine": cosine(word2vec[d1], word2vec[d2]),
                })

    # Control-vs-control: normal ASCII English baseline for calibration.
    cross_control_sims = []
    for i, c1 in enumerate(CONTROL_WORDS):
        for j, c2 in enumerate(CONTROL_WORDS):
            if i < j:
                cross_control_sims.append({
                    "a": c1,
                    "b": c2,
                    "cosine": cosine(word2vec[c1], word2vec[c2]),
                })

    # Summary statistics
    def stats(xs):
        if not xs:
            return {"n": 0}
        vals = [x["cosine"] for x in xs]
        vals_sorted = sorted(vals)
        n = len(vals)
        return {
            "n": n,
            "mean": sum(vals) / n,
            "min": min(vals),
            "max": max(vals),
            "median": vals_sorted[n // 2],
        }

    same_stats = stats(same_word_sims)
    cross_d_stats = stats(cross_diacritic_sims)
    cross_c_stats = stats(cross_control_sims)

    # Classify the failure mode rather than emitting a binary flag.
    # The diagnostic quantities:
    #   S  = mean cosine(diacritic_i, ascii_i)  — "is the same word recognized?"
    #   D  = mean cosine(diacritic_i, diacritic_j≠i) — "do unrelated diacritic words collapse?"
    #   C  = mean cosine(control_i, control_j≠i) — "normal ASCII baseline for calibration"
    #
    # A well-behaved model should have S high (≈1.0) and D near C.
    # Failure modes we care about:
    #   - "unk_collapse": D > C + 0.3 AND S ≈ C — diacritical forms
    #     cluster into a single attractor, and the ASCII equivalent is
    #     treated as unrelated to them. This is mxbai-archived's mode.
    #   - "diacritic_attractor": D > C + 0.3 AND S is decent — the model
    #     recognizes the ASCII equivalent but unrelated diacritical words
    #     still cluster tightly together. This is nomic-embed-text's mode.
    #   - "ascii_equivalence_broken": S < C + 0.1 — the model fails to
    #     recognize a word's own ASCII form (this alone is already a bug;
    #     all-minilm exhibits this).
    S = same_stats.get("mean", 1.0)
    D = cross_d_stats.get("mean", 0.0)
    C = cross_c_stats.get("mean", 0.5)

    failure_modes = []
    if D - C > 0.3:
        if S <= C + 0.1:
            failure_modes.append("unk_collapse")
        else:
            failure_modes.append("diacritic_attractor")
    if S <= C + 0.1:
        if "unk_collapse" not in failure_modes:
            failure_modes.append("ascii_equivalence_broken")

    # Severity score: distance above the control baseline that unrelated
    # diacritical words cluster to, penalized by how far the same-word
    # similarity falls below a healthy threshold.
    severity = max(0.0, D - C) + max(0.0, 0.9 - S)

    divergence_active = len(failure_modes) > 0

    # Strip digest info from the show response for traceability.
    # Different Ollama versions structure this differently; grab whatever
    # identifying fields are available without assuming a schema.
    details = show.get("details", {})
    model_info = {
        "family": details.get("family"),
        "families": details.get("families"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "format": details.get("format"),
    }

    return {
        "status": "ok",
        "model": model_name,
        "model_info": model_info,
        "same_word_sims": same_word_sims,
        "cross_diacritic_sims": cross_diacritic_sims,
        "cross_control_sims": cross_control_sims,
        "stats": {
            "diacritic_vs_ascii_same_word": same_stats,
            "cross_diacritic": cross_d_stats,
            "cross_control": cross_c_stats,
        },
        "divergence_active": divergence_active,
        "failure_modes": failure_modes,
        "severity": severity,
    }


# ── Pretty printing ──────────────────────────────────────────────────

def print_probe_summary(probe, indent=""):
    """Human-readable summary of a single Ollama probe result."""
    s = probe["stats"]
    same = s["diacritic_vs_ascii_same_word"]["mean"]
    crossd = s["cross_diacritic"]["mean"]
    crossc = s["cross_control"]["mean"]
    severity = probe.get("severity", 0.0)
    modes = probe.get("failure_modes", [])
    name = probe.get("model", "?")
    print(f"{indent}{name}")
    print(f"{indent}  S (diacritic↔ASCII same word): mean={same:.3f}")
    print(f"{indent}  D (diacritic↔diacritic diff):  mean={crossd:.3f}")
    print(f"{indent}  C (control ASCII baseline):    mean={crossc:.3f}")
    if modes:
        print(f"{indent}  failure modes: {', '.join(modes)}   severity={severity:.3f}")
    else:
        print(f"{indent}  failure modes: none detected  severity={severity:.3f}")


# ── Orchestration ────────────────────────────────────────────────────

def preflight():
    """Verify the archived gguf and Ollama model are in place."""
    errors = []

    if not GGUF_PATH.exists():
        errors.append(
            f"  Archived gguf not found at {GGUF_PATH}. "
            "This repo ships the frozen weights — clone with LFS or "
            "restore from the model/ directory."
        )

    available = list_ollama_models()
    if not available:
        errors.append(
            "  Ollama is not running or is not reachable at "
            f"{OLLAMA_URL}. Start it with: ollama serve"
        )
    elif PRIMARY_MODEL not in available:
        errors.append(
            f"  Primary model '{PRIMARY_MODEL}' is not registered in "
            "Ollama. Register the archived gguf with:\n"
            "      cd model/\n"
            f"      ollama create {PRIMARY_MODEL} -f Modelfile"
        )

    if errors:
        print("Preflight failed:")
        for e in errors:
            print(e)
        sys.exit(1)

    return available


def main():
    print("=" * 68)
    print("Tokenizer divergence verification")
    print("  HF upstream vs Ollama (archived gguf)")
    print("=" * 68)

    available = preflight()

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── HF probe ──
    print()
    print(f"[1/3] Running upstream HF tokenizer probe on {HF_MODEL_ID}...")
    hf = run_hf_tokenizer_probe()
    if hf.get("status") == "ok":
        n_total = len(hf["pairs"])
        n_stripped = sum(1 for p in hf["pairs"] if p["ids_identical"])
        print(f"  HF BertTokenizer strips accents for {n_stripped}/{n_total} pairs")
        print(f"    (do_lower_case={hf['config']['do_lower_case']}, "
              f"strip_accents={hf['config']['basic_tokenizer_strip_accents']})")
        for p in hf["pairs"]:
            marker = "==" if p["ids_identical"] else "!="
            print(f"    {p['diacritic']:14s} {marker} {p['ascii']:14s}  "
                  f"{p['diacritic_tokens']} / {p['ascii_tokens']}")
        if n_stripped < n_total:
            non_stripped = [p["diacritic"] for p in hf["pairs"] if not p["ids_identical"]]
            print(f"    (non-decomposable characters not handled by NFD: "
                  f"{', '.join(non_stripped)})")
    else:
        print(f"  HF probe {hf.get('status')}: {hf.get('reason')}")

    # ── Primary Ollama probe (archived mxbai) ──
    print()
    print(f"[2/3] Running Ollama probe on PRIMARY model: {PRIMARY_MODEL}")
    print(f"      (built from model/mxbai-embed-large-v1.gguf via model/Modelfile)")
    primary = run_ollama_probe(PRIMARY_MODEL)
    if primary.get("status") == "ok":
        print_probe_summary(primary)
    else:
        print(f"  Probe failed: {primary.get('reason')}")

    # ── Secondary probes (other Ollama-registered BERT embedding models) ──
    print()
    print("[3/3] Running Ollama probes on secondary BERT-derived models")
    print("      (scope check: is the defect class Ollama-wide?)")
    secondary_results = {}
    for name in SECONDARY_MODELS:
        if name not in available:
            print(f"  {name}: not registered, skipping")
            secondary_results[name] = {"status": "not_registered"}
            continue
        print(f"  Probing {name}...")
        r = run_ollama_probe(name)
        secondary_results[name] = r
        if r.get("status") == "ok":
            print_probe_summary(r, indent="    ")
        else:
            print(f"    failed: {r.get('reason')}")

    # ── Write JSON artifact ──
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_model": PRIMARY_MODEL,
        "gguf_path": str(GGUF_PATH.relative_to(ROOT)),
        "hf_probe": hf,
        "ollama_primary": primary,
        "ollama_secondary": secondary_results,
    }
    JSON_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print()
    print(f"Wrote {JSON_PATH.relative_to(ROOT)}")
    print()
    print("Next step: regenerate the page with corrected mechanism by running:")
    print("  python scripts/generate_defect_page.py")


if __name__ == "__main__":
    main()
