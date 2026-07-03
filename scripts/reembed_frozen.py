"""
Re-embed a frozen entity/alias set with a different embedding model.

Reads an existing data store (items.json + embedding_index.json produced by
random_walk.py / import_wikidata.py) and embeds the *identical* text list with
another model, writing a parallel store that fol_discovery.py can consume via
FOL_DATA_DIR. This guarantees byte-identical input across models for the
cross-model comparison (Section 4.6 of the paper) — no re-crawl, no snapshot
drift.

Usage:
  python reembed_frozen.py --model nomic-embed-text
  python reembed_frozen.py --model all-minilm --source data --out data_all-minilm
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import ollama

if hasattr(sys.stdout, 'buffer'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
COPY_FILES = ["items.json", "embedding_index.json", "properties.json"]


def main():
    parser = argparse.ArgumentParser(description="Re-embed a frozen text set with another model")
    parser.add_argument("--model", required=True, help="Ollama embedding model name")
    parser.add_argument("--source", default=str(REPO_ROOT / "data"), help="Source data dir (frozen store)")
    parser.add_argument("--out", default=None, help="Output data dir (default: data_<model>)")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out) if args.out else REPO_ROOT / f"data_{args.model}"
    out.mkdir(parents=True, exist_ok=True)

    with open(source / "embedding_index.json", encoding="utf-8") as f:
        index = json.load(f)
    texts = [row["text"] for row in index]
    print(f"Frozen set: {len(texts)} texts from {source}")

    vectors = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        result = ollama.embed(model=args.model, input=batch)
        vectors.extend(result.embeddings)
        done = i + len(batch)
        if done % (BATCH_SIZE * 20) < BATCH_SIZE or done == len(texts):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(texts)} embedded ({rate:.0f}/s)", flush=True)

    emb = np.array(vectors)
    assert emb.shape[0] == len(index), f"row mismatch: {emb.shape[0]} vectors vs {len(index)} index rows"
    np.savez_compressed(str(out / "embeddings.npz"), vectors=emb)

    for name in COPY_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(str(src), str(out / name))

    print(f"Wrote {emb.shape[0]} x {emb.shape[1]} embeddings to {out / 'embeddings.npz'}")
    print(f"Run analysis with: FOL_DATA_DIR={out} python scripts/fol_discovery.py --min-triples 5")


if __name__ == "__main__":
    main()
