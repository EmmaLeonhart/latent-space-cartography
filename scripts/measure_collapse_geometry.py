"""
Measure the geometry of the collision collapse zone, vectorized.

Port of old/scripts/measure_collapse_geometry.py rewritten with batched
matrix products so it runs on 100k+ embeddings (the old per-element
Python loops were O(n^2) interpreter-bound and infeasible at this scale).

Two passes over the (n x n) cosine matrix in row batches:
  pass 1 - per-row cross-entity collision flag (cosine >= threshold with a
           row of a different QID)
  pass 2 - per-row mean k-NN cosine distance, and for colliding rows the
           distance to the nearest NON-colliding row

Outputs data/collapse_geometry.json with the statistics used in paper
section 5.4.1: colliding/non-colliding counts, density means and ratio,
quartile occupancy, and isolation (nearest non-colliding neighbor) stats.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'buffer'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = Path(os.environ.get("FOL_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
THRESHOLD = 0.95
K = 10
BATCH = 2048


def main():
    emb = np.load(str(DATA_DIR / 'embeddings.npz'))['vectors'].astype(np.float32)
    with open(str(DATA_DIR / 'embedding_index.json'), encoding='utf-8') as f:
        index = json.load(f)
    n = emb.shape[0]
    assert n == len(index)
    print(f"Store: {n} x {emb.shape[1]}")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / (norms + 1e-10)

    qids = [row['qid'] for row in index]
    uniq = {q: i for i, q in enumerate(dict.fromkeys(qids))}
    qid_codes = np.array([uniq[q] for q in qids], dtype=np.int32)

    # Pass 1: cross-entity collision flags
    colliding = np.zeros(n, dtype=bool)
    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        sims = normed[s:e] @ normed.T          # (b, n)
        hit = sims >= THRESHOLD
        hit[np.arange(e - s), np.arange(s, e)] = False
        same_qid = qid_codes[s:e, None] == qid_codes[None, :]
        cross = hit & ~same_qid
        colliding[s:e] |= cross.any(axis=1)
        if (s // BATCH) % 10 == 0:
            print(f"  pass1 {e}/{n} ({time.time()-t0:.0f}s)", flush=True)

    n_col = int(colliding.sum())
    print(f"Colliding embeddings: {n_col}, non-colliding: {n - n_col}")

    # Pass 2: k-NN density + nearest non-colliding neighbor
    knn_dist = np.zeros(n, dtype=np.float32)
    nearest_noncol = np.zeros(n, dtype=np.float32)
    noncol_mask = ~colliding
    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        sims = normed[s:e] @ normed.T
        sims[np.arange(e - s), np.arange(s, e)] = -np.inf
        top = np.partition(sims, n - K, axis=1)[:, n - K:]
        knn_dist[s:e] = 1.0 - top.mean(axis=1)
        sims_noncol = np.where(noncol_mask[None, :], sims, -np.inf)
        nearest_noncol[s:e] = 1.0 - sims_noncol.max(axis=1)
        if (s // BATCH) % 10 == 0:
            print(f"  pass2 {e}/{n} ({time.time()-t0:.0f}s)", flush=True)

    col_d = knn_dist[colliding]
    non_d = knn_dist[noncol_mask]
    q25, q75 = np.percentile(knn_dist, [25, 75])
    densest_frac = float((col_d <= q25).mean())
    sparsest_frac = float((col_d >= q75).mean())

    out = {
        "threshold": THRESHOLD,
        "k": K,
        "n_embeddings": n,
        "n_colliding": n_col,
        "n_noncolliding": n - n_col,
        "mean_knn_dist_colliding": float(col_d.mean()),
        "mean_knn_dist_noncolliding": float(non_d.mean()),
        "density_ratio_col_over_noncol": float(col_d.mean() / non_d.mean()),
        "frac_colliding_in_densest_quartile": densest_frac,
        "frac_colliding_in_sparsest_quartile": sparsest_frac,
        "mean_nearest_noncol_from_colliding": float(nearest_noncol[colliding].mean()),
        "mean_nearest_noncol_from_noncolliding": float(nearest_noncol[noncol_mask].mean()),
    }
    with open(str(DATA_DIR / 'collapse_geometry.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
