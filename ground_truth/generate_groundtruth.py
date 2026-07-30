#!/usr/bin/env python3
"""
generate_groundtruth.py

Reads base and query vectors (fvecs format) along with their scalar constraints (csv format).
Filters the base vectors that satisfy the query scalar constraints, then calculates
the exact L2 nearest neighbors to generate a ground truth file (ivecs format).

Uses FAISS for accelerated exact search with scalar pre-filtering.
Masks are computed per-query (not pre-materialised) to avoid OOM for large datasets
such as AudioSet unbal_train (~2M vectors × 10K queries = 20 GB if pre-allocated).

Usage example:
    python generate_groundtruth.py \\
        --base-vecs ../sift/sift_base.fvecs \\
        --base-scalars ../scalar/example_build_scalars.csv \\
        --query-vecs ../sift/sift_query.fvecs \\
        --query-ranges ../scalar/example_query_ranges.csv \\
        --k 100 \\
        --output gt.ivecs
"""

import sys
import os
import argparse
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not found, falling back to numpy search (slower).")


def read_fvecs(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find fvecs file: {filename}")

    print(f"Reading vectors from {filename}...")
    with open(filename, 'rb') as f:
        d_out = np.fromfile(f, dtype=np.int32, count=1)
        if len(d_out) == 0:
            return np.empty((0, 0), dtype=np.float32)
        dim = d_out[0]
        f.seek(0)
        dt = np.dtype([('d', np.int32), ('vecs', np.float32, (dim,))])
        data = np.fromfile(f, dtype=dt)
        return data['vecs']


def write_ivecs(filename, data):
    print(f"Writing ivecs to {filename}...")
    n, dim = data.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = data
    with open(filename, 'wb') as f:
        out.tofile(f)


def load_scalars(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find csv file: {filename}")

    print(f"Reading scalars from {filename}...")
    return np.loadtxt(filename, delimiter=',', ndmin=2)


def search_faiss(index, query_vecs, base_scalars, query_ranges, k):
    """
    Exact L2 search with per-query IDSelectorArray.
    Mask is computed inline for each query — O(n_base) memory, not O(n_q × n_base).
    """
    n_queries = query_vecs.shape[0]
    n_dims    = query_ranges.shape[1] // 2
    n_base    = base_scalars.shape[0]
    ground_truth = np.full((n_queries, k), -1, dtype=np.int32)

    for i in range(n_queries):
        if i % max(1, n_queries // 10) == 0:
            print(f"  Processed {i}/{n_queries} queries")

        mins = query_ranges[i, 0::2]
        maxs = query_ranges[i, 1::2]

        mask = np.ones(n_base, dtype=bool)
        for d in range(n_dims):
            mask &= base_scalars[:, d] >= mins[d]
            mask &= base_scalars[:, d] <= maxs[d]

        valid = np.where(mask)[0].astype(np.int64)
        n_valid = len(valid)
        if n_valid == 0:
            continue

        ki  = min(k, n_valid)
        sel = faiss.IDSelectorArray(valid)
        params = faiss.SearchParametersIVF()
        params.sel = sel

        q = query_vecs[i:i+1]          # shape (1, D) — already float32
        _, I = index.search(q, ki, params=params)
        ground_truth[i, :ki] = I[0, :ki]

    return ground_truth


def search_numpy(base_vecs, query_vecs, base_scalars, query_ranges, k):
    """Fallback exact search using numpy (no FAISS). Same per-query mask strategy."""
    n_queries = query_vecs.shape[0]
    n_dims    = query_ranges.shape[1] // 2
    n_base    = base_scalars.shape[0]
    ground_truth = np.full((n_queries, k), -1, dtype=np.int32)

    for i in range(n_queries):
        if i % max(1, n_queries // 10) == 0:
            print(f"  Processed {i}/{n_queries} queries")

        mins = query_ranges[i, 0::2]
        maxs = query_ranges[i, 1::2]

        mask = np.ones(n_base, dtype=bool)
        for d in range(n_dims):
            mask &= base_scalars[:, d] >= mins[d]
            mask &= base_scalars[:, d] <= maxs[d]

        valid = np.where(mask)[0]
        n_valid = len(valid)
        if n_valid == 0:
            continue

        diff  = base_vecs[valid] - query_vecs[i]
        dists = np.einsum('ij,ij->i', diff, diff)
        ki    = min(k, n_valid)

        if n_valid <= k:
            order = np.argsort(dists)
        else:
            order = np.argpartition(dists, ki - 1)[:ki]
            order = order[np.argsort(dists[order])]

        ground_truth[i, :ki] = valid[order[:ki]]

    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Generate exact ground truth with scalar range filtering (FAISS-accelerated).")
    parser.add_argument("--base-vecs",    required=True,  help="Path to base vectors (.fvecs)")
    parser.add_argument("--base-scalars", required=True,  help="Path to base scalars (.csv)")
    parser.add_argument("--query-vecs",   required=True,  help="Path to query vectors (.fvecs)")
    parser.add_argument("--query-ranges", required=True,  help="Path to query ranges (.csv)")
    parser.add_argument("--k",    type=int, default=100,  help="Number of nearest neighbors")
    parser.add_argument("--output", "-o",  required=True, help="Output ground truth (.ivecs)")
    parser.add_argument("--n-base", type=int, default=None,
                        help="Use only the first N base vectors.")
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2",
                        help="Distance metric: l2 (default) or cosine (L2-normalise vectors).")
    args = parser.parse_args()

    base_vecs    = read_fvecs(args.base_vecs)
    query_vecs   = read_fvecs(args.query_vecs)
    base_scalars = load_scalars(args.base_scalars)
    query_ranges = load_scalars(args.query_ranges)

    if args.n_base is not None and args.n_base < base_vecs.shape[0]:
        base_vecs    = base_vecs[:args.n_base]
        base_scalars = base_scalars[:args.n_base]
        print(f"Truncated base to first {args.n_base} rows.")

    print(f"Base: {base_vecs.shape[0]} vecs  |  Queries: {query_vecs.shape[0]} vecs")

    if args.metric == "cosine":
        print("Normalizing vectors to unit L2 norm (cosine metric)...")
        norms = np.linalg.norm(base_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        base_vecs = base_vecs / norms
        norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_vecs = query_vecs / norms

    if base_vecs.shape[0] != base_scalars.shape[0]:
        sys.exit(f"ERROR: base vecs/scalars length mismatch: "
                 f"{base_vecs.shape[0]} vs {base_scalars.shape[0]}")
    if query_vecs.shape[0] != query_ranges.shape[0]:
        sys.exit(f"ERROR: query vecs/ranges length mismatch: "
                 f"{query_vecs.shape[0]} vs {query_ranges.shape[0]}")

    k = args.k
    print(f"Running exact KNN search (k={k})...")

    if FAISS_AVAILABLE:
        base_f32 = np.ascontiguousarray(base_vecs, dtype=np.float32)
        dim      = base_f32.shape[1]
        if args.metric == "cosine":
            index = faiss.IndexFlatIP(dim)
            index_label = "IndexFlatIP"
        else:
            index = faiss.IndexFlatL2(dim)
            index_label = "IndexFlatL2"
        index.add(base_f32)
        print(f"  FAISS {index_label} built ({index.ntotal} vectors, dim={dim})")
        ground_truth = search_faiss(index, query_vecs, base_scalars, query_ranges, k)
    else:
        ground_truth = search_numpy(base_vecs, query_vecs, base_scalars, query_ranges, k)

    print(f"  Processed {query_vecs.shape[0]}/{query_vecs.shape[0]} queries")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_ivecs(args.output, ground_truth)
    print("Done.")


if __name__ == "__main__":
    main()
