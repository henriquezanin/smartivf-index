# SmartIVF

SmartIVF is a range-filtered approximate nearest-neighbor index that combines an
equi-depth inverted file with a Z-order space-filling curve (SFC). Each object's
scalar attributes are mapped to a single SFC key; at query time a budget-controlled
BFS decomposes the multidimensional range predicate into a compact set of 1-D SFC
intervals, and only the partitions whose scalar bounds intersect those intervals are
probed in vector-proximity order. The number of scalar dimensions is not
structurally limited, and no auxiliary graph structure is required.

This is the C++26 implementation. It relies on:

- FAISS for the distance kernels (`fvec_inner_product`, `fvec_norm_L2sqr`) and for
  k-means++ clustering (`faiss::Clustering`).
- BMI2 `_pdep_u64` / `_pext_u64` in the Z-order encoder on Haswell+ / Zen 3+ CPUs,
  with a scalar fallback elsewhere.
- Precomputed `‖x‖²` per object, so the L2 path evaluates
  `‖q-x‖² = ‖q‖² + ‖x‖² − 2·(q·x)` rather than subtracting inside the SIMD loop.
- OpenMP parallelism over the query batch at search time (one thread per query).

## Reference

SmartIVF: Scalable Range-Filtered Approximate Nearest-Neighbor Search via
Space-Filling Curves. ADBIS 2026 (submitted).

## Repository layout

```text
├── CMakeLists.txt
├── include/smartivf/
│   ├── definitions.hpp           # Object, Partition, Index, SmartIVFConfig, Result<T>
│   ├── smartivf.hpp              # BuildIndex / Search facade
│   ├── range_decompose.hpp       # Algorithm 3
│   ├── build_partition.hpp       # Flat + k-means via FAISS
│   ├── compute_quantile.hpp      # Equi-depth split
│   ├── quantize.hpp              # Scalar quantisation
│   ├── store.hpp                 # Binary index file (magic + version + body)
│   ├── sfc/{sfc,z_order}.hpp     # Z-order Morton encoder (BMI2 / scalar fallback)
│   └── utils/{fvecs,ivecs,ranges,scalar}.hpp
├── src/                          # Same module structure, .cpp side
├── tests/                        # GoogleTest suite, one test_*.cpp per module
└── ground_truth/
    └── generate_groundtruth.py   # FAISS brute-force ground truth
```

## Requirements

| Component | Version | Notes |
|---|---|---|
| CMake | ≥ 3.28 | for FetchContent + C++26 |
| Compiler | GCC ≥ 14 or Clang ≥ 19 | with `-std=c++26` (`-std=c++2c`) |
| FAISS | ≥ 1.7.4 (CPU build) | `yay -S faiss-cpu` on Arch / `apt install libfaiss-dev` on Debian trixie+ |
| OpenMP | libgomp ≥ 14 / libomp ≥ 19 | shipped with the compiler |
| Python | ≥ 3.9 (ground truth only) | + `faiss-cpu` and `numpy` |

### Installing FAISS (CPU)

Arch Linux:
```bash
yay -S faiss-cpu
```

Debian / Ubuntu (recent):
```bash
sudo apt install libfaiss-dev
```

From source: clone https://github.com/facebookresearch/faiss, follow its
`INSTALL.md` (CPU-only build), and pass `-DCMAKE_PREFIX_PATH=<install-prefix>`
to this project's CMake invocation.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/smartivf --help
```

CMake options:

| CMake option | Default | Effect |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | `Release` | `RelWithDebInfo` keeps frame pointers + LTO |
| `-DSMARTIVF_USE_NATIVE=ON` | `ON` | compiles with `-march=native` (BMI2, AVX2/AVX-512) |
| `-DSMARTIVF_USE_LTO=ON` | `ON` | enables LTO in Release builds |
| `-DSMARTIVF_BUILD_TESTS=ON` | `ON` | builds the GoogleTest suite |
| `-DCMAKE_CXX_COMPILER=g++-14` | auto | force a specific compiler |

### Run the test suite

```bash
ctest --test-dir build --output-on-failure
```

## CLI reference

The binary exposes three subcommands: `build`, `search`, and `run`.

| Flag | Default | Effect |
|---|---|---|
| `--threads N` | `0` (= `omp_get_max_threads()`) | OpenMP thread count; `--threads 1` for single-threaded runs |
| `--seed N` | `42` | k-means RNG seed for reproducibility |

### `build` — construct and save the index

```text
./smartivf build \
  --base-vecs     <path.fvecs>
  --base-scalars  <path.csv>
  --index-out     <path.bin>
  --sfc           z-order
  --scalar-dims   <D>
  --scalar-bits   8
  --partitions    256
  --partition-threshold 2000
  --metric        l2          # or cosine
  --seed          42
  --threads       0
```

### `search` — query a pre-built index

```text
./smartivf search \
  --index-in      <path.bin>
  --query-vecs    <path.fvecs>
  --query-ranges  <path.csv>
  --ground-truth  <path.ivecs>   # optional, enables Recall@K reporting
  --k             10
  --sfc           z-order
  --scalar-dims   <D>            # must match build configuration
  --scalar-bits   8              # must match build configuration
  --partitions    256            # must match build configuration
  --partition-threshold 2000     # must match build configuration
  --metric        l2             # must match build configuration
  --sfc-budget    256
  --min-probe     1
  --max-probe     <P>            # sweep for the recall–QPS curve
  --candidate-target 1000
  --threads       0
```

### `run` — build and search in one shot (no index file)

```bash
./smartivf run \
  --base-vecs     <path.fvecs> \
  --base-scalars  <path.csv> \
  --query-vecs    <path.fvecs> \
  --query-ranges  <path.csv> \
  --ground-truth  <path.ivecs> \
  --scalar-dims   <D> \
  --metric        l2 \
  --k             10
```

## Reproducing the experiments

| Dataset | Vectors | Dim | Metric | `--scalar-dims` |
|---|---|---|---|---|
| SIFT-1M | 1 M | 128 | L2 | 1 – 4 |
| AudioSet | ~2 M | 128 | L2 | 1 – 4 |
| LAION-1M | 990 K (base) | 768 | cosine | 1 – 3 |

Common parameters across all datasets:

```text
--sfc z-order --scalar-bits 8 --partitions 256 --partition-threshold 2000
--sfc-budget 256 --min-probe 1 --candidate-target 1000 --k 10
```

The recall–QPS curve is obtained by sweeping `--max-probe` over
`1 2 3 5 10 25 50 100 150 200`.

Pass `--threads 1` for single-threaded measurements; omit it to use the full
thread count of the experiment machine.

### SIFT-1M (L2)

```bash
# Build
./smartivf build \
  --base-vecs sift_base.fvecs \
  --base-scalars sift_scalars.csv \
  --index-out sift_index.bin \
  --metric l2 --scalar-dims 1 \
  --sfc z-order --scalar-bits 8 --partitions 256 --partition-threshold 2000

# Search (example: max-probe = 50)
./smartivf search \
  --index-in sift_index.bin \
  --query-vecs sift_query.fvecs \
  --query-ranges sift_query_ranges.csv \
  --ground-truth sift_gt.ivecs \
  --metric l2 --scalar-dims 1 \
  --sfc z-order --scalar-bits 8 --partitions 256 --partition-threshold 2000 \
  --sfc-budget 256 --min-probe 1 --max-probe 50 --candidate-target 1000 --k 10
```

### AudioSet (L2)

```bash
./smartivf build \
  --base-vecs audioset_base.fvecs \
  --base-scalars audioset_scalars.csv \
  --index-out audioset_index.bin \
  --metric l2 --scalar-dims 3 \
  --sfc z-order --scalar-bits 8 --partitions 256 --partition-threshold 2000

./smartivf search \
  --index-in audioset_index.bin \
  --query-vecs audioset_query.fvecs \
  --query-ranges audioset_query_ranges.csv \
  --ground-truth audioset_gt.ivecs \
  --metric l2 --scalar-dims 3 \
  --sfc z-order --scalar-bits 8 --partitions 256 --partition-threshold 2000 \
  --sfc-budget 256 --min-probe 1 --max-probe 50 --candidate-target 1000 --k 10
```

### LAION-1M (cosine)

LAION split: first 990 000 rows as base, last 10 000 as queries.

```bash
./smartivf build \
  --base-vecs laion_base.fvecs \
  --base-scalars laion_base_scalars.csv \
  --index-out laion_index.bin \
  --metric cosine --scalar-dims 2 \
  --sfc z-order --scalar-bits 8 --partitions 256 --partition-threshold 2000

./smartivf search \
  --index-in laion_index.bin \
  --query-vecs laion_query.fvecs \
  --query-ranges laion_query_ranges.csv \
  --ground-truth laion_gt.ivecs \
  --metric cosine --scalar-dims 2 \
  --sfc z-order --scalar-bits 8 --partitions 256 --partition-threshold 2000 \
  --sfc-budget 256 --min-probe 1 --max-probe 50 --candidate-target 1000 --k 10
```

## Ground truth generation

Run `python ground_truth/generate_groundtruth.py --help` for the full option
list. Cosine datasets must pass `--metric cosine`; LAION-1M additionally needs
`--n-base 990000`.

## File formats

Vectors (`.fvecs`): 4-byte LE int32 `d`, then `d` × 4-byte LE float32.

Ground truth (`.ivecs`): same as `.fvecs` but int32.

Base scalars (`.csv`): plain CSV, no header, D columns of float per row.

Query ranges (`.csv`): plain CSV, no header, interleaved `(min, max)` pairs per
scalar dimension (`min_0, max_0, min_1, max_1, ...`).

Binary index (`.bin`): custom layout. Header is the magic number `0x53494643`
(`'SIFC'`) followed by a `uint32` version (currently `1`).
