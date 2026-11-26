# lgbm-boost

[![CI](https://github.com/1orgar/lgbm-boost/actions/workflows/ci.yml/badge.svg)](https://github.com/1orgar/lgbm-boost/actions/workflows/ci.yml)
[![Publish to PyPI](https://github.com/1orgar/lgbm-boost/actions/workflows/publish.yml/badge.svg)](https://github.com/1orgar/lgbm-boost/actions/workflows/publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/lgbm-boost.svg)](https://pypi.org/project/lgbm-boost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**`lgbm-boost`** is a high-throughput C++17 SIMD batch inference engine for **LightGBM** gradient boosted decision tree ensembles.

It provides both:
1. A lightweight, zero-dependency Python package (`lgbm_boost`) with a drop-in `FastBooster` wrapper for `lightgbm.Booster` and `scikit-learn` estimators (`LGBMClassifier`, `LGBMRegressor`, `LGBMRanker`).
2. A self-contained C++17 `fast_scorer` module (`fork_integration/`) ready to vendor directly into a custom fork of `LightGBM`.

---

## Key Optimizations

Stock `LightGBM` evaluates trees row-by-row via pointer-chasing across multiple heap vectors per tree, incurring ~50% branch mispredictions at every tree level, L1/L2 cache thrashing on large ensembles, and full-matrix `float64` allocations when scoring `pandas.DataFrame` inputs.

`lgbm-boost` eliminates these bottlenecks through five architectural layers:

- **16-Row Vectorized QuickScorer (`V-QuickScorer`, `QS32` & `QS64` for `num_leaves <= 64`)**: Replaces pointer-chasing tree traversal with L1d-resident bitmask intersections (`64` trees per block = `4 KB` working set), branchless monobound threshold range narrowing, hardware `AVX512CD` `vplzcntd`/`vplzcntq` + `{1to16}` embedded broadcast `vpandd {%k1}` on `AVX-512`, and an IEEE-754 float32-exponent parallel `tzcnt` trick on `AVX2`.
- **SIMD Branchless Compact Traversal (`CompactTree` for deep trees `> 64` leaves)**: Packs internal nodes into aligned 16-byte BFS-ordered records with unrolled root evaluation, 8/16-row SIMD gathers, and zero-cost `NaN` routing via ordered/unordered `VCMPPS` predicates.
- **Exact `float64 -> float32` Threshold Lowering**: Rounds `double` split thresholds downward via `std::nextafterf` at model compile time so `x_f32 <= t_f32_floor` is mathematically identical to `double(x_f32) <= t_f64` for every representable `float32` value (`max_abs_diff ~ 1e-7` vs stock LightGBM).
- **Static `NaN` & `Zero` Missing-Value Resolution**: Precomputes LightGBM's exact `NumericalDecision` bitmask semantics (`MissingType::None`, `MissingType::Zero`, `MissingType::NaN`, and `default_left`) at load time, keeping the inner SIMD loop free of missing-type branches.
- **2D Cache Tiling (`RowTile=128 × TreeBlock=64`) & Zero-Copy `pandas.DataFrame` Streaming**: Streams input columns directly into 64-byte-aligned thread-local L2 buffers without allocating a temporary `N × F` `float64` matrix, followed by fused vectorized `Sigmoid` / `Softmax` / `Exp` (degree-6 Cody–Waite FMA `expf`, `< 1.5 ULP`).
- **Runtime CPUID + `XGETBV` Dispatch**: Single binary wheel automatically selects `avx512` → `avx2` → `scalar` at runtime and uses a built-in lock-free atomic C++17 worker pool (no external `libomp` runtime required).

---

## Benchmarks

Measured on a `1,000`-tree, `100`-feature binary classification ensemble (`5% NaN` rate) against `lightgbm 4.7.0`.

### Linux `x86_64` (`AVX-512`)

| `num_leaves` | Rows (`N`) | Input Type | Threads | LightGBM (ms) | `lgbm-boost` (ms) | Speedup | Max Abs Diff |
| :---: | ---: | :---: | :---: | ---: | ---: | :---: | :---: |
| **31** | 1,000 | `ndarray float32` | 8 | 15.57 | **2.02** | **7.69x** | `1.12e-07` |
| **31** | 10,000 | `ndarray float32` | 1 | 467.09 | **62.60** | **7.46x** | `1.60e-07` |
| **31** | 100,000 | `ndarray float32` | 1 | 4,652.96 | **625.44** | **7.44x** | `1.61e-07` |
| **31** | 100,000 | `ndarray float32` | 8 | 1,356.72 | **194.74** | **6.97x** | `1.61e-07` |
| **31** | 100,000 | `pandas.DataFrame` | 8 | 1,374.67 | **196.36** | **7.00x** | `1.61e-07` |
| **63** | 1,000 | `ndarray float32` | 1 | 75.02 | **15.19** | **4.94x** | `1.44e-07` |
| **63** | 100,000 | `ndarray float32` | 1 | 7,489.82 | **1,525.04** | **4.91x** | `2.03e-07` |
| **63** | 100,000 | `pandas.DataFrame` | 8 | 2,127.35 | **562.37** | **3.78x** | `2.03e-07` |

### Apple Silicon `arm64` (`Scalar` target)

| `num_leaves` | Rows (`N`) | Input Type | Threads | LightGBM (ms) | `lgbm-boost` (ms) | Speedup | Max Abs Diff |
| :---: | ---: | :---: | :---: | ---: | ---: | :---: | :---: |
| **31** | 1,000 | `ndarray float32` | 1 | 35.82 | **6.46** | **5.55x** | `1.12e-07` |
| **31** | 100,000 | `ndarray float32` | 8 | 539.18 | **106.47** | **5.06x** | `1.59e-07` |
| **31** | 500,000 | `pandas.DataFrame` | 8 | 2,802.03 | **575.82** | **4.87x** | `1.70e-07` |
| **63** | 100,000 | `ndarray float32` | 1 | 5,621.73 | **1,428.87** | **3.93x** | `1.92e-07` |
| **127** | 500,000 | `pandas.DataFrame` | 8 | 7,338.70 | **3,201.17** | **2.29x** | `9.88e-07` |

---

## Installation

### From PyPI

```bash
pip install lgbm-boost
```

### From GitHub / Source

Requires a C++17 compiler (`g++`, `clang++`, or MSVC). `CMake`, `Ninja`, and `nanobind` are fetched automatically in an isolated build environment:

```bash
pip install "git+https://github.com/1orgar/lgbm-boost.git"
```

---

## Quickstart

```python
import lightgbm as lgb
import numpy as np
from lgbm_boost import FastBooster, detect_best_isa

print("Active SIMD backend:", detect_best_isa())  # 'avx512', 'avx2', or 'scalar'

# Load from an in-memory lightgbm.Booster, LGBMClassifier/LGBMRegressor, file path, or model string
booster = lgb.Booster(model_file="model.txt")
fast_model = FastBooster(booster)

# Score numpy.ndarray (C-contiguous, F-contiguous, float32/float64) or pandas.DataFrame (zero-copy per column)
X = np.random.randn(100_000, fast_model.num_features).astype(np.float32)
scores = fast_model.predict(
    X,
    num_threads=8,          # 0 = all hardware threads
    raw_score=False,        # True = raw margins before sigmoid/softmax/exp
    start_iteration=0,
    num_iteration=-1,
)
```

### Supported Model Features
- **Objectives**: `regression`, `regression_l1`, `huber`, `fair`, `quantile`, `lambdarank`, `binary` (including custom `sigmoid`), `cross_entropy`, `multiclass`, `multiclassova`, `poisson`, `gamma`, `tweedie`, and Random Forest (`boosting="rf"` / `average_output`).
- **Missing Values**: Full parity with LightGBM `use_missing=True/False`, `zero_as_missing=True/False`, `NaN`, `±0.0`, and `±Inf`.
- **Tree Sizes**: Arbitrary leaf counts up to `32,768` leaves per tree (including `1`-leaf constant trees and mixed-depth ensembles).

*(Note: Categorical bitset splits and `linear_tree=True` are not supported in the standalone engine; when integrated into a LightGBM fork via `fork_integration/`, unsupported models transparently fall back to LightGBM's stock predictor.)*

---

## Vendoring into a LightGBM Fork

To embed `fast_scorer` directly inside a cloned `lightgbm-org/LightGBM` repository so standard `Booster.predict()` accelerates dense batch calls automatically:

```bash
python fork_integration/install_into_lightgbm.py /path/to/LightGBM
```

See [`fork_integration/lgbm_c_api_hook.hpp`](fork_integration/lgbm_c_api_hook.hpp) for the `LGBM_BoosterPredictForMat` integration cache.

---

## Local Development & Testing

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -v
.venv/bin/python benchmarks/run_suite.py --rows 1000 10000 100000 500000 --leaves 31 63 127
```

## License

MIT License — see [LICENSE](LICENSE).
