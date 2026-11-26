# lgbm-boost

[![CI](https://github.com/1orgar/lgbm-boost/actions/workflows/ci.yml/badge.svg)](https://github.com/1orgar/lgbm-boost/actions/workflows/ci.yml)
[![Publish to PyPI](https://github.com/1orgar/lgbm-boost/actions/workflows/publish.yml/badge.svg)](https://github.com/1orgar/lgbm-boost/actions/workflows/publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/lgbm-boost.svg)](https://pypi.org/project/lgbm-boost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**`lgbm-boost`** is a high-throughput, exact `float64` C++17 SIMD batch inference engine for **LightGBM** gradient boosted decision tree ensembles.

It provides both:
1. A lightweight, Python-ABI-independent (`py3-none`) package (`lgbm_boost`) with a drop-in `FastBooster` wrapper for `lightgbm.Booster` and `scikit-learn` estimators (`LGBMClassifier`, `LGBMRegressor`, `LGBMRanker`) matching `booster.predict()` to machine epsilon (`~1e-15`, `dtype=float64`).
2. A self-contained C++17 `fast_scorer` module (`fork_integration/`) ready to vendor directly into a custom fork of `LightGBM`.

---

## Key Optimizations

Stock `LightGBM` evaluates trees row-by-row via pointer-chasing across multiple heap vectors per tree, incurring ~50% branch mispredictions at every tree level, L1/L2 cache thrashing on large ensembles, and full-matrix `float64` allocations when scoring `pandas.DataFrame` inputs.

`lgbm-boost` eliminates these bottlenecks while preserving **exact IEEE-754 `float64` precision**:

- **16-Row Vectorized QuickScorer (`V-QuickScorer`, `QS32` & `QS64` for `num_leaves <= 64`)**: Replaces pointer-chasing tree traversal with L1d-resident bitmask intersections (`64` trees per block), branchless monobound threshold range narrowing on exact `double` thresholds, hardware `AVX512CD` `vplzcntd`/`vplzcntq` + `k`-mask predicated `vpandd {%k1}` on `AVX-512`, and single-instruction swizzled mask packing + IEEE-754 exponent `tzcnt` on `AVX2`.
- **SIMD Branchless Compact Traversal (`CompactTree` for deep trees `> 64` leaves)**: Packs internal nodes into aligned 32-byte BFS-ordered records with exact `double` thresholds, unrolled root evaluation, 4/8-row `vgatherdpd` SIMD gathers, and zero-cost `NaN` routing via ordered/unordered `VCMPPD` predicates (`MaxDiff = 0.00` bit-identical to LightGBM).
- **Exact `float64` Numerics (`~1e-15` Parity)**: Evaluates split thresholds, leaf accumulations, and `Sigmoid` / `Softmax` / `Exp` link functions in `double` (`dtype=np.float64` by default, with optional `dtype=np.float32` output when compact buffers are desired).
- **Static `NaN` & `Zero` Missing-Value Resolution**: Precomputes LightGBM's exact `NumericalDecision` bitmask semantics (`MissingType::None`, `MissingType::Zero`, `MissingType::NaN`, and `default_left`) at load time, keeping the inner SIMD loop free of missing-type branches.
- **2D Cache Tiling (`RowTile=128 × TreeBlock=64`) & Zero-Copy `pandas.DataFrame` Streaming**: Streams input columns (`float64`, `float32`, `int8`–`int64`, `uint8`–`uint32`, `bool`) directly into 64-byte-aligned thread-local L2 buffers without allocating a temporary `N × F` matrix.
- **Single Universal `py3-none` Wheel per Architecture (`ctypes` Pure C ABI)**: Uses a pure `extern "C"` shared library (`liblgbm_boost_c`) with zero `Python.h` ABI coupling — **one `.whl` per OS/CPU architecture** works across Python `3.9`–`3.15+`, PyPy, and free-threaded Python, with runtime `CPUID` + `XGETBV` dispatch (`avx512` → `avx2` → `scalar`) and a built-in lock-free C++17 worker pool (no `libomp` required).

---

## Benchmarks (Exact `float64` Output)

Measured on a `1,000`-tree, `100`-feature binary classification ensemble (`5% NaN` rate, `output=float64`) against `lightgbm 4.7.0`.

### Linux `x86_64` (`AVX-512`)

| `num_leaves` | Rows (`N`) | Input Type | Threads | LightGBM (ms) | `lgbm-boost` (ms) | Speedup | Max Abs Diff (`float64`) |
| :---: | ---: | :---: | :---: | ---: | ---: | :---: | :---: |
| **31** | 1,000 | `ndarray float64` | 1 | 51.60 | **8.08** | **6.39x** | `7.22e-16` |
| **31** | 10,000 | `ndarray float64` | 8 | 157.78 | **28.07** | **5.62x** | `9.44e-16` |
| **31** | 100,000 | `ndarray float64` | 1 | 5,091.52 | **795.92** | **6.40x** | `9.99e-16` |
| **31** | 100,000 | `ndarray float64` | 8 | 1,558.67 | **278.90** | **5.59x** | `9.99e-16` |
| **31** | 100,000 | `pandas.DataFrame` | 8 | 1,586.22 | **280.94** | **5.65x** | `9.99e-16` |
| **63** | 100,000 | `ndarray float64` | 1 | 9,451.62 | **2,047.15** | **4.62x** | `1.22e-15` |
| **63** | 100,000 | `pandas.DataFrame` | 8 | 2,891.97 | **763.10** | **3.79x** | `1.22e-15` |
| **127** | 100,000 | `ndarray float64` | 1 | 20,785.00 | **5,254.95** | **3.96x** | `0.00e+00` |
| **127** | 100,000 | `pandas.DataFrame` | 8 | 5,068.76 | **1,322.09** | **3.83x** | `0.00e+00` |

### Apple Silicon `arm64` (`Scalar` target)

| `num_leaves` | Rows (`N`) | Input Type | Threads | LightGBM (ms) | `lgbm-boost` (ms) | Speedup | Max Abs Diff (`float64`) |
| :---: | ---: | :---: | :---: | ---: | ---: | :---: | :---: |
| **31** | 1,000 | `ndarray float64` | 1 | 32.77 | **6.71** | **4.89x** | `7.22e-16` |
| **31** | 100,000 | `ndarray float64` | 8 | 516.26 | **112.38** | **4.59x** | `9.99e-16` |
| **31** | 500,000 | `pandas.DataFrame` | 8 | 2,806.76 | **625.59** | **4.49x** | `1.22e-15` |
| **63** | 100,000 | `ndarray float64` | 1 | 5,637.52 | **1,562.31** | **3.61x** | `1.11e-15` |
| **127** | 500,000 | `ndarray float64` | 1 | 49,948.72 | **21,604.38** | **2.31x** | `0.00e+00` |

---

## Installation

### From PyPI (Single Universal Precompiled Binary Wheel per OS/Arch — No Compiler Needed)

Precompiled Python-version-independent (`py3-none`) C++ SIMD wheels (`AVX-512` + `AVX2` + `Scalar` with runtime CPUID dispatch) are published to **PyPI** and attached directly to **[GitHub Releases](https://github.com/1orgar/lgbm-boost/releases)** — **just 1 `.whl` file per OS/CPU architecture** covering all Python `3.9+` versions:
- **Linux `x86_64`**: `lgbm_boost-0.2.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (compatible out-of-the-box with **Oracle Linux 8 (`ol8`)**, **Oracle Linux 9 (`ol9`)**, RHEL/Alma/Rocky 8 & 9, Debian, and Ubuntu across Python `3.9`–`3.15+`, even with older system `pip`)
- **Linux `aarch64`**: `lgbm_boost-0.2.0-py3-none-manylinux_2_28_aarch64.whl`
- **macOS (`arm64` & `x86_64`)**: `lgbm_boost-0.2.0-py3-none-macosx_11_0_arm64.whl` / `x86_64.whl`
- **Windows (`AMD64`)**: `lgbm_boost-0.2.0-py3-none-win_amd64.whl`

```bash
pip install lgbm-boost

# Or install the single universal x86_64 .whl on air-gapped OL8 / OL9 servers (works on any Python 3.9+):
pip install --only-binary=:all: ./lgbm_boost-0.2.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### From GitHub / Source

Requires only a C++17 compiler (`g++`, `clang++`, or MSVC). `CMake` and `Ninja` are fetched automatically by `scikit-build-core`:

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

# Score numpy.ndarray (C-contiguous, F-contiguous, float64/float32) or pandas.DataFrame (zero-copy per column)
X = np.random.randn(100_000, fast_model.num_features)
scores = fast_model.predict(
    X,
    num_threads=8,          # 0 = all hardware threads
    raw_score=False,        # True = raw margins before sigmoid/softmax/exp
    start_iteration=0,
    num_iteration=-1,
    dtype=np.float64,       # default float64 (matches booster.predict); pass np.float32 if desired
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
