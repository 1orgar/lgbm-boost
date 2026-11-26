#!/usr/bin/env python3
"""Stage 4 installer: vendors fast_scorer into a cloned LightGBM repository.

Usage:
    python fork_integration/install_into_lightgbm.py /path/to/LightGBM
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent

CORE_HEADERS = [
    "compiler.h",
    "kernels.h",
    "math_simd.h",
    "model.h",
    "parser.h",
    "scorer.h",
]

CORE_SOURCES = [
    "compiler.cpp",
    "kernels_avx2.cpp",
    "kernels_avx512.cpp",
    "kernels_scalar.cpp",
    "model_parser.cpp",
    "scorer.cpp",
]


def copy_sources(lgbm_dir: Path) -> None:
    inc_dst = lgbm_dir / "include" / "LightGBM" / "fast_scorer"
    src_dst = lgbm_dir / "src" / "boosting" / "fast_scorer"
    inc_dst.mkdir(parents=True, exist_ok=True)
    src_dst.mkdir(parents=True, exist_ok=True)

    for h in CORE_HEADERS:
        txt = (REPO_ROOT / "include" / "fast_scorer" / h).read_text(encoding="utf-8")
        txt = txt.replace('#include "fast_scorer/', '#include <LightGBM/fast_scorer/')
        (inc_dst / h).write_text(txt, encoding="utf-8")

    shutil.copy2(
        REPO_ROOT / "fork_integration" / "lgbm_c_api_hook.hpp",
        inc_dst / "lgbm_c_api_hook.hpp",
    )

    for c in CORE_SOURCES:
        txt = (REPO_ROOT / "src" / c).read_text(encoding="utf-8")
        txt = txt.replace('#include "fast_scorer/', '#include <LightGBM/fast_scorer/')
        (src_dst / c).write_text(txt, encoding="utf-8")

    print(f"[OK] Copied {len(CORE_HEADERS) + 1} headers -> {inc_dst}")
    print(f"[OK] Copied {len(CORE_SOURCES)} sources -> {src_dst}")


CMAKE_SNIPPET = """
# --- fast_scorer SIMD batch scoring integration ---
list(APPEND LIGHTGBM_SRCS
    src/boosting/fast_scorer/compiler.cpp
    src/boosting/fast_scorer/kernels_scalar.cpp
    src/boosting/fast_scorer/model_parser.cpp
    src/boosting/fast_scorer/scorer.cpp
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    list(APPEND LIGHTGBM_SRCS
        src/boosting/fast_scorer/kernels_avx2.cpp
        src/boosting/fast_scorer/kernels_avx512.cpp
    )
    add_compile_definitions(FAST_SCORER_ARCH_X86_64=1 FAST_SCORER_HAVE_OPENMP=1)
    if(MSVC)
        set_source_files_properties(src/boosting/fast_scorer/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/O2;/arch:AVX2")
        set_source_files_properties(src/boosting/fast_scorer/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/O2;/arch:AVX512")
    else()
        set_source_files_properties(src/boosting/fast_scorer/kernels_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-O3;-mavx2;-mfma;-mbmi;-mbmi2;-fno-math-errno;-fno-trapping-math")
        set_source_files_properties(src/boosting/fast_scorer/kernels_avx512.cpp PROPERTIES
            COMPILE_OPTIONS "-O3;-mavx512f;-mavx512cd;-mavx512bw;-mavx512vl;-mavx512dq;-mfma;-mbmi;-mbmi2;-fno-math-errno;-fno-trapping-math")
    endif()
endif()
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor fast_scorer into a LightGBM fork")
    parser.add_argument("lightgbm_path", type=Path, help="Path to cloned LightGBM repository")
    args = parser.parse_args()

    lgbm_dir = args.lightgbm_path.resolve()
    if not (lgbm_dir / "include" / "LightGBM").is_dir():
        raise SystemExit(f"Not a valid LightGBM repository root: {lgbm_dir}")

    copy_sources(lgbm_dir)
    print("\nNext steps in LightGBM fork:")
    print("1. Append the following block to LightGBM/CMakeLists.txt:")
    print(CMAKE_SNIPPET)
    print("2. In src/c_api.cpp:")
    print("   - Add `#include <LightGBM/fast_scorer/lgbm_c_api_hook.hpp>`")
    print("   - Add member `mutable FastBatchScorerCache fast_batch_cache_;` to `class Booster`")
    print("   - At the start of `Booster::Predict(...)` (row-major dense branch), call:")
    print("     `if (is_row_major && fast_batch_cache_.TryPredictRowMajor(boosting_.get(), data, data_type, nrow, ncol, predict_type, start_iteration, num_iteration, config, out_len, out_result)) return;`")


if __name__ == "__main__":
    main()
