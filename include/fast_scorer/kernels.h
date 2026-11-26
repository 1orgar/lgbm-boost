#pragma once

#include "fast_scorer/model.h"
#include <cstddef>
#include <cstdint>

namespace fast_scorer {

// Fixed SIMD-friendly tile row capacity (multiple of 16 doubles = two 64B cache lines, power of 2 = shift << 7)
inline constexpr size_t kTileStride = 128;

// Exact LightGBM kZeroThreshold (include/LightGBM/meta.h: const double kZeroThreshold = 1e-35f)
inline constexpr double kZeroThreshold = static_cast<double>(1e-35f);

// Pack input rows [row_start, row_start + n_rows) into 64B-aligned column-major float64 tile buffer:
// dst_col_major[f * kTileStride + r] for f in [0, n_features), r in [0, n_rows),
// with zero-padding for r in [n_rows, padded_rows).
void PackTileRowMajorF32_Scalar(
    const float* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    double* dst_col_major
) noexcept;

void PackTileRowMajorF64_Scalar(
    const double* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    double* dst_col_major
) noexcept;

void PackTileColumnar_Scalar(
    const ColumnView* cols,
    size_t row_start,
    size_t n_rows,
    size_t n_features,
    double* dst_col_major
) noexcept;

// Exact float64 output finalization (scaling + Sigmoid / Softmax / Exp) writing directly
// to out_f64 (default) or out_f32 (optional float32 output).
void FinalizeTileOutput_Scalar(
    double* tile_acc,
    size_t n_rows,
    int num_outputs,
    ObjectiveType objective,
    double sigmoid_alpha,
    double scale_factor,
    bool raw_score,
    double* out_f64,
    float* out_f32
) noexcept;

struct KernelTable {
    SimdIsa isa = SimdIsa::Scalar;
    const char* name = "scalar";

    void (*eval_qs32_block)(
        const QS32Block& blk,
        const double* tile_col_major,
        size_t actual_rows,
        double* tile_acc
    ) noexcept = nullptr;

    void (*eval_qs64_block)(
        const QS64Block& blk,
        const double* tile_col_major,
        size_t actual_rows,
        double* tile_acc
    ) noexcept = nullptr;

    void (*eval_compact_trees)(
        const CompactTreeMeta* trees,
        size_t num_trees,
        const CompactNode* all_nodes,
        const double* all_leaves,
        const double* tile_col_major,
        size_t n_rows,
        double* tile_acc
    ) noexcept = nullptr;

    void (*finalize_tile_output)(
        double* tile_acc,
        size_t n_rows,
        int num_outputs,
        ObjectiveType objective,
        double sigmoid_alpha,
        double scale_factor,
        bool raw_score,
        double* out_f64,
        float* out_f32
    ) noexcept = nullptr;
};

const KernelTable& GetScalarKernels() noexcept;

#if defined(FAST_SCORER_ARCH_X86_64)
const KernelTable& GetAvx2Kernels() noexcept;
const KernelTable& GetAvx512Kernels() noexcept;
#endif

} // namespace fast_scorer
