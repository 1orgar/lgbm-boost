#pragma once

#include "fast_scorer/model.h"
#include <cstddef>
#include <cstdint>

namespace fast_scorer {

// Fixed SIMD-friendly tile row capacity (multiple of 16 floats = 64B cache line, power of 2 = shift << 7)
inline constexpr size_t kTileStride = 128;

// Pack input rows [row_start, row_start + n_rows) into column-major tile buffer:
// dst_col_major[f * kTileStride + r] for f in [0, n_features), r in [0, n_rows),
// with zero-padding for r in [n_rows, padded_rows).
void PackTileRowMajorF32_Scalar(
    const float* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    float* dst_col_major
) noexcept;

void PackTileRowMajorF64_Scalar(
    const double* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    float* dst_col_major
) noexcept;

void PackTileColumnar_Scalar(
    const ColumnView* cols,
    size_t row_start,
    size_t n_rows,
    size_t n_features,
    float* dst_col_major
) noexcept;

// Discretize tile_col_major into 16-bit global bins once per 128-row tile using branchless monobound search.
// Special sentinels: 0xFFFF = NaN, 0xFFFE = Zero (when feature_has_zero_missing[f] != 0).
void BinTileFeatures_Scalar(
    const CompiledModel& model,
    const float* tile_col_major,
    size_t n_rows,
    uint16_t* dst_tile_bins
) noexcept;

struct KernelTable {
    SimdIsa isa = SimdIsa::Scalar;
    const char* name = "scalar";

    void (*eval_qs32_block)(
        const QS32Block& blk,
        const float* tile_col_major,
        const uint16_t* tile_bins,
        size_t actual_rows,
        float* tile_acc
    ) noexcept = nullptr;

    void (*eval_qs64_block)(
        const QS64Block& blk,
        const float* tile_col_major,
        const uint16_t* tile_bins,
        size_t actual_rows,
        float* tile_acc
    ) noexcept = nullptr;

    void (*eval_compact_trees)(
        const CompactTreeMeta* trees,
        size_t num_trees,
        const CompactNode* all_nodes,
        const float* all_leaves,
        const float* tile_col_major,
        size_t n_rows,
        float* tile_acc
    ) noexcept = nullptr;

    void (*finalize_tile_output)(
        float* tile_acc,
        size_t n_rows,
        int num_outputs,
        ObjectiveType objective,
        float sigmoid_alpha,
        float scale_factor,
        bool raw_score,
        float* out_row_major
    ) noexcept = nullptr;
};

const KernelTable& GetScalarKernels() noexcept;

#if defined(FAST_SCORER_ARCH_X86_64)
const KernelTable& GetAvx2Kernels() noexcept;
const KernelTable& GetAvx512Kernels() noexcept;
#endif

} // namespace fast_scorer
