#include "fast_scorer/kernels.h"
#include "fast_scorer/math_simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace fast_scorer {

namespace {

inline uint32_t Ctz32(uint32_t v) noexcept {
    if (v == 0) return 0;
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward(&idx, v);
    return static_cast<uint32_t>(idx);
#else
    return static_cast<uint32_t>(__builtin_ctz(v));
#endif
}

inline uint32_t Ctz64(uint64_t v) noexcept {
    if (v == 0) return 0;
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward64(&idx, v);
    return static_cast<uint32_t>(idx);
#else
    return static_cast<uint32_t>(__builtin_ctzll(v));
#endif
}

inline bool IsZeroF64(double x) noexcept {
    return std::fabs(x) <= kZeroThreshold;
}

inline uint32_t CountStrictlyLessF64(const double* sorted, uint32_t len, double val) noexcept {
    if (len == 0u || !(sorted[0] < val)) return 0u;
    if (sorted[len - 1u] < val) return len;
    const double* base = sorted;
    uint32_t n = len;
    while (n > 1u) {
        const uint32_t half = n >> 1;
        base += (base[half - 1u] < val) ? half : 0u;
        n -= half;
    }
    return static_cast<uint32_t>(base - sorted) + static_cast<uint32_t>(base[0] < val);
}

template <typename SrcT>
inline void CopyCastColumnSliceF64(
    const void* raw_ptr,
    size_t row_start,
    size_t n_rows,
    size_t padded_rows,
    double* dcol
) noexcept {
    const SrcT* s = static_cast<const SrcT*>(raw_ptr) + row_start;
    for (size_t r = 0; r < n_rows; ++r) {
        dcol[r] = static_cast<double>(s[r]);
    }
    for (size_t r = n_rows; r < padded_rows; ++r) {
        dcol[r] = 0.0;
    }
}

} // namespace

// ============================================================================
// Cache-friendly 8x8 blocked tile packers -> exact float64 column-major tile
// ============================================================================

void PackTileRowMajorF32_Scalar(
    const float* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    double* dst_col_major
) noexcept {
    const size_t padded_rows = (n_rows + 15u) & ~size_t(15u);

    size_t r0 = 0;
    for (; r0 + 8 <= n_rows; r0 += 8) {
        size_t f0 = 0;
        for (; f0 + 8 <= n_features; f0 += 8) {
            double buf[8][8];
            for (size_t dr = 0; dr < 8; ++dr) {
                const float* srow = src + static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + f0;
                for (size_t df = 0; df < 8; ++df) {
                    buf[dr][df] = static_cast<double>(srow[df]);
                }
            }
            for (size_t df = 0; df < 8; ++df) {
                double* dcol = dst_col_major + (f0 + df) * kTileStride + r0;
                for (size_t dr = 0; dr < 8; ++dr) {
                    dcol[dr] = buf[dr][df];
                }
            }
        }
        for (; f0 < n_features; ++f0) {
            double* dcol = dst_col_major + f0 * kTileStride + r0;
            for (size_t dr = 0; dr < 8; ++dr) {
                dcol[dr] = static_cast<double>(
                    src[static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + static_cast<ptrdiff_t>(f0)]
                );
            }
        }
    }

    if (r0 < padded_rows) {
        for (size_t f = 0; f < n_features; ++f) {
            double* dcol = dst_col_major + f * kTileStride;
            for (size_t r = r0; r < n_rows; ++r) {
                dcol[r] = static_cast<double>(
                    src[static_cast<ptrdiff_t>(r) * row_stride_elems + static_cast<ptrdiff_t>(f)]
                );
            }
            for (size_t r = n_rows; r < padded_rows; ++r) {
                dcol[r] = 0.0;
            }
        }
    }
}

void PackTileRowMajorF64_Scalar(
    const double* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    double* dst_col_major
) noexcept {
    const size_t padded_rows = (n_rows + 15u) & ~size_t(15u);

    size_t r0 = 0;
    for (; r0 + 8 <= n_rows; r0 += 8) {
        size_t f0 = 0;
        for (; f0 + 8 <= n_features; f0 += 8) {
            double buf[8][8];
            for (size_t dr = 0; dr < 8; ++dr) {
                const double* srow = src + static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + f0;
                std::memcpy(buf[dr], srow, 8 * sizeof(double));
            }
            for (size_t df = 0; df < 8; ++df) {
                double* dcol = dst_col_major + (f0 + df) * kTileStride + r0;
                for (size_t dr = 0; dr < 8; ++dr) {
                    dcol[dr] = buf[dr][df];
                }
            }
        }
        for (; f0 < n_features; ++f0) {
            double* dcol = dst_col_major + f0 * kTileStride + r0;
            for (size_t dr = 0; dr < 8; ++dr) {
                dcol[dr] = src[static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + static_cast<ptrdiff_t>(f0)];
            }
        }
    }

    if (r0 < padded_rows) {
        for (size_t f = 0; f < n_features; ++f) {
            double* dcol = dst_col_major + f * kTileStride;
            for (size_t r = r0; r < n_rows; ++r) {
                dcol[r] = src[static_cast<ptrdiff_t>(r) * row_stride_elems + static_cast<ptrdiff_t>(f)];
            }
            for (size_t r = n_rows; r < padded_rows; ++r) {
                dcol[r] = 0.0;
            }
        }
    }
}

void PackTileColumnar_Scalar(
    const ColumnView* cols,
    size_t row_start,
    size_t n_rows,
    size_t n_features,
    double* dst_col_major
) noexcept {
    const size_t padded_rows = (n_rows + 15u) & ~size_t(15u);

    for (size_t f = 0; f < n_features; ++f) {
        double* dcol = dst_col_major + f * kTileStride;
        const ColumnView& cv = cols[f];
        switch (cv.dtype) {
            case ColumnDType::Float64: {
                const double* s = static_cast<const double*>(cv.data) + row_start;
                std::memcpy(dcol, s, n_rows * sizeof(double));
                for (size_t r = n_rows; r < padded_rows; ++r) dcol[r] = 0.0;
                break;
            }
            case ColumnDType::Float32:
                CopyCastColumnSliceF64<float>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int32:
                CopyCastColumnSliceF64<int32_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int64:
                CopyCastColumnSliceF64<int64_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int16:
                CopyCastColumnSliceF64<int16_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int8:
                CopyCastColumnSliceF64<int8_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::UInt8:
                CopyCastColumnSliceF64<uint8_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::UInt16:
                CopyCastColumnSliceF64<uint16_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::UInt32:
                CopyCastColumnSliceF64<uint32_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
        }
    }
}

// ============================================================================
// Scalar / Auto-Vectorized Kernels (ARM64 NEON & Generic Fallback, Exact FP64)
// ============================================================================

namespace {

void EvalQS32Block_Scalar(
    const QS32Block& blk,
    const double* tile_col_major,
    size_t actual_rows,
    double* tile_acc
) noexcept {
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    constexpr size_t W = 16; // 16x uint32_t = 64 bytes = 1 exact cache line per tree
    const size_t n_rows = (actual_rows + 15u) & ~size_t(15u);
    alignas(64) uint32_t v[128][W];

    const double* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks = blk.masks.data();
    const QS32SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS32SpecialEntry* zero_entries = blk.zero_entries.data();
    const uint32_t stride = blk.leaf_stride;
    const double* leaves_base = blk.leaf_values.data();

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            for (size_t l = 0; l < W; ++l) {
                v[t][l] = 0xFFFFFFFFu;
            }
        }

        for (const QS32FeatureSpan& span : blk.features) {
            const double* fx = tile_col_major + static_cast<size_t>(span.feature_idx) * kTileStride + r0;

            uint32_t nan_lanes = 0;
            uint32_t zero_lanes = 0;
            alignas(64) double x_clean[W];
            double x_max = -std::numeric_limits<double>::infinity();
            double x_min = std::numeric_limits<double>::infinity();

            if (!span.has_zero_missing) {
                for (size_t l = 0; l < W; ++l) {
                    const double val = fx[l];
                    if (std::isnan(val)) {
                        nan_lanes |= (1u << l);
                        x_clean[l] = -std::numeric_limits<double>::infinity();
                    } else {
                        x_clean[l] = val;
                        if (val > x_max) x_max = val;
                        if (val < x_min) x_min = val;
                    }
                }
            } else {
                for (size_t l = 0; l < W; ++l) {
                    const double val = fx[l];
                    if (std::isnan(val)) {
                        nan_lanes |= (1u << l);
                        x_clean[l] = -std::numeric_limits<double>::infinity();
                    } else if (IsZeroF64(val)) {
                        zero_lanes |= (1u << l);
                        x_clean[l] = -std::numeric_limits<double>::infinity();
                    } else {
                        x_clean[l] = val;
                        if (val > x_max) x_max = val;
                        if (val < x_min) x_min = val;
                    }
                }
            }

            uint32_t idx = span.split_begin;
            const uint32_t len = span.split_end - idx;
            const uint32_t idx_stop = idx + CountStrictlyLessF64(thresholds + idx, len, x_max);

            if (idx < idx_stop) {
                if ((nan_lanes | zero_lanes) == 0u) {
                    const uint32_t idx_all = idx + CountStrictlyLessF64(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        const uint32_t m = masks[idx];
                        uint32_t* vt_arr = v[tid];
                        for (size_t l = 0; l < W; ++l) {
                            vt_arr[l] &= m;
                        }
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    const double vt = thresholds[idx];
                    const uint16_t tid = tree_ids[idx];
                    const uint32_t m = masks[idx];
                    uint32_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        const uint32_t keep = (x_clean[l] <= vt) ? 0xFFFFFFFFu : m;
                        vt_arr[l] &= keep;
                    }
                }
            }

            if (nan_lanes != 0) {
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    const uint32_t m = nan_entries[ni].mask;
                    uint32_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        if ((nan_lanes & (1u << l)) != 0) vt_arr[l] &= m;
                    }
                }
            }

            if (zero_lanes != 0) {
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    const uint32_t m = zero_entries[zi].mask;
                    uint32_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        if ((zero_lanes & (1u << l)) != 0) vt_arr[l] &= m;
                    }
                }
            }
        }

        if (blk.single_class_id >= 0) {
            alignas(64) double acc_local[W] = {0};
            for (uint32_t t = 0; t < n_trees; ++t) {
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                for (size_t l = 0; l < W; ++l) {
                    acc_local[l] += tl[Ctz32(v[t][l])];
                }
            }
            double* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            for (size_t l = 0; l < W; ++l) {
                dst[l] += acc_local[l];
            }
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                double* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                for (size_t l = 0; l < W; ++l) {
                    dst[l] += tl[Ctz32(v[t][l])];
                }
            }
        }
    }
}

void EvalQS64Block_Scalar(
    const QS64Block& blk,
    const double* tile_col_major,
    size_t actual_rows,
    double* tile_acc
) noexcept {
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    constexpr size_t W = 8;
    const size_t n_rows = (actual_rows + 7u) & ~size_t(7u);
    alignas(64) uint64_t v[64][W];

    const double* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks_lo = blk.masks_lo.data();
    const uint32_t* masks_hi = blk.masks_hi.data();
    const QS64SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS64SpecialEntry* zero_entries = blk.zero_entries.data();
    const uint32_t stride = blk.leaf_stride;
    const double* leaves_base = blk.leaf_values.data();

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            for (size_t l = 0; l < W; ++l) {
                v[t][l] = ~0ULL;
            }
        }

        for (const QS64FeatureSpan& span : blk.features) {
            const double* fx = tile_col_major + static_cast<size_t>(span.feature_idx) * kTileStride + r0;

            uint32_t nan_lanes = 0;
            uint32_t zero_lanes = 0;
            alignas(64) double x_clean[W];
            double x_max = -std::numeric_limits<double>::infinity();
            double x_min = std::numeric_limits<double>::infinity();

            for (size_t l = 0; l < W; ++l) {
                const double val = fx[l];
                if (std::isnan(val)) {
                    nan_lanes |= (1u << l);
                    x_clean[l] = -std::numeric_limits<double>::infinity();
                } else if (span.has_zero_missing && IsZeroF64(val)) {
                    zero_lanes |= (1u << l);
                    x_clean[l] = -std::numeric_limits<double>::infinity();
                } else {
                    x_clean[l] = val;
                    if (val > x_max) x_max = val;
                    if (val < x_min) x_min = val;
                }
            }

            uint32_t idx = span.split_begin;
            const uint32_t len = span.split_end - idx;
            const uint32_t idx_stop = idx + CountStrictlyLessF64(thresholds + idx, len, x_max);

            if (idx < idx_stop) {
                if ((nan_lanes | zero_lanes) == 0u) {
                    const uint32_t idx_all = idx + CountStrictlyLessF64(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                        uint64_t* vt_arr = v[tid];
                        for (size_t l = 0; l < W; ++l) {
                            vt_arr[l] &= m64;
                        }
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    const double vt = thresholds[idx];
                    const uint16_t tid = tree_ids[idx];
                    const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                    uint64_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        const uint64_t keep = (x_clean[l] <= vt) ? ~0ULL : m64;
                        vt_arr[l] &= keep;
                    }
                }
            }

            if (nan_lanes != 0) {
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    const uint64_t m64 = (static_cast<uint64_t>(nan_entries[ni].mask_hi) << 32) | nan_entries[ni].mask_lo;
                    uint64_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        if ((nan_lanes & (1u << l)) != 0) vt_arr[l] &= m64;
                    }
                }
            }

            if (zero_lanes != 0) {
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    const uint64_t m64 = (static_cast<uint64_t>(zero_entries[zi].mask_hi) << 32) | zero_entries[zi].mask_lo;
                    uint64_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        if ((zero_lanes & (1u << l)) != 0) vt_arr[l] &= m64;
                    }
                }
            }
        }

        if (blk.single_class_id >= 0) {
            alignas(64) double acc_local[W] = {0};
            for (uint32_t t = 0; t < n_trees; ++t) {
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                for (size_t l = 0; l < W; ++l) {
                    acc_local[l] += tl[Ctz64(v[t][l])];
                }
            }
            double* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            for (size_t l = 0; l < W; ++l) {
                dst[l] += acc_local[l];
            }
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                double* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                for (size_t l = 0; l < W; ++l) {
                    dst[l] += tl[Ctz64(v[t][l])];
                }
            }
        }
    }
}

void EvalCompactTrees_Scalar(
    const CompactTreeMeta* trees,
    size_t num_trees,
    const CompactNode* all_nodes,
    const double* all_leaves,
    const double* tile_col_major,
    size_t n_rows,
    double* tile_acc
) noexcept {
    for (size_t t = 0; t < num_trees; ++t) {
        const CompactTreeMeta& meta = trees[t];
        const CompactNode* nodes = all_nodes + meta.node_offset;
        const double* leaves = all_leaves + meta.leaf_offset;
        double* dst = tile_acc + static_cast<size_t>(meta.class_id) * kTileStride;

        if (!meta.has_zero_missing) {
            for (size_t r = 0; r < n_rows; ++r) {
                int32_t idx = 0;
                while (idx >= 0) {
                    const CompactNode& n = nodes[idx];
                    const double x = tile_col_major[(static_cast<size_t>(n.feature_idx()) << 7) + r];
                    const bool go_left = std::isnan(x) ? n.nan_goes_left() : (x <= n.threshold);
                    idx = go_left ? n.left_child : n.right_child;
                }
                dst[r] += leaves[~idx];
            }
        } else {
            for (size_t r = 0; r < n_rows; ++r) {
                int32_t idx = 0;
                while (idx >= 0) {
                    const CompactNode& n = nodes[idx];
                    const double x = tile_col_major[(static_cast<size_t>(n.feature_idx()) << 7) + r];
                    bool go_left;
                    if (std::isnan(x)) {
                        go_left = n.nan_goes_left();
                    } else if (n.has_zero_missing() && IsZeroF64(x)) {
                        go_left = n.zero_goes_left();
                    } else {
                        go_left = (x <= n.threshold);
                    }
                    idx = go_left ? n.left_child : n.right_child;
                }
                dst[r] += leaves[~idx];
            }
        }
    }
}

} // namespace

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
) noexcept {
    const bool need_scale = (scale_factor != 1.0);

    if (num_outputs == 1) {
        const double* src = tile_acc;
        if (out_f64 != nullptr) {
            if (raw_score || objective == ObjectiveType::Regression) {
                if (need_scale) {
                    for (size_t r = 0; r < n_rows; ++r) out_f64[r] = src[r] * scale_factor;
                } else {
                    std::memcpy(out_f64, src, n_rows * sizeof(double));
                }
            } else if (objective == ObjectiveType::BinaryLogloss) {
                for (size_t r = 0; r < n_rows; ++r) {
                    const double s = need_scale ? (src[r] * scale_factor) : src[r];
                    out_f64[r] = simd_math::SafeSigmoidF64(s, sigmoid_alpha);
                }
            } else if (objective == ObjectiveType::Exponential) {
                for (size_t r = 0; r < n_rows; ++r) {
                    const double s = need_scale ? (src[r] * scale_factor) : src[r];
                    out_f64[r] = simd_math::SafeExpF64(s);
                }
            }
        } else if (out_f32 != nullptr) {
            if (raw_score || objective == ObjectiveType::Regression) {
                for (size_t r = 0; r < n_rows; ++r) {
                    const double s = need_scale ? (src[r] * scale_factor) : src[r];
                    out_f32[r] = static_cast<float>(s);
                }
            } else if (objective == ObjectiveType::BinaryLogloss) {
                for (size_t r = 0; r < n_rows; ++r) {
                    const double s = need_scale ? (src[r] * scale_factor) : src[r];
                    out_f32[r] = static_cast<float>(simd_math::SafeSigmoidF64(s, sigmoid_alpha));
                }
            } else if (objective == ObjectiveType::Exponential) {
                for (size_t r = 0; r < n_rows; ++r) {
                    const double s = need_scale ? (src[r] * scale_factor) : src[r];
                    out_f32[r] = static_cast<float>(simd_math::SafeExpF64(s));
                }
            }
        }
        return;
    }

    // Multi-output (K > 1)
    const size_t K = static_cast<size_t>(num_outputs);
    if (out_f64 != nullptr) {
        if (raw_score || objective == ObjectiveType::Regression) {
            for (size_t r = 0; r < n_rows; ++r) {
                double* drow = out_f64 + r * K;
                for (size_t c = 0; c < K; ++c) {
                    const double s = tile_acc[c * kTileStride + r];
                    drow[c] = need_scale ? (s * scale_factor) : s;
                }
            }
        } else if (objective == ObjectiveType::BinaryLogloss) {
            for (size_t r = 0; r < n_rows; ++r) {
                double* drow = out_f64 + r * K;
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    drow[c] = simd_math::SafeSigmoidF64(s, sigmoid_alpha);
                }
            }
        } else if (objective == ObjectiveType::MulticlassSoftmax) {
            for (size_t r = 0; r < n_rows; ++r) {
                double* drow = out_f64 + r * K;
                double max_v = -std::numeric_limits<double>::infinity();
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    if (s > max_v) max_v = s;
                }
                double sum_e = 0.0;
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    const double e = std::exp(s - max_v);
                    drow[c] = e;
                    sum_e += e;
                }
                const double inv = 1.0 / sum_e;
                for (size_t c = 0; c < K; ++c) {
                    drow[c] *= inv;
                }
            }
        } else if (objective == ObjectiveType::Exponential) {
            for (size_t r = 0; r < n_rows; ++r) {
                double* drow = out_f64 + r * K;
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    drow[c] = simd_math::SafeExpF64(s);
                }
            }
        }
    } else if (out_f32 != nullptr) {
        if (raw_score || objective == ObjectiveType::Regression) {
            for (size_t r = 0; r < n_rows; ++r) {
                float* drow = out_f32 + r * K;
                for (size_t c = 0; c < K; ++c) {
                    const double s = tile_acc[c * kTileStride + r];
                    drow[c] = static_cast<float>(need_scale ? (s * scale_factor) : s);
                }
            }
        } else if (objective == ObjectiveType::BinaryLogloss) {
            for (size_t r = 0; r < n_rows; ++r) {
                float* drow = out_f32 + r * K;
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    drow[c] = static_cast<float>(simd_math::SafeSigmoidF64(s, sigmoid_alpha));
                }
            }
        } else if (objective == ObjectiveType::MulticlassSoftmax) {
            for (size_t r = 0; r < n_rows; ++r) {
                float* drow = out_f32 + r * K;
                double max_v = -std::numeric_limits<double>::infinity();
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    if (s > max_v) max_v = s;
                }
                double sum_e = 0.0;
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    const double e = std::exp(s - max_v);
                    tile_acc[c * kTileStride + r] = e;
                    sum_e += e;
                }
                const double inv = 1.0 / sum_e;
                for (size_t c = 0; c < K; ++c) {
                    drow[c] = static_cast<float>(tile_acc[c * kTileStride + r] * inv);
                }
            }
        } else if (objective == ObjectiveType::Exponential) {
            for (size_t r = 0; r < n_rows; ++r) {
                float* drow = out_f32 + r * K;
                for (size_t c = 0; c < K; ++c) {
                    const double s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                                : tile_acc[c * kTileStride + r];
                    drow[c] = static_cast<float>(simd_math::SafeExpF64(s));
                }
            }
        }
    }
}

const KernelTable& GetScalarKernels() noexcept {
    static const KernelTable kTable = {
        SimdIsa::Scalar,
        "scalar",
        &EvalQS32Block_Scalar,
        &EvalQS64Block_Scalar,
        &EvalCompactTrees_Scalar,
        &FinalizeTileOutput_Scalar
    };
    return kTable;
}

} // namespace fast_scorer
