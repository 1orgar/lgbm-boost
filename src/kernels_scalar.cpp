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

inline bool IsZeroF32(float x) noexcept {
    return std::fabs(x) <= 1e-35f;
}

} // namespace

// ============================================================================
// Cache-friendly 8x8 blocked tile packers
// ============================================================================

void PackTileRowMajorF32_Scalar(
    const float* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    float* dst_col_major
) noexcept {
    const size_t padded_rows = (n_rows + 15u) & ~size_t(15u);

    size_t r0 = 0;
    for (; r0 + 8 <= n_rows; r0 += 8) {
        size_t f0 = 0;
        for (; f0 + 8 <= n_features; f0 += 8) {
            float buf[8][8];
            for (size_t dr = 0; dr < 8; ++dr) {
                const float* srow = src + static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + f0;
                std::memcpy(buf[dr], srow, 8 * sizeof(float));
            }
            for (size_t df = 0; df < 8; ++df) {
                float* dcol = dst_col_major + (f0 + df) * kTileStride + r0;
                for (size_t dr = 0; dr < 8; ++dr) {
                    dcol[dr] = buf[dr][df];
                }
            }
        }
        for (; f0 < n_features; ++f0) {
            float* dcol = dst_col_major + f0 * kTileStride + r0;
            for (size_t dr = 0; dr < 8; ++dr) {
                dcol[dr] = src[static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + static_cast<ptrdiff_t>(f0)];
            }
        }
    }

    // Remainder rows [r0, n_rows) and zero padding [n_rows, padded_rows)
    if (r0 < padded_rows) {
        for (size_t f = 0; f < n_features; ++f) {
            float* dcol = dst_col_major + f * kTileStride;
            for (size_t r = r0; r < n_rows; ++r) {
                dcol[r] = src[static_cast<ptrdiff_t>(r) * row_stride_elems + static_cast<ptrdiff_t>(f)];
            }
            for (size_t r = n_rows; r < padded_rows; ++r) {
                dcol[r] = 0.0f;
            }
        }
    }
}

void PackTileRowMajorF64_Scalar(
    const double* src,
    size_t n_rows,
    size_t n_features,
    ptrdiff_t row_stride_elems,
    float* dst_col_major
) noexcept {
    const size_t padded_rows = (n_rows + 15u) & ~size_t(15u);

    size_t r0 = 0;
    for (; r0 + 8 <= n_rows; r0 += 8) {
        size_t f0 = 0;
        for (; f0 + 8 <= n_features; f0 += 8) {
            float buf[8][8];
            for (size_t dr = 0; dr < 8; ++dr) {
                const double* srow = src + static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + f0;
                for (size_t df = 0; df < 8; ++df) {
                    buf[dr][df] = static_cast<float>(srow[df]);
                }
            }
            for (size_t df = 0; df < 8; ++df) {
                float* dcol = dst_col_major + (f0 + df) * kTileStride + r0;
                for (size_t dr = 0; dr < 8; ++dr) {
                    dcol[dr] = buf[dr][df];
                }
            }
        }
        for (; f0 < n_features; ++f0) {
            float* dcol = dst_col_major + f0 * kTileStride + r0;
            for (size_t dr = 0; dr < 8; ++dr) {
                dcol[dr] = static_cast<float>(
                    src[static_cast<ptrdiff_t>(r0 + dr) * row_stride_elems + static_cast<ptrdiff_t>(f0)]
                );
            }
        }
    }

    if (r0 < padded_rows) {
        for (size_t f = 0; f < n_features; ++f) {
            float* dcol = dst_col_major + f * kTileStride;
            for (size_t r = r0; r < n_rows; ++r) {
                dcol[r] = static_cast<float>(
                    src[static_cast<ptrdiff_t>(r) * row_stride_elems + static_cast<ptrdiff_t>(f)]
                );
            }
            for (size_t r = n_rows; r < padded_rows; ++r) {
                dcol[r] = 0.0f;
            }
        }
    }
}

namespace {

template <typename SrcT>
inline void CopyCastColumnSlice(
    const void* raw_ptr,
    size_t row_start,
    size_t n_rows,
    size_t padded_rows,
    float* dcol
) noexcept {
    const SrcT* s = static_cast<const SrcT*>(raw_ptr) + row_start;
    for (size_t r = 0; r < n_rows; ++r) {
        dcol[r] = static_cast<float>(s[r]);
    }
    for (size_t r = n_rows; r < padded_rows; ++r) {
        dcol[r] = 0.0f;
    }
}

} // namespace

void PackTileColumnar_Scalar(
    const ColumnView* cols,
    size_t row_start,
    size_t n_rows,
    size_t n_features,
    float* dst_col_major
) noexcept {
    const size_t padded_rows = (n_rows + 15u) & ~size_t(15u);

    for (size_t f = 0; f < n_features; ++f) {
        float* dcol = dst_col_major + f * kTileStride;
        const ColumnView& cv = cols[f];
        switch (cv.dtype) {
            case ColumnDType::Float32: {
                const float* s = static_cast<const float*>(cv.data) + row_start;
                std::memcpy(dcol, s, n_rows * sizeof(float));
                for (size_t r = n_rows; r < padded_rows; ++r) dcol[r] = 0.0f;
                break;
            }
            case ColumnDType::Float64:
                CopyCastColumnSlice<double>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int32:
                CopyCastColumnSlice<int32_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int64:
                CopyCastColumnSlice<int64_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int16:
                CopyCastColumnSlice<int16_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::Int8:
                CopyCastColumnSlice<int8_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::UInt8:
                CopyCastColumnSlice<uint8_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::UInt16:
                CopyCastColumnSlice<uint16_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
            case ColumnDType::UInt32:
                CopyCastColumnSlice<uint32_t>(cv.data, row_start, n_rows, padded_rows, dcol);
                break;
        }
    }
}

void BinTileFeatures_Scalar(
    const CompiledModel& model,
    const float* tile_col_major,
    size_t n_rows,
    uint16_t* dst_tile_bins
) noexcept {
    const size_t n_features = static_cast<size_t>(model.num_features);
    const size_t padded_rows = (n_rows + 15u) & ~size_t(15u);
    const uint32_t* bin_starts = model.feature_bin_starts.data();
    const float* all_th = model.shared_bin_thresholds.data();
    const uint8_t* has_zm = model.feature_has_zero_missing.data();

    for (size_t f = 0; f < n_features; ++f) {
        const uint32_t beg = bin_starts[f];
        const uint32_t end = bin_starts[f + 1];
        const uint32_t u_cnt = end - beg;
        uint16_t* dbins = dst_tile_bins + f * kTileStride;

        if (u_cnt == 0) {
            std::memset(dbins, 0, padded_rows * sizeof(uint16_t));
            continue;
        }

        const float* sorted_th = all_th + beg;
        const float first_th = sorted_th[0];
        const bool zm = (has_zm[f] != 0);
        const float* fx = tile_col_major + f * kTileStride;

        for (size_t r = 0; r < n_rows; ++r) {
            const float x = fx[r];
            if (std::isnan(x)) {
                dbins[r] = 0xFFFFu;
                continue;
            }
            if (zm && IsZeroF32(x)) {
                dbins[r] = 0xFFFEu;
                continue;
            }
            if (x <= first_th) {
                dbins[r] = 0u;
                continue;
            }
            // Branchless monobound binary search: exact count of elements < x
            const float* base = sorted_th;
            uint32_t n = u_cnt;
            while (n > 1u) {
                const uint32_t half = n >> 1;
                base += (base[half - 1u] < x) ? half : 0u;
                n -= half;
            }
            const uint32_t rank = static_cast<uint32_t>(base - sorted_th) + static_cast<uint32_t>(base[0] < x);
            dbins[r] = static_cast<uint16_t>(rank);
        }

        const uint16_t pad_val = (n_rows > 0 && dbins[n_rows - 1] < 0xFFFEu) ? dbins[n_rows - 1] : 0u;
        for (size_t r = n_rows; r < padded_rows; ++r) {
            dbins[r] = pad_val;
        }
    }
}

// ============================================================================
// Scalar Kernels (Pre-binned O(1) jump-table fast path + float fallback)
// ============================================================================

namespace {

void EvalQS32Block_Scalar(
    const QS32Block& blk,
    const float* tile_col_major,
    const uint16_t* tile_bins,
    size_t actual_rows,
    float* tile_acc
) noexcept {
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks = blk.masks.data();
    const QS32SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS32SpecialEntry* zero_entries = blk.zero_entries.data();
    const uint32_t stride = blk.leaf_stride;
    const float* leaves_base = blk.leaf_values.data();

    // Fast path: Pre-binned single-row L1d-resident execution (512B mask array, 0 float compares, 50% splits!)
    if (tile_bins != nullptr) {
        alignas(64) uint32_t v_row[128];
        const uint16_t* bin_offsets = blk.bin_offsets.data();

        for (size_t r = 0; r < actual_rows; ++r) {
            std::fill_n(v_row, n_trees, 0xFFFFFFFFu);

            for (const QS32FeatureSpan& span : blk.features) {
                const uint16_t b = tile_bins[(static_cast<size_t>(span.feature_idx) << 7) + r];
                if (b < 0xFFFEu) {
                    const uint32_t end = bin_offsets[span.bin_offset_start + b];
                    for (uint32_t idx = span.split_begin; idx < end; ++idx) {
                        v_row[tree_ids[idx]] &= masks[idx];
                    }
                } else if (b == 0xFFFFu) {
                    for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                        v_row[nan_entries[ni].tree_in_block] &= nan_entries[ni].mask;
                    }
                } else {
                    for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                        v_row[zero_entries[zi].tree_in_block] &= zero_entries[zi].mask;
                    }
                }
            }

            if (blk.single_class_id >= 0) {
                float sum = 0.0f;
                for (uint32_t t = 0; t < n_trees; ++t) {
                    sum += leaves_base[static_cast<size_t>(t) * stride + Ctz32(v_row[t])];
                }
                tile_acc[static_cast<size_t>(blk.single_class_id) * kTileStride + r] += sum;
            } else {
                for (uint32_t t = 0; t < n_trees; ++t) {
                    const float val = leaves_base[static_cast<size_t>(t) * stride + Ctz32(v_row[t])];
                    tile_acc[static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r] += val;
                }
            }
        }
        return;
    }

    constexpr size_t W = 16; // 16x uint32_t = 64 bytes = 1 exact cache line per tree!
    const size_t n_rows = (actual_rows + 15u) & ~size_t(15u);
    alignas(64) uint32_t v[128][W];
    const float* thresholds = blk.thresholds.data();

    auto count_lt = [](const float* sorted, uint32_t len, float val) noexcept -> uint32_t {
        if (len == 0u || !(sorted[0] < val)) return 0u;
        if (sorted[len - 1u] < val) return len;
        const float* base = sorted;
        uint32_t n = len;
        while (n > 1u) {
            const uint32_t half = n >> 1;
            base += (base[half - 1u] < val) ? half : 0u;
            n -= half;
        }
        return static_cast<uint32_t>(base - sorted) + static_cast<uint32_t>(base[0] < val);
    };

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            for (size_t l = 0; l < W; ++l) {
                v[t][l] = 0xFFFFFFFFu;
            }
        }

        for (const QS32FeatureSpan& span : blk.features) {
            const float* fx = tile_col_major + static_cast<size_t>(span.feature_idx) * kTileStride + r0;

            uint32_t nan_lanes = 0;
            uint32_t zero_lanes = 0;
            alignas(64) float x_clean[W];
            float x_max = -std::numeric_limits<float>::infinity();
            float x_min = std::numeric_limits<float>::infinity();

            if (!span.has_zero_missing) {
                for (size_t l = 0; l < W; ++l) {
                    const float val = fx[l];
                    if (std::isnan(val)) {
                        nan_lanes |= (1u << l);
                        x_clean[l] = -std::numeric_limits<float>::infinity();
                    } else {
                        x_clean[l] = val;
                        if (val > x_max) x_max = val;
                        if (val < x_min) x_min = val;
                    }
                }
            } else {
                for (size_t l = 0; l < W; ++l) {
                    const float val = fx[l];
                    if (std::isnan(val)) {
                        nan_lanes |= (1u << l);
                        x_clean[l] = -std::numeric_limits<float>::infinity();
                    } else if (IsZeroF32(val)) {
                        zero_lanes |= (1u << l);
                        x_clean[l] = -std::numeric_limits<float>::infinity();
                    } else {
                        x_clean[l] = val;
                        if (val > x_max) x_max = val;
                        if (val < x_min) x_min = val;
                    }
                }
            }

            uint32_t idx = span.split_begin;
            const uint32_t len = span.split_end - idx;
            const uint32_t idx_stop = idx + count_lt(thresholds + idx, len, x_max);

            if (idx < idx_stop) {
                if ((nan_lanes | zero_lanes) == 0u) {
                    const uint32_t idx_all = idx + count_lt(thresholds + idx, idx_stop - idx, x_min);
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
                    const float vt = thresholds[idx];
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
            alignas(64) float acc_local[W] = {0};
            for (uint32_t t = 0; t < n_trees; ++t) {
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                for (size_t l = 0; l < W; ++l) {
                    acc_local[l] += tl[Ctz32(v[t][l])];
                }
            }
            float* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            for (size_t l = 0; l < W; ++l) {
                dst[l] += acc_local[l];
            }
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                float* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                for (size_t l = 0; l < W; ++l) {
                    dst[l] += tl[Ctz32(v[t][l])];
                }
            }
        }
    }
}

void EvalQS64Block_Scalar(
    const QS64Block& blk,
    const float* tile_col_major,
    const uint16_t* tile_bins,
    size_t actual_rows,
    float* tile_acc
) noexcept {
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks_lo = blk.masks_lo.data();
    const uint32_t* masks_hi = blk.masks_hi.data();
    const QS64SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS64SpecialEntry* zero_entries = blk.zero_entries.data();
    const uint32_t stride = blk.leaf_stride;
    const float* leaves_base = blk.leaf_values.data();

    if (tile_bins != nullptr) {
        alignas(64) uint64_t v_row[64];
        const uint16_t* bin_offsets = blk.bin_offsets.data();

        for (size_t r = 0; r < actual_rows; ++r) {
            std::fill_n(v_row, n_trees, ~0ULL);

            for (const QS64FeatureSpan& span : blk.features) {
                const uint16_t b = tile_bins[(static_cast<size_t>(span.feature_idx) << 7) + r];
                if (b < 0xFFFEu) {
                    const uint32_t end = bin_offsets[span.bin_offset_start + b];
                    for (uint32_t idx = span.split_begin; idx < end; ++idx) {
                        const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                        v_row[tree_ids[idx]] &= m64;
                    }
                } else if (b == 0xFFFFu) {
                    for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                        const uint64_t m64 = (static_cast<uint64_t>(nan_entries[ni].mask_hi) << 32) | nan_entries[ni].mask_lo;
                        v_row[nan_entries[ni].tree_in_block] &= m64;
                    }
                } else {
                    for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                        const uint64_t m64 = (static_cast<uint64_t>(zero_entries[zi].mask_hi) << 32) | zero_entries[zi].mask_lo;
                        v_row[zero_entries[zi].tree_in_block] &= m64;
                    }
                }
            }

            if (blk.single_class_id >= 0) {
                float sum = 0.0f;
                for (uint32_t t = 0; t < n_trees; ++t) {
                    sum += leaves_base[static_cast<size_t>(t) * stride + Ctz64(v_row[t])];
                }
                tile_acc[static_cast<size_t>(blk.single_class_id) * kTileStride + r] += sum;
            } else {
                for (uint32_t t = 0; t < n_trees; ++t) {
                    const float val = leaves_base[static_cast<size_t>(t) * stride + Ctz64(v_row[t])];
                    tile_acc[static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r] += val;
                }
            }
        }
        return;
    }

    constexpr size_t W = 8;
    const size_t n_rows = (actual_rows + 7u) & ~size_t(7u);
    alignas(64) uint64_t v[64][W];
    const float* thresholds = blk.thresholds.data();

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            for (size_t l = 0; l < W; ++l) {
                v[t][l] = ~0ULL;
            }
        }

        for (const QS64FeatureSpan& span : blk.features) {
            const float* fx = tile_col_major + static_cast<size_t>(span.feature_idx) * kTileStride + r0;

            uint32_t nan_lanes = 0;
            uint32_t zero_lanes = 0;
            alignas(32) float x_clean[W];
            float x_max = -std::numeric_limits<float>::infinity();

            for (size_t l = 0; l < W; ++l) {
                const float val = fx[l];
                if (std::isnan(val)) {
                    nan_lanes |= (1u << l);
                    x_clean[l] = -std::numeric_limits<float>::infinity();
                } else if (span.has_zero_missing && IsZeroF32(val)) {
                    zero_lanes |= (1u << l);
                    x_clean[l] = -std::numeric_limits<float>::infinity();
                } else {
                    x_clean[l] = val;
                    if (val > x_max) x_max = val;
                }
            }

            if (x_max > thresholds[span.split_begin]) {
                for (uint32_t idx = span.split_begin; idx < span.split_end; ++idx) {
                    const float vt = thresholds[idx];
                    if (vt >= x_max) break;
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
                for (uint32_t idx = span.nan_begin; idx < span.nan_end; ++idx) {
                    const uint16_t tid = nan_entries[idx].tree_in_block;
                    const uint64_t m64 = (static_cast<uint64_t>(nan_entries[idx].mask_hi) << 32) | nan_entries[idx].mask_lo;
                    uint64_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        if ((nan_lanes & (1u << l)) != 0) vt_arr[l] &= m64;
                    }
                }
            }

            if (zero_lanes != 0) {
                for (uint32_t idx = span.zero_begin; idx < span.zero_end; ++idx) {
                    const uint16_t tid = zero_entries[idx].tree_in_block;
                    const uint64_t m64 = (static_cast<uint64_t>(zero_entries[idx].mask_hi) << 32) | zero_entries[idx].mask_lo;
                    uint64_t* vt_arr = v[tid];
                    for (size_t l = 0; l < W; ++l) {
                        if ((zero_lanes & (1u << l)) != 0) vt_arr[l] &= m64;
                    }
                }
            }
        }

        const uint32_t stride = blk.leaf_stride;
        const float* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            alignas(32) float acc_local[W] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t t = 0; t < n_trees; ++t) {
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                for (size_t l = 0; l < W; ++l) {
                    acc_local[l] += tl[Ctz64(v[t][l])];
                }
            }
            float* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            for (size_t l = 0; l < W; ++l) {
                dst[l] += acc_local[l];
            }
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                float* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
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
    const float* all_leaves,
    const float* tile_col_major,
    size_t n_rows,
    float* tile_acc
) noexcept {
    for (size_t t = 0; t < num_trees; ++t) {
        const CompactTreeMeta& meta = trees[t];
        const CompactNode* nodes = all_nodes + meta.node_offset;
        const float* leaves = all_leaves + meta.leaf_offset;
        float* dst = tile_acc + static_cast<size_t>(meta.class_id) * kTileStride;

        if (!meta.has_zero_missing) {
            for (size_t r = 0; r < n_rows; ++r) {
                int32_t idx = 0;
                while (idx >= 0) {
                    const CompactNode& n = nodes[idx];
                    const float x = tile_col_major[(static_cast<size_t>(n.feature_idx()) << 7) + r];
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
                    const float x = tile_col_major[(static_cast<size_t>(n.feature_idx()) << 7) + r];
                    bool go_left;
                    if (std::isnan(x)) {
                        go_left = n.nan_goes_left();
                    } else if (n.has_zero_missing() && IsZeroF32(x)) {
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

void FinalizeTileOutput_Scalar(
    float* tile_acc,
    size_t n_rows,
    int num_outputs,
    ObjectiveType objective,
    float sigmoid_alpha,
    float scale_factor,
    bool raw_score,
    float* out_row_major
) noexcept {
    const bool need_scale = (scale_factor != 1.0f);

    if (num_outputs == 1) {
        float* src = tile_acc;
        if (raw_score || objective == ObjectiveType::Regression) {
            if (need_scale) {
                for (size_t r = 0; r < n_rows; ++r) out_row_major[r] = src[r] * scale_factor;
            } else {
                std::memcpy(out_row_major, src, n_rows * sizeof(float));
            }
        } else if (objective == ObjectiveType::BinaryLogloss) {
            for (size_t r = 0; r < n_rows; ++r) {
                const float s = need_scale ? (src[r] * scale_factor) : src[r];
                out_row_major[r] = simd_math::SafeSigmoidScalar(s, sigmoid_alpha);
            }
        } else if (objective == ObjectiveType::Exponential) {
            for (size_t r = 0; r < n_rows; ++r) {
                const float s = need_scale ? (src[r] * scale_factor) : src[r];
                out_row_major[r] = simd_math::SafeExpScalar(s);
            }
        }
        return;
    }

    // Multi-output (K > 1)
    const size_t K = static_cast<size_t>(num_outputs);
    if (raw_score || objective == ObjectiveType::Regression) {
        for (size_t r = 0; r < n_rows; ++r) {
            float* drow = out_row_major + r * K;
            for (size_t c = 0; c < K; ++c) {
                const float s = tile_acc[c * kTileStride + r];
                drow[c] = need_scale ? (s * scale_factor) : s;
            }
        }
    } else if (objective == ObjectiveType::BinaryLogloss) {
        // multiclassova
        for (size_t r = 0; r < n_rows; ++r) {
            float* drow = out_row_major + r * K;
            for (size_t c = 0; c < K; ++c) {
                const float s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                           : tile_acc[c * kTileStride + r];
                drow[c] = simd_math::SafeSigmoidScalar(s, sigmoid_alpha);
            }
        }
    } else if (objective == ObjectiveType::MulticlassSoftmax) {
        for (size_t r = 0; r < n_rows; ++r) {
            float* drow = out_row_major + r * K;
            double max_v = -std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < K; ++c) {
                const double s = static_cast<double>(
                    need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                               : tile_acc[c * kTileStride + r]
                );
                if (s > max_v) max_v = s;
            }
            double sum_e = 0.0;
            for (size_t c = 0; c < K; ++c) {
                const double s = static_cast<double>(
                    need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                               : tile_acc[c * kTileStride + r]
                );
                const double e = std::exp(s - max_v);
                drow[c] = static_cast<float>(e);
                sum_e += e;
            }
            const float inv = static_cast<float>(1.0 / sum_e);
            for (size_t c = 0; c < K; ++c) {
                drow[c] *= inv;
            }
        }
    } else if (objective == ObjectiveType::Exponential) {
        for (size_t r = 0; r < n_rows; ++r) {
            float* drow = out_row_major + r * K;
            for (size_t c = 0; c < K; ++c) {
                const float s = need_scale ? (tile_acc[c * kTileStride + r] * scale_factor)
                                           : tile_acc[c * kTileStride + r];
                drow[c] = simd_math::SafeExpScalar(s);
            }
        }
    }
}

} // namespace

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
