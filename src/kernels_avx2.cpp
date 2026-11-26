#include "fast_scorer/kernels.h"
#include "fast_scorer/math_simd.h"

#if defined(FAST_SCORER_ARCH_X86_64)

#include <immintrin.h>
#include <cstring>
#include <limits>

namespace fast_scorer {

namespace {

inline double HorizontalMaxPd256(__m256d v) noexcept {
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d m1 = _mm_max_pd(lo, hi);
    __m128d m2 = _mm_max_sd(m1, _mm_unpackhi_pd(m1, m1));
    return _mm_cvtsd_f64(m2);
}

inline double HorizontalMinPd256(__m256d v) noexcept {
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d m1 = _mm_min_pd(lo, hi);
    __m128d m2 = _mm_min_sd(m1, _mm_unpackhi_pd(m1, m1));
    return _mm_cvtsd_f64(m2);
}

// Single-instruction pack of two 4x64-bit masks c0=[m0,m1,m2,m3] and c1=[m4,m5,m6,m7]
// into 8x32-bit lanes in swizzled order [m0, m1, m4, m5, m2, m3, m6, m7].
inline __m256i PackMaskPd2x4Swizzled(__m256d c0, __m256d c1) noexcept {
    return _mm256_castps_si256(
        _mm256_shuffle_ps(_mm256_castpd_ps(c0), _mm256_castpd_ps(c1), _MM_SHUFFLE(2, 0, 2, 0))
    );
}

// Pack a single 4x64-bit mask [m0, m1, m2, m3] into 4x32-bit __m128i [m0, m1, m2, m3]
inline __m128i PackMaskPd4ToEpi32(__m256d c) noexcept {
    const __m128 lo = _mm_castpd_ps(_mm256_castpd256_pd128(c));
    const __m128 hi = _mm_castpd_ps(_mm256_extractf128_pd(c, 1));
    return _mm_castps_si128(_mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0)));
}

// Branchless 8-lane parallel 32-bit trailing zero count via IEEE-754 float32 exponent
inline __m256i Ctz32Avx2(__m256i v) noexcept {
    __m256i neg_v = _mm256_sub_epi32(_mm256_setzero_si256(), v);
    __m256i lsb = _mm256_and_si256(v, neg_v);
    __m256i flt_bits = _mm256_castps_si256(_mm256_cvtepi32_ps(lsb));
    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(flt_bits, 23), _mm256_set1_epi32(0xFF));
    return _mm256_sub_epi32(exp, _mm256_set1_epi32(127));
}

inline __m256i Ctz64SplitAvx2(__m256i vlo, __m256i vhi) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    __m256i lo_is_zero = _mm256_cmpeq_epi32(vlo, zero);
    __m256i active = _mm256_blendv_epi8(vlo, vhi, lo_is_zero);
    __m256i tz = Ctz32Avx2(active);
    __m256i add32 = _mm256_and_si256(lo_is_zero, _mm256_set1_epi32(32));
    return _mm256_add_epi32(tz, add32);
}

// Branchless monobound search: exact count of elements strictly < val
inline uint32_t CountStrictlyLessAvx2(const double* sorted, uint32_t len, double val) noexcept {
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

void EvalQS32Block_Avx2(
    const QS32Block& blk,
    const double* tile_col_major,
    size_t actual_rows,
    double* tile_acc
) noexcept {
    constexpr size_t W = 8; // 8 rows: two __m256d feature loads + single __m256i 8x32-bit mask per tree
    const size_t n_rows = (actual_rows + 7u) & ~size_t(7u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(64) __m256i v[128];
    const double* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks = blk.masks.data();
    const QS32SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS32SpecialEntry* zero_entries = blk.zero_entries.data();

    const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d zero_eps = _mm256_set1_pd(kZeroThreshold);
    const __m256i all_ones = _mm256_set1_epi32(-1);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v[t] = all_ones;
        }

        for (const QS32FeatureSpan& span : blk.features) {
            const double* fx_ptr = tile_col_major + (static_cast<size_t>(span.feature_idx) << 7) + r0;
            const __m256d vx0 = _mm256_load_pd(fx_ptr);
            const __m256d vx1 = _mm256_load_pd(fx_ptr + 4);

            const __m256d nan0 = _mm256_cmp_pd(vx0, vx0, _CMP_UNORD_Q);
            const __m256d nan1 = _mm256_cmp_pd(vx1, vx1, _CMP_UNORD_Q);
            const int nan_bits = _mm256_movemask_pd(nan0) | (_mm256_movemask_pd(nan1) << 4);

            __m256d zero0 = _mm256_setzero_pd();
            __m256d zero1 = _mm256_setzero_pd();
            int zero_bits = 0;
            if (span.has_zero_missing) {
                zero0 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, vx0), zero_eps, _CMP_LE_OQ);
                zero1 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, vx1), zero_eps, _CMP_LE_OQ);
                zero_bits = _mm256_movemask_pd(zero0) | (_mm256_movemask_pd(zero1) << 4);
            }

            const __m256d vx0_clean = (nan_bits | zero_bits)
                                          ? _mm256_blendv_pd(vx0, neg_inf, _mm256_or_pd(nan0, zero0))
                                          : vx0;
            const __m256d vx1_clean = (nan_bits | zero_bits)
                                          ? _mm256_blendv_pd(vx1, neg_inf, _mm256_or_pd(nan1, zero1))
                                          : vx1;

            const double x_max = HorizontalMaxPd256(_mm256_max_pd(vx0_clean, vx1_clean));
            uint32_t idx = span.split_begin;
            const uint32_t len = span.split_end - idx;
            const uint32_t idx_stop = idx + CountStrictlyLessAvx2(thresholds + idx, len, x_max);

            if (idx < idx_stop) {
                if ((nan_bits | zero_bits) == 0) {
                    const double x_min = HorizontalMinPd256(_mm256_min_pd(vx0_clean, vx1_clean));
                    const uint32_t idx_all = idx + CountStrictlyLessAvx2(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(masks[idx]));
                        v[tid] = _mm256_and_si256(v[tid], vm);
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    const __m256d vt = _mm256_broadcast_sd(&thresholds[idx]);
                    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(masks[idx]));
                    const __m256d le0 = _mm256_cmp_pd(vx0_clean, vt, _CMP_LE_OQ);
                    const __m256d le1 = _mm256_cmp_pd(vx1_clean, vt, _CMP_LE_OQ);
                    const __m256i le_sw = PackMaskPd2x4Swizzled(le0, le1);
                    const uint16_t tid = tree_ids[idx];
                    v[tid] = _mm256_and_si256(v[tid], _mm256_or_si256(le_sw, vm));
                }
            }

            if (nan_bits != 0) {
                const __m256i not_nan_sw = _mm256_andnot_si256(PackMaskPd2x4Swizzled(nan0, nan1), all_ones);
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(nan_entries[ni].mask));
                    v[tid] = _mm256_and_si256(v[tid], _mm256_or_si256(not_nan_sw, vm));
                }
            }

            if (zero_bits != 0) {
                const __m256i not_z_sw = _mm256_andnot_si256(PackMaskPd2x4Swizzled(zero0, zero1), all_ones);
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(zero_entries[zi].mask));
                    v[tid] = _mm256_and_si256(v[tid], _mm256_or_si256(not_z_sw, vm));
                }
            }
        }

        const uint32_t stride = blk.leaf_stride;
        const double* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m256d acc_sw0 = _mm256_setzero_pd(); // [r0, r1, r4, r5]
            __m256d acc_sw1 = _mm256_setzero_pd(); // [r2, r3, r6, r7]
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_sw = Ctz32Avx2(v[t]);
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                acc_sw0 = _mm256_add_pd(acc_sw0, _mm256_i32gather_pd(tl, _mm256_castsi256_si128(leaf_sw), 8));
                acc_sw1 = _mm256_add_pd(acc_sw1, _mm256_i32gather_pd(tl, _mm256_extracti128_si256(leaf_sw, 1), 8));
            }
            const __m256d acc0 = _mm256_permute2f128_pd(acc_sw0, acc_sw1, 0x20); // [r0, r1, r2, r3]
            const __m256d acc1 = _mm256_permute2f128_pd(acc_sw0, acc_sw1, 0x31); // [r4, r5, r6, r7]
            double* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(dst), acc0));
            _mm256_store_pd(dst + 4, _mm256_add_pd(_mm256_load_pd(dst + 4), acc1));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i vt_ord = _mm256_permute4x64_epi64(v[t], _MM_SHUFFLE(3, 1, 2, 0));
                const __m256i leaf_ord = Ctz32Avx2(vt_ord);
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m256d vals0 = _mm256_i32gather_pd(tl, _mm256_castsi256_si128(leaf_ord), 8);
                const __m256d vals1 = _mm256_i32gather_pd(tl, _mm256_extracti128_si256(leaf_ord, 1), 8);
                double* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(dst), vals0));
                _mm256_store_pd(dst + 4, _mm256_add_pd(_mm256_load_pd(dst + 4), vals1));
            }
        }
    }
}

void EvalQS64Block_Avx2(
    const QS64Block& blk,
    const double* tile_col_major,
    size_t actual_rows,
    double* tile_acc
) noexcept {
    constexpr size_t W = 8;
    const size_t n_rows = (actual_rows + 7u) & ~size_t(7u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(64) __m256i v_lo[64];
    alignas(64) __m256i v_hi[64];

    const double* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks_lo = blk.masks_lo.data();
    const uint32_t* masks_hi = blk.masks_hi.data();
    const QS64SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS64SpecialEntry* zero_entries = blk.zero_entries.data();

    const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d zero_eps = _mm256_set1_pd(kZeroThreshold);
    const __m256i all_ones = _mm256_set1_epi32(-1);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v_lo[t] = all_ones;
            v_hi[t] = all_ones;
        }

        for (const QS64FeatureSpan& span : blk.features) {
            const double* fx_ptr = tile_col_major + (static_cast<size_t>(span.feature_idx) << 7) + r0;
            const __m256d vx0 = _mm256_load_pd(fx_ptr);
            const __m256d vx1 = _mm256_load_pd(fx_ptr + 4);

            const __m256d nan0 = _mm256_cmp_pd(vx0, vx0, _CMP_UNORD_Q);
            const __m256d nan1 = _mm256_cmp_pd(vx1, vx1, _CMP_UNORD_Q);
            const int nan_bits = _mm256_movemask_pd(nan0) | (_mm256_movemask_pd(nan1) << 4);

            __m256d zero0 = _mm256_setzero_pd();
            __m256d zero1 = _mm256_setzero_pd();
            int zero_bits = 0;
            if (span.has_zero_missing) {
                zero0 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, vx0), zero_eps, _CMP_LE_OQ);
                zero1 = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, vx1), zero_eps, _CMP_LE_OQ);
                zero_bits = _mm256_movemask_pd(zero0) | (_mm256_movemask_pd(zero1) << 4);
            }

            const __m256d vx0_clean = (nan_bits | zero_bits)
                                          ? _mm256_blendv_pd(vx0, neg_inf, _mm256_or_pd(nan0, zero0))
                                          : vx0;
            const __m256d vx1_clean = (nan_bits | zero_bits)
                                          ? _mm256_blendv_pd(vx1, neg_inf, _mm256_or_pd(nan1, zero1))
                                          : vx1;

            const double x_max = HorizontalMaxPd256(_mm256_max_pd(vx0_clean, vx1_clean));
            uint32_t idx = span.split_begin;
            const uint32_t len = span.split_end - idx;
            const uint32_t idx_stop = idx + CountStrictlyLessAvx2(thresholds + idx, len, x_max);

            if (idx < idx_stop) {
                if ((nan_bits | zero_bits) == 0) {
                    const double x_min = HorizontalMinPd256(_mm256_min_pd(vx0_clean, vx1_clean));
                    const uint32_t idx_all = idx + CountStrictlyLessAvx2(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        v_lo[tid] = _mm256_and_si256(
                            v_lo[tid], _mm256_set1_epi32(static_cast<int32_t>(masks_lo[idx]))
                        );
                        v_hi[tid] = _mm256_and_si256(
                            v_hi[tid], _mm256_set1_epi32(static_cast<int32_t>(masks_hi[idx]))
                        );
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    const __m256d vt = _mm256_broadcast_sd(&thresholds[idx]);
                    const __m256d le0 = _mm256_cmp_pd(vx0_clean, vt, _CMP_LE_OQ);
                    const __m256d le1 = _mm256_cmp_pd(vx1_clean, vt, _CMP_LE_OQ);
                    const __m256i le_sw = PackMaskPd2x4Swizzled(le0, le1);
                    const uint16_t tid = tree_ids[idx];
                    v_lo[tid] = _mm256_and_si256(
                        v_lo[tid],
                        _mm256_or_si256(le_sw, _mm256_set1_epi32(static_cast<int32_t>(masks_lo[idx])))
                    );
                    v_hi[tid] = _mm256_and_si256(
                        v_hi[tid],
                        _mm256_or_si256(le_sw, _mm256_set1_epi32(static_cast<int32_t>(masks_hi[idx])))
                    );
                }
            }

            if (nan_bits != 0) {
                const __m256i not_nan_sw = _mm256_andnot_si256(PackMaskPd2x4Swizzled(nan0, nan1), all_ones);
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    v_lo[tid] = _mm256_and_si256(
                        v_lo[tid],
                        _mm256_or_si256(not_nan_sw, _mm256_set1_epi32(static_cast<int32_t>(nan_entries[ni].mask_lo)))
                    );
                    v_hi[tid] = _mm256_and_si256(
                        v_hi[tid],
                        _mm256_or_si256(not_nan_sw, _mm256_set1_epi32(static_cast<int32_t>(nan_entries[ni].mask_hi)))
                    );
                }
            }

            if (zero_bits != 0) {
                const __m256i not_z_sw = _mm256_andnot_si256(PackMaskPd2x4Swizzled(zero0, zero1), all_ones);
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    v_lo[tid] = _mm256_and_si256(
                        v_lo[tid],
                        _mm256_or_si256(not_z_sw, _mm256_set1_epi32(static_cast<int32_t>(zero_entries[zi].mask_lo)))
                    );
                    v_hi[tid] = _mm256_and_si256(
                        v_hi[tid],
                        _mm256_or_si256(not_z_sw, _mm256_set1_epi32(static_cast<int32_t>(zero_entries[zi].mask_hi)))
                    );
                }
            }
        }

        const uint32_t stride = blk.leaf_stride;
        const double* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m256d acc_sw0 = _mm256_setzero_pd();
            __m256d acc_sw1 = _mm256_setzero_pd();
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_sw = Ctz64SplitAvx2(v_lo[t], v_hi[t]);
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                acc_sw0 = _mm256_add_pd(acc_sw0, _mm256_i32gather_pd(tl, _mm256_castsi256_si128(leaf_sw), 8));
                acc_sw1 = _mm256_add_pd(acc_sw1, _mm256_i32gather_pd(tl, _mm256_extracti128_si256(leaf_sw, 1), 8));
            }
            const __m256d acc0 = _mm256_permute2f128_pd(acc_sw0, acc_sw1, 0x20);
            const __m256d acc1 = _mm256_permute2f128_pd(acc_sw0, acc_sw1, 0x31);
            double* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(dst), acc0));
            _mm256_store_pd(dst + 4, _mm256_add_pd(_mm256_load_pd(dst + 4), acc1));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_sw = Ctz64SplitAvx2(v_lo[t], v_hi[t]);
                const __m256i leaf_ord = _mm256_permute4x64_epi64(leaf_sw, _MM_SHUFFLE(3, 1, 2, 0));
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m256d vals0 = _mm256_i32gather_pd(tl, _mm256_castsi256_si128(leaf_ord), 8);
                const __m256d vals1 = _mm256_i32gather_pd(tl, _mm256_extracti128_si256(leaf_ord, 1), 8);
                double* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(dst), vals0));
                _mm256_store_pd(dst + 4, _mm256_add_pd(_mm256_load_pd(dst + 4), vals1));
            }
        }
    }
}

void EvalCompactTrees_Avx2(
    const CompactTreeMeta* trees,
    size_t num_trees,
    const CompactNode* all_nodes,
    const double* all_leaves,
    const double* tile_col_major,
    size_t n_rows,
    double* tile_acc
) noexcept {
    constexpr size_t W = 4; // 4x double in __m256d + 4x int32 in __m128i (1-to-1 width, zero half-splits!)
    const __m128i lane_ids = _mm_setr_epi32(0, 1, 2, 3);
    const __m128i minus_one = _mm_set1_epi32(-1);
    const __m128i feat_mask = _mm_set1_epi32(0x00FFFFFF);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d zero_eps = _mm256_set1_pd(kZeroThreshold);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        const __m128i lane_r0 = _mm_add_epi32(lane_ids, _mm_set1_epi32(static_cast<int32_t>(r0)));

        for (size_t t = 0; t < num_trees; ++t) {
            const CompactTreeMeta& meta = trees[t];
            const CompactNode* nodes = all_nodes + meta.node_offset;
            const double* leaves = all_leaves + meta.leaf_offset;

            // Unroll root node 0 (all 4 lanes start at root 0 -> contiguous load instead of gather!)
            const CompactNode& root = nodes[0];
            const __m256d vx0 = _mm256_load_pd(
                tile_col_major + (static_cast<size_t>(root.feature_idx()) << 7) + r0
            );
            const __m256d vt0 = _mm256_set1_pd(root.threshold);
            __m128i go_left0 = PackMaskPd4ToEpi32(_mm256_cmp_pd(vx0, vt0, _CMP_LE_OQ));
            if (root.nan_goes_left()) {
                __m128i unord0 = PackMaskPd4ToEpi32(_mm256_cmp_pd(vx0, vx0, _CMP_UNORD_Q));
                go_left0 = _mm_or_si128(go_left0, unord0);
            }
            if (root.has_zero_missing()) {
                __m256d abs_x0 = _mm256_andnot_pd(sign_mask, vx0);
                __m128i is_z0 = PackMaskPd4ToEpi32(_mm256_cmp_pd(abs_x0, zero_eps, _CMP_LE_OQ));
                __m128i z_dir0 = root.zero_goes_left() ? minus_one : _mm_setzero_si128();
                go_left0 = _mm_blendv_epi8(go_left0, z_dir0, is_z0);
            }

            __m128i v_idx = _mm_blendv_epi8(
                _mm_set1_epi32(root.right_child),
                _mm_set1_epi32(root.left_child),
                go_left0
            );

            const double* nodes_f64 = reinterpret_cast<const double*>(nodes);
            const int* nodes_i32 = reinterpret_cast<const int*>(nodes);

            if (!meta.has_zero_missing) {
                while (_mm_movemask_ps(_mm_castsi128_ps(v_idx)) != 0xF) {
                    const __m128i active = _mm_cmpgt_epi32(v_idx, minus_one);
                    const __m128i safe_idx = _mm_and_si128(v_idx, active);
                    const __m128i idx4 = _mm_slli_epi32(safe_idx, 2);
                    const __m128i idx8 = _mm_slli_epi32(safe_idx, 3);

                    const __m256d v_thresh = _mm256_i32gather_pd(nodes_f64, idx4, 8);
                    const __m128i v_ff = _mm_i32gather_epi32(nodes_i32 + 2, idx8, 4);
                    const __m128i v_left = _mm_i32gather_epi32(nodes_i32 + 3, idx8, 4);
                    const __m128i v_right = _mm_i32gather_epi32(nodes_i32 + 4, idx8, 4);

                    const __m128i v_feat = _mm_and_si128(v_ff, feat_mask);
                    const __m128i feat_off = _mm_add_epi32(_mm_slli_epi32(v_feat, 7), lane_r0);
                    const __m256d vx = _mm256_i32gather_pd(tile_col_major, feat_off, 8);

                    const __m128i le_mask = PackMaskPd4ToEpi32(_mm256_cmp_pd(vx, v_thresh, _CMP_LE_OQ));
                    const __m128i unord = PackMaskPd4ToEpi32(_mm256_cmp_pd(vx, vx, _CMP_UNORD_Q));
                    const __m128i nan_left = _mm_srai_epi32(_mm_slli_epi32(v_ff, 7), 31);
                    const __m128i go_left = _mm_or_si128(le_mask, _mm_and_si128(unord, nan_left));

                    const __m128i next_idx = _mm_blendv_epi8(v_right, v_left, go_left);
                    v_idx = _mm_blendv_epi8(v_idx, next_idx, active);
                }
            } else {
                while (_mm_movemask_ps(_mm_castsi128_ps(v_idx)) != 0xF) {
                    const __m128i active = _mm_cmpgt_epi32(v_idx, minus_one);
                    const __m128i safe_idx = _mm_and_si128(v_idx, active);
                    const __m128i idx4 = _mm_slli_epi32(safe_idx, 2);
                    const __m128i idx8 = _mm_slli_epi32(safe_idx, 3);

                    const __m256d v_thresh = _mm256_i32gather_pd(nodes_f64, idx4, 8);
                    const __m128i v_ff = _mm_i32gather_epi32(nodes_i32 + 2, idx8, 4);
                    const __m128i v_left = _mm_i32gather_epi32(nodes_i32 + 3, idx8, 4);
                    const __m128i v_right = _mm_i32gather_epi32(nodes_i32 + 4, idx8, 4);

                    const __m128i v_feat = _mm_and_si128(v_ff, feat_mask);
                    const __m128i feat_off = _mm_add_epi32(_mm_slli_epi32(v_feat, 7), lane_r0);
                    const __m256d vx = _mm256_i32gather_pd(tile_col_major, feat_off, 8);

                    const __m128i le_mask = PackMaskPd4ToEpi32(_mm256_cmp_pd(vx, v_thresh, _CMP_LE_OQ));
                    const __m128i unord = PackMaskPd4ToEpi32(_mm256_cmp_pd(vx, vx, _CMP_UNORD_Q));
                    const __m128i nan_left = _mm_srai_epi32(_mm_slli_epi32(v_ff, 7), 31);
                    __m128i go_left = _mm_or_si128(le_mask, _mm_and_si128(unord, nan_left));

                    const __m128i has_zm = _mm_srai_epi32(_mm_slli_epi32(v_ff, 6), 31);
                    const __m128i zm_left = _mm_srai_epi32(_mm_slli_epi32(v_ff, 5), 31);
                    const __m256d abs_vx = _mm256_andnot_pd(sign_mask, vx);
                    const __m128i is_zero = PackMaskPd4ToEpi32(_mm256_cmp_pd(abs_vx, zero_eps, _CMP_LE_OQ));
                    const __m128i override_z = _mm_and_si128(has_zm, is_zero);
                    go_left = _mm_blendv_epi8(go_left, zm_left, override_z);

                    const __m128i next_idx = _mm_blendv_epi8(v_right, v_left, go_left);
                    v_idx = _mm_blendv_epi8(v_idx, next_idx, active);
                }
            }

            const __m128i leaf_idx = _mm_xor_si128(v_idx, minus_one);
            const __m256d vals = _mm256_i32gather_pd(leaves, leaf_idx, 8);
            double* dst = tile_acc + static_cast<size_t>(meta.class_id) * kTileStride + r0;
            _mm256_store_pd(dst, _mm256_add_pd(_mm256_load_pd(dst), vals));
        }
    }
}

void FinalizeTileOutput_Avx2(
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
    // Fast AVX2 vectorized path for identity / raw_score single-output; exact IEEE-754 std::exp for link functions
    if (num_outputs == 1 && (raw_score || objective == ObjectiveType::Regression)) {
        const bool need_scale = (scale_factor != 1.0);
        const __m256d v_scale = _mm256_set1_pd(scale_factor);
        size_t r = 0;
        if (out_f64 != nullptr) {
            if (!need_scale) {
                std::memcpy(out_f64, tile_acc, n_rows * sizeof(double));
                return;
            }
            for (; r + 4 <= n_rows; r += 4) {
                _mm256_storeu_pd(out_f64 + r, _mm256_mul_pd(_mm256_load_pd(tile_acc + r), v_scale));
            }
            for (; r < n_rows; ++r) {
                out_f64[r] = tile_acc[r] * scale_factor;
            }
            return;
        }
        if (out_f32 != nullptr) {
            for (; r + 4 <= n_rows; r += 4) {
                __m256d v = _mm256_load_pd(tile_acc + r);
                if (need_scale) v = _mm256_mul_pd(v, v_scale);
                _mm_storeu_ps(out_f32 + r, _mm256_cvtpd_ps(v));
            }
            for (; r < n_rows; ++r) {
                const double s = need_scale ? (tile_acc[r] * scale_factor) : tile_acc[r];
                out_f32[r] = static_cast<float>(s);
            }
            return;
        }
    }

    FinalizeTileOutput_Scalar(
        tile_acc, n_rows, num_outputs, objective, sigmoid_alpha, scale_factor, raw_score, out_f64, out_f32
    );
}

} // namespace

const KernelTable& GetAvx2Kernels() noexcept {
    static const KernelTable kTable = {
        SimdIsa::Avx2,
        "avx2",
        &EvalQS32Block_Avx2,
        &EvalQS64Block_Avx2,
        &EvalCompactTrees_Avx2,
        &FinalizeTileOutput_Avx2
    };
    return kTable;
}

} // namespace fast_scorer

#endif // FAST_SCORER_ARCH_X86_64
