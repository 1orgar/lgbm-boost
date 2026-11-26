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

// Portable AVX-512F/DQ helpers compatible with older GCC 8 on Oracle Linux 8 / RHEL 8
inline __m512d AbsPd512(__m512d v) noexcept {
    return _mm512_castsi512_pd(
        _mm512_and_epi64(_mm512_castpd_si512(v), _mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL))
    );
}

inline double HorizontalMaxPd512(__m512d v) noexcept {
    const __m256d lo = _mm512_castpd512_pd256(v);
    const __m256d hi = _mm512_extractf64x4_pd(v, 1);
    return HorizontalMaxPd256(_mm256_max_pd(lo, hi));
}

inline double HorizontalMinPd512(__m512d v) noexcept {
    const __m256d lo = _mm512_castpd512_pd256(v);
    const __m256d hi = _mm512_extractf64x4_pd(v, 1);
    return HorizontalMinPd256(_mm256_min_pd(lo, hi));
}

// Hardware AVX-512CD 16-lane 32-bit trailing zero count: tzcnt32(v) = 31 - lzcnt32(v & -v)
inline __m512i Ctz32Avx512(__m512i v) noexcept {
    const __m512i lsb = _mm512_and_si512(v, _mm512_sub_epi32(_mm512_setzero_si512(), v));
    const __m512i lz = _mm512_lzcnt_epi32(lsb);
    return _mm512_sub_epi32(_mm512_set1_epi32(31), lz);
}

// Hardware AVX-512CD 8-lane 64-bit trailing zero count: tzcnt64(v) = 63 - lzcnt64(v & -v)
inline __m256i Ctz64Avx512(__m512i v) noexcept {
    const __m512i lsb = _mm512_and_si512(v, _mm512_sub_epi64(_mm512_setzero_si512(), v));
    const __m512i lz = _mm512_lzcnt_epi64(lsb);
    const __m512i tz64 = _mm512_sub_epi64(_mm512_set1_epi64(63), lz);
    return _mm512_cvtepi64_epi32(tz64);
}

inline uint32_t CountStrictlyLess512(const double* sorted, uint32_t len, double val) noexcept {
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

void EvalQS32Block_Avx512(
    const QS32Block& blk,
    const double* tile_col_major,
    size_t actual_rows,
    double* tile_acc
) noexcept {
    constexpr size_t W = 16; // 16 rows: two __m512d feature loads + single __m512i 16x32-bit mask per tree
    const size_t n_rows = (actual_rows + 15u) & ~size_t(15u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(64) __m512i v_vec[128];
    const double* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks = blk.masks.data();
    const QS32SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS32SpecialEntry* zero_entries = blk.zero_entries.data();

    const __m512d neg_inf = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d zero_eps = _mm512_set1_pd(kZeroThreshold);
    const __m512i all_ones = _mm512_set1_epi32(-1);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v_vec[t] = all_ones;
        }

        for (const QS32FeatureSpan& span : blk.features) {
            const double* fx_ptr = tile_col_major + (static_cast<size_t>(span.feature_idx) << 7) + r0;
            const __m512d vx0 = _mm512_load_pd(fx_ptr);
            const __m512d vx1 = _mm512_load_pd(fx_ptr + 8);

            const __mmask8 nan0 = _mm512_cmp_pd_mask(vx0, vx0, _CMP_UNORD_Q);
            const __mmask8 nan1 = _mm512_cmp_pd_mask(vx1, vx1, _CMP_UNORD_Q);
            const __mmask16 nan_k = static_cast<__mmask16>(
                static_cast<uint16_t>(nan0) | (static_cast<uint16_t>(nan1) << 8)
            );

            __mmask8 z0 = 0;
            __mmask8 z1 = 0;
            __mmask16 zero_k = 0;
            if (span.has_zero_missing) {
                z0 = _mm512_cmp_pd_mask(AbsPd512(vx0), zero_eps, _CMP_LE_OQ);
                z1 = _mm512_cmp_pd_mask(AbsPd512(vx1), zero_eps, _CMP_LE_OQ);
                zero_k = static_cast<__mmask16>(
                    static_cast<uint16_t>(z0) | (static_cast<uint16_t>(z1) << 8)
                );
            }

            const __mmask16 special_k = nan_k | zero_k;
            const __m512d vx0_clean = (nan0 | z0) ? _mm512_mask_mov_pd(vx0, nan0 | z0, neg_inf) : vx0;
            const __m512d vx1_clean = (nan1 | z1) ? _mm512_mask_mov_pd(vx1, nan1 | z1, neg_inf) : vx1;

            const double x_max = HorizontalMaxPd512(_mm512_max_pd(vx0_clean, vx1_clean));
            uint32_t idx = span.split_begin;
            const uint32_t idx_stop = idx + CountStrictlyLess512(thresholds + idx, span.split_end - idx, x_max);

            if (idx < idx_stop) {
                if (special_k == 0) {
                    const double x_min = HorizontalMinPd512(_mm512_min_pd(vx0_clean, vx1_clean));
                    const uint32_t idx_all = idx + CountStrictlyLess512(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        v_vec[tid] = _mm512_and_si512(
                            v_vec[tid], _mm512_set1_epi32(static_cast<int32_t>(masks[idx]))
                        );
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    const __m512d vt = _mm512_set1_pd(thresholds[idx]);
                    const __mmask8 k0 = _mm512_cmp_pd_mask(vx0_clean, vt, _CMP_GT_OQ);
                    const __mmask8 k1 = _mm512_cmp_pd_mask(vx1_clean, vt, _CMP_GT_OQ);
                    const __mmask16 gt_k = static_cast<__mmask16>(
                        static_cast<uint16_t>(k0) | (static_cast<uint16_t>(k1) << 8)
                    );
                    const uint16_t tid = tree_ids[idx];
                    v_vec[tid] = _mm512_mask_and_epi32(
                        v_vec[tid], gt_k, v_vec[tid], _mm512_set1_epi32(static_cast<int32_t>(masks[idx]))
                    );
                }
            }

            if (nan_k != 0) {
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    v_vec[tid] = _mm512_mask_and_epi32(
                        v_vec[tid], nan_k, v_vec[tid],
                        _mm512_set1_epi32(static_cast<int32_t>(nan_entries[ni].mask))
                    );
                }
            }

            if (zero_k != 0) {
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    v_vec[tid] = _mm512_mask_and_epi32(
                        v_vec[tid], zero_k, v_vec[tid],
                        _mm512_set1_epi32(static_cast<int32_t>(zero_entries[zi].mask))
                    );
                }
            }
        }

        const uint32_t stride = blk.leaf_stride;
        const double* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m512i leaf_idx = Ctz32Avx512(v_vec[t]);
                const __m256i idx0 = _mm512_castsi512_si256(leaf_idx);
                const __m256i idx1 = _mm512_extracti32x8_epi32(leaf_idx, 1);
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                acc0 = _mm512_add_pd(acc0, _mm512_i32gather_pd(idx0, tl, 8));
                acc1 = _mm512_add_pd(acc1, _mm512_i32gather_pd(idx1, tl, 8));
            }
            double* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm512_store_pd(dst, _mm512_add_pd(_mm512_load_pd(dst), acc0));
            _mm512_store_pd(dst + 8, _mm512_add_pd(_mm512_load_pd(dst + 8), acc1));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m512i leaf_idx = Ctz32Avx512(v_vec[t]);
                const __m256i idx0 = _mm512_castsi512_si256(leaf_idx);
                const __m256i idx1 = _mm512_extracti32x8_epi32(leaf_idx, 1);
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m512d vals0 = _mm512_i32gather_pd(idx0, tl, 8);
                const __m512d vals1 = _mm512_i32gather_pd(idx1, tl, 8);
                double* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm512_store_pd(dst, _mm512_add_pd(_mm512_load_pd(dst), vals0));
                _mm512_store_pd(dst + 8, _mm512_add_pd(_mm512_load_pd(dst + 8), vals1));
            }
        }
    }
}

void EvalQS64Block_Avx512(
    const QS64Block& blk,
    const double* tile_col_major,
    size_t actual_rows,
    double* tile_acc
) noexcept {
    constexpr size_t W = 8; // 8x double in __m512d + 8x uint64_t in __m512i (1-to-1 width!)
    const size_t n_rows = (actual_rows + 7u) & ~size_t(7u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(64) __m512i v_vec[64];
    const double* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks_lo = blk.masks_lo.data();
    const uint32_t* masks_hi = blk.masks_hi.data();
    const QS64SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS64SpecialEntry* zero_entries = blk.zero_entries.data();

    const __m512d neg_inf = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d zero_eps = _mm512_set1_pd(kZeroThreshold);
    const __m512i all_ones = _mm512_set1_epi64(-1LL);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v_vec[t] = all_ones;
        }

        for (const QS64FeatureSpan& span : blk.features) {
            const double* fx_ptr = tile_col_major + (static_cast<size_t>(span.feature_idx) << 7) + r0;
            const __m512d vx = _mm512_load_pd(fx_ptr);

            const __mmask8 nan_k = _mm512_cmp_pd_mask(vx, vx, _CMP_UNORD_Q);
            __mmask8 zero_k = 0;
            if (span.has_zero_missing) {
                zero_k = _mm512_cmp_pd_mask(AbsPd512(vx), zero_eps, _CMP_LE_OQ);
            }

            const __mmask8 special_k = nan_k | zero_k;
            const __m512d vx_clean = special_k ? _mm512_mask_mov_pd(vx, special_k, neg_inf) : vx;

            const double x_max = HorizontalMaxPd512(vx_clean);
            uint32_t idx = span.split_begin;
            const uint32_t idx_stop = idx + CountStrictlyLess512(thresholds + idx, span.split_end - idx, x_max);

            if (idx < idx_stop) {
                if (special_k == 0) {
                    const double x_min = HorizontalMinPd512(vx_clean);
                    const uint32_t idx_all = idx + CountStrictlyLess512(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                        v_vec[tid] = _mm512_and_si512(v_vec[tid], _mm512_set1_epi64(static_cast<int64_t>(m64)));
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    const __mmask8 gt_k = _mm512_cmp_pd_mask(vx_clean, _mm512_set1_pd(thresholds[idx]), _CMP_GT_OQ);
                    const uint16_t tid = tree_ids[idx];
                    const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                    v_vec[tid] = _mm512_mask_and_epi64(
                        v_vec[tid], gt_k, v_vec[tid], _mm512_set1_epi64(static_cast<int64_t>(m64))
                    );
                }
            }

            if (nan_k != 0) {
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    const uint64_t m64 = (static_cast<uint64_t>(nan_entries[ni].mask_hi) << 32) | nan_entries[ni].mask_lo;
                    v_vec[tid] = _mm512_mask_and_epi64(
                        v_vec[tid], nan_k, v_vec[tid], _mm512_set1_epi64(static_cast<int64_t>(m64))
                    );
                }
            }

            if (zero_k != 0) {
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    const uint64_t m64 = (static_cast<uint64_t>(zero_entries[zi].mask_hi) << 32) | zero_entries[zi].mask_lo;
                    v_vec[tid] = _mm512_mask_and_epi64(
                        v_vec[tid], zero_k, v_vec[tid], _mm512_set1_epi64(static_cast<int64_t>(m64))
                    );
                }
            }
        }

        const uint32_t stride = blk.leaf_stride;
        const double* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m512d acc = _mm512_setzero_pd();
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_idx = Ctz64Avx512(v_vec[t]);
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m512d vals = _mm512_i32gather_pd(leaf_idx, tl, 8);
                acc = _mm512_add_pd(acc, vals);
            }
            double* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm512_store_pd(dst, _mm512_add_pd(_mm512_load_pd(dst), acc));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_idx = Ctz64Avx512(v_vec[t]);
                const double* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m512d vals = _mm512_i32gather_pd(leaf_idx, tl, 8);
                double* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm512_store_pd(dst, _mm512_add_pd(_mm512_load_pd(dst), vals));
            }
        }
    }
}

void EvalCompactTrees_Avx512(
    const CompactTreeMeta* trees,
    size_t num_trees,
    const CompactNode* all_nodes,
    const double* all_leaves,
    const double* tile_col_major,
    size_t n_rows,
    double* tile_acc
) noexcept {
    constexpr size_t W = 8; // 8x double in __m512d + 8x int32 in __m256i
    const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i zero_i = _mm256_setzero_si256();
    const __m256i minus_one = _mm256_set1_epi32(-1);
    const __m256i feat_mask = _mm256_set1_epi32(0x00FFFFFF);
    const __m512d zero_eps = _mm512_set1_pd(kZeroThreshold);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        const __m256i lane_r0 = _mm256_add_epi32(lane_ids, _mm256_set1_epi32(static_cast<int32_t>(r0)));

        for (size_t t = 0; t < num_trees; ++t) {
            const CompactTreeMeta& meta = trees[t];
            const CompactNode* nodes = all_nodes + meta.node_offset;
            const double* leaves = all_leaves + meta.leaf_offset;

            // Unroll root node 0 across 8 rows
            const CompactNode& root = nodes[0];
            const __m512d vx0 = _mm512_load_pd(
                tile_col_major + (static_cast<size_t>(root.feature_idx()) << 7) + r0
            );
            __mmask8 go_left0 = _mm512_cmp_pd_mask(vx0, _mm512_set1_pd(root.threshold), _CMP_LE_OQ);
            if (root.nan_goes_left()) {
                go_left0 |= _mm512_cmp_pd_mask(vx0, vx0, _CMP_UNORD_Q);
            }
            if (root.has_zero_missing()) {
                const __mmask8 is_z0 = _mm512_cmp_pd_mask(AbsPd512(vx0), zero_eps, _CMP_LE_OQ);
                if (root.zero_goes_left()) {
                    go_left0 |= is_z0;
                } else {
                    go_left0 &= static_cast<__mmask8>(~is_z0);
                }
            }

            __m256i v_idx = _mm256_mask_blend_epi32(
                go_left0,
                _mm256_set1_epi32(root.right_child),
                _mm256_set1_epi32(root.left_child)
            );

            const double* nodes_f64 = reinterpret_cast<const double*>(nodes);
            const int* nodes_i32 = reinterpret_cast<const int*>(nodes);

            while (true) {
                const __mmask8 active = _mm256_cmpge_epi32_mask(v_idx, zero_i);
                if (active == 0) break;

                const __m256i idx4 = _mm256_slli_epi32(v_idx, 2);
                const __m256i idx8 = _mm256_slli_epi32(v_idx, 3);
                const __m512d v_thresh = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), active, idx4, nodes_f64, 8);
                const __m256i v_ff = _mm256_mmask_i32gather_epi32(zero_i, active, idx8, nodes_i32 + 2, 4);
                const __m256i v_left = _mm256_mmask_i32gather_epi32(zero_i, active, idx8, nodes_i32 + 3, 4);
                const __m256i v_right = _mm256_mmask_i32gather_epi32(zero_i, active, idx8, nodes_i32 + 4, 4);

                const __m256i v_feat = _mm256_and_si256(v_ff, feat_mask);
                const __m256i feat_off = _mm256_add_epi32(_mm256_slli_epi32(v_feat, 7), lane_r0);
                const __m512d vx = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), active, feat_off, tile_col_major, 8);

                const __mmask8 le_k = _mm512_cmp_pd_mask(vx, v_thresh, _CMP_LE_OQ);
                const __mmask8 unord_k = _mm512_cmp_pd_mask(vx, vx, _CMP_UNORD_Q);
                const __mmask8 nan_left_k = _mm256_test_epi32_mask(v_ff, _mm256_set1_epi32(1 << 24));
                __mmask8 go_left = le_k | (unord_k & nan_left_k);

                if (meta.has_zero_missing) {
                    const __mmask8 has_zm_k = _mm256_test_epi32_mask(v_ff, _mm256_set1_epi32(1 << 25));
                    const __mmask8 zm_left_k = _mm256_test_epi32_mask(v_ff, _mm256_set1_epi32(1 << 26));
                    const __mmask8 is_zero_k = _mm512_cmp_pd_mask(AbsPd512(vx), zero_eps, _CMP_LE_OQ);
                    const __mmask8 ovr = has_zm_k & is_zero_k;
                    go_left = static_cast<__mmask8>((go_left & ~ovr) | (zm_left_k & ovr));
                }

                const __m256i next_idx = _mm256_mask_blend_epi32(go_left, v_right, v_left);
                v_idx = _mm256_mask_mov_epi32(v_idx, active, next_idx);
            }

            const __m256i leaf_idx = _mm256_xor_si256(v_idx, minus_one);
            const __m512d vals = _mm512_i32gather_pd(leaf_idx, leaves, 8);
            double* dst = tile_acc + static_cast<size_t>(meta.class_id) * kTileStride + r0;
            _mm512_store_pd(dst, _mm512_add_pd(_mm512_load_pd(dst), vals));
        }
    }
}

void FinalizeTileOutput_Avx512(
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
    if (num_outputs == 1 && (raw_score || objective == ObjectiveType::Regression)) {
        const bool need_scale = (scale_factor != 1.0);
        const __m512d v_scale = _mm512_set1_pd(scale_factor);
        size_t r = 0;
        if (out_f64 != nullptr) {
            if (!need_scale) {
                std::memcpy(out_f64, tile_acc, n_rows * sizeof(double));
                return;
            }
            for (; r + 8 <= n_rows; r += 8) {
                _mm512_storeu_pd(out_f64 + r, _mm512_mul_pd(_mm512_load_pd(tile_acc + r), v_scale));
            }
            for (; r < n_rows; ++r) {
                out_f64[r] = tile_acc[r] * scale_factor;
            }
            return;
        }
        if (out_f32 != nullptr) {
            for (; r + 8 <= n_rows; r += 8) {
                __m512d v = _mm512_load_pd(tile_acc + r);
                if (need_scale) v = _mm512_mul_pd(v, v_scale);
                _mm256_storeu_ps(out_f32 + r, _mm512_cvtpd_ps(v));
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

const KernelTable& GetAvx512Kernels() noexcept {
    static const KernelTable kTable = {
        SimdIsa::Avx512,
        "avx512",
        &EvalQS32Block_Avx512,
        &EvalQS64Block_Avx512,
        &EvalCompactTrees_Avx512,
        &FinalizeTileOutput_Avx512
    };
    return kTable;
}

} // namespace fast_scorer

#endif // FAST_SCORER_ARCH_X86_64
