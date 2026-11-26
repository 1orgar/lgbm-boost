#include "fast_scorer/kernels.h"
#include "fast_scorer/math_simd.h"

#if defined(FAST_SCORER_ARCH_X86_64)

#include <immintrin.h>
#include <cstring>
#include <limits>

namespace fast_scorer {

namespace {

inline float HorizontalMaxAvx2(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m1 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m1, _mm_movehl_ps(m1, m1));
    __m128 m3 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m3);
}

inline float HorizontalMinAvx2(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m1 = _mm_min_ps(lo, hi);
    __m128 m2 = _mm_min_ps(m1, _mm_movehl_ps(m1, m1));
    __m128 m3 = _mm_min_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m3);
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

// Hardware PHMINPOSUW horizontal min/max of 8 unsigned uint16_t bins
inline void MinMaxU16x8(__m128i vb, uint16_t& out_min, uint16_t& out_max) noexcept {
    out_min = static_cast<uint16_t>(_mm_extract_epi16(_mm_minpos_epu16(vb), 0));
    const __m128i inv = _mm_xor_si128(vb, _mm_set1_epi16(-1));
    out_max = static_cast<uint16_t>(0xFFFFu ^ static_cast<uint16_t>(_mm_extract_epi16(_mm_minpos_epu16(inv), 0)));
}

// Branchless monobound search: exact count of elements strictly < val
inline uint32_t CountStrictlyLessAvx2(const float* sorted, uint32_t len, float val) noexcept {
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
}

void EvalQS32Block_Avx2(
    const QS32Block& blk,
    const float* tile_col_major,
    const uint16_t* /*tile_bins*/,
    size_t actual_rows,
    float* tile_acc
) noexcept {
    // 16-row dual-YMM register blocking: 2x ILP & 2x amortization of threshold/mask/tree_id loads!
    constexpr size_t W = 16;
    const size_t n_rows = (actual_rows + 15u) & ~size_t(15u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(64) __m256i v0[128];
    alignas(64) __m256i v1[128];
    const float* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks = blk.masks.data();
    const QS32SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS32SpecialEntry* zero_entries = blk.zero_entries.data();

    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 zero_eps = _mm256_set1_ps(1e-35f);
    const __m256i all_ones = _mm256_set1_epi32(-1);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v0[t] = all_ones;
            v1[t] = all_ones;
        }

        for (const QS32FeatureSpan& span : blk.features) {
            const float* fx_ptr = tile_col_major + (static_cast<size_t>(span.feature_idx) << 7) + r0;
            const __m256 vx0 = _mm256_load_ps(fx_ptr);
            const __m256 vx1 = _mm256_load_ps(fx_ptr + 8);

            const __m256 nan0 = _mm256_cmp_ps(vx0, vx0, _CMP_UNORD_Q);
            const __m256 nan1 = _mm256_cmp_ps(vx1, vx1, _CMP_UNORD_Q);
            const int nan_bits = _mm256_movemask_ps(nan0) | (_mm256_movemask_ps(nan1) << 8);

            __m256 zero0 = _mm256_setzero_ps();
            __m256 zero1 = _mm256_setzero_ps();
            int zero_bits = 0;
            if (span.has_zero_missing) {
                zero0 = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, vx0), zero_eps, _CMP_LE_OQ);
                zero1 = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, vx1), zero_eps, _CMP_LE_OQ);
                zero_bits = _mm256_movemask_ps(zero0) | (_mm256_movemask_ps(zero1) << 8);
            }

            const __m256 vx0_clean = (nan_bits | zero_bits)
                                         ? _mm256_blendv_ps(vx0, neg_inf, _mm256_or_ps(nan0, zero0))
                                         : vx0;
            const __m256 vx1_clean = (nan_bits | zero_bits)
                                         ? _mm256_blendv_ps(vx1, neg_inf, _mm256_or_ps(nan1, zero1))
                                         : vx1;

            const float x_max = HorizontalMaxAvx2(_mm256_max_ps(vx0_clean, vx1_clean));
            uint32_t idx = span.split_begin;
            const uint32_t len = span.split_end - idx;
            const uint32_t idx_stop = idx + CountStrictlyLessAvx2(thresholds + idx, len, x_max);

            if (idx < idx_stop) {
                if ((nan_bits | zero_bits) == 0) {
                    const float x_min = HorizontalMinAvx2(_mm256_min_ps(vx0_clean, vx1_clean));
                    const uint32_t idx_all = idx + CountStrictlyLessAvx2(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(masks[idx]));
                        v0[tid] = _mm256_and_si256(v0[tid], vm);
                        v1[tid] = _mm256_and_si256(v1[tid], vm);
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    const __m256 vt = _mm256_broadcast_ss(&thresholds[idx]);
                    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(masks[idx]));
                    const __m256i le0 = _mm256_castps_si256(_mm256_cmp_ps(vx0_clean, vt, _CMP_LE_OQ));
                    const __m256i le1 = _mm256_castps_si256(_mm256_cmp_ps(vx1_clean, vt, _CMP_LE_OQ));
                    const uint16_t tid = tree_ids[idx];
                    v0[tid] = _mm256_and_si256(v0[tid], _mm256_or_si256(le0, vm));
                    v1[tid] = _mm256_and_si256(v1[tid], _mm256_or_si256(le1, vm));
                }
            }

            if (nan_bits != 0) {
                const __m256i not_nan0 = _mm256_andnot_si256(_mm256_castps_si256(nan0), all_ones);
                const __m256i not_nan1 = _mm256_andnot_si256(_mm256_castps_si256(nan1), all_ones);
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(nan_entries[ni].mask));
                    v0[tid] = _mm256_and_si256(v0[tid], _mm256_or_si256(not_nan0, vm));
                    v1[tid] = _mm256_and_si256(v1[tid], _mm256_or_si256(not_nan1, vm));
                }
            }

            if (zero_bits != 0) {
                const __m256i not_z0 = _mm256_andnot_si256(_mm256_castps_si256(zero0), all_ones);
                const __m256i not_z1 = _mm256_andnot_si256(_mm256_castps_si256(zero1), all_ones);
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(zero_entries[zi].mask));
                    v0[tid] = _mm256_and_si256(v0[tid], _mm256_or_si256(not_z0, vm));
                    v1[tid] = _mm256_and_si256(v1[tid], _mm256_or_si256(not_z1, vm));
                }
            }
        }

        const uint32_t stride = blk.leaf_stride;
        const float* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (uint32_t t = 0; t < n_trees; ++t) {
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(tl, Ctz32Avx2(v0[t]), 4));
                acc1 = _mm256_add_ps(acc1, _mm256_i32gather_ps(tl, Ctz32Avx2(v1[t]), 4));
            }
            float* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), acc0));
            _mm256_store_ps(dst + 8, _mm256_add_ps(_mm256_load_ps(dst + 8), acc1));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m256 vals0 = _mm256_i32gather_ps(tl, Ctz32Avx2(v0[t]), 4);
                const __m256 vals1 = _mm256_i32gather_ps(tl, Ctz32Avx2(v1[t]), 4);
                float* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), vals0));
                _mm256_store_ps(dst + 8, _mm256_add_ps(_mm256_load_ps(dst + 8), vals1));
            }
        }
    }
}

void EvalQS64Block_Avx2(
    const QS64Block& blk,
    const float* tile_col_major,
    const uint16_t* tile_bins,
    size_t actual_rows,
    float* tile_acc
) noexcept {
    constexpr size_t W = 8;
    const size_t n_rows = (actual_rows + 7u) & ~size_t(7u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(32) __m256i v_lo[64];
    alignas(32) __m256i v_hi[64];

    const float* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks_lo = blk.masks_lo.data();
    const uint32_t* masks_hi = blk.masks_hi.data();
    const uint16_t* bin_offsets = blk.bin_offsets.data();
    const QS64SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS64SpecialEntry* zero_entries = blk.zero_entries.data();

    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 zero_eps = _mm256_set1_ps(1e-35f);
    const __m256i all_ones = _mm256_set1_epi32(-1);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v_lo[t] = all_ones;
            v_hi[t] = all_ones;
        }

        for (const QS64FeatureSpan& span : blk.features) {
            const size_t f_base = (static_cast<size_t>(span.feature_idx) << 7) + r0;

            if (tile_bins != nullptr) {
                const __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(tile_bins + f_base));
                uint16_t b_min = 0, b_max = 0;
                MinMaxU16x8(vb, b_min, b_max);
                if (b_max < 0xFFFEu) {
                    if (b_max == 0u) continue;
                    const uint16_t* boff = bin_offsets + span.bin_offset_start;
                    uint32_t idx = span.split_begin;
                    const uint32_t idx_all = boff[b_min];
                    const uint32_t idx_stop = boff[b_max];

                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        v_lo[tid] = _mm256_and_si256(
                            v_lo[tid], _mm256_set1_epi32(static_cast<int32_t>(masks_lo[idx]))
                        );
                        v_hi[tid] = _mm256_and_si256(
                            v_hi[tid], _mm256_set1_epi32(static_cast<int32_t>(masks_hi[idx]))
                        );
                    }
                    if (idx < idx_stop) {
                        const __m256 vx = _mm256_load_ps(tile_col_major + f_base);
                        for (; idx < idx_stop; ++idx) {
                            const __m256 vt = _mm256_broadcast_ss(&thresholds[idx]);
                            const __m256i cmp_le = _mm256_castps_si256(_mm256_cmp_ps(vx, vt, _CMP_LE_OQ));
                            const uint16_t tid = tree_ids[idx];
                            v_lo[tid] = _mm256_and_si256(
                                v_lo[tid],
                                _mm256_or_si256(cmp_le, _mm256_set1_epi32(static_cast<int32_t>(masks_lo[idx])))
                            );
                            v_hi[tid] = _mm256_and_si256(
                                v_hi[tid],
                                _mm256_or_si256(cmp_le, _mm256_set1_epi32(static_cast<int32_t>(masks_hi[idx])))
                            );
                        }
                    }
                    continue;
                }
            }

            const float* fx_ptr = tile_col_major + f_base;
            const __m256 vx = _mm256_load_ps(fx_ptr);

            const __m256 cmp_nan = _mm256_cmp_ps(vx, vx, _CMP_UNORD_Q);
            const int nan_bits = _mm256_movemask_ps(cmp_nan);

            __m256 cmp_zero = _mm256_setzero_ps();
            int zero_bits = 0;
            if (span.has_zero_missing) {
                __m256 abs_vx = _mm256_andnot_ps(sign_mask, vx);
                cmp_zero = _mm256_cmp_ps(abs_vx, zero_eps, _CMP_LE_OQ);
                zero_bits = _mm256_movemask_ps(cmp_zero);
            }

            const __m256 special_mask = _mm256_or_ps(cmp_nan, cmp_zero);
            const __m256 vx_clean = (nan_bits | zero_bits)
                                        ? _mm256_blendv_ps(vx, neg_inf, special_mask)
                                        : vx;

            const float x_max = HorizontalMaxAvx2(vx_clean);
            uint32_t idx = span.split_begin;
            const uint32_t end = span.split_end;

            if (x_max > thresholds[idx]) {
                if ((nan_bits | zero_bits) == 0) {
                    const float x_min = HorizontalMinAvx2(vx_clean);
                    for (; idx < end && thresholds[idx] < x_min; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        v_lo[tid] = _mm256_and_si256(
                            v_lo[tid], _mm256_set1_epi32(static_cast<int32_t>(masks_lo[idx]))
                        );
                        v_hi[tid] = _mm256_and_si256(
                            v_hi[tid], _mm256_set1_epi32(static_cast<int32_t>(masks_hi[idx]))
                        );
                    }
                }

                for (; idx < end; ++idx) {
                    const float th = thresholds[idx];
                    if (th >= x_max) break;
                    const __m256 vt = _mm256_broadcast_ss(&thresholds[idx]);
                    const __m256i cmp_le = _mm256_castps_si256(_mm256_cmp_ps(vx_clean, vt, _CMP_LE_OQ));
                    const uint16_t tid = tree_ids[idx];
                    v_lo[tid] = _mm256_and_si256(
                        v_lo[tid],
                        _mm256_or_si256(cmp_le, _mm256_set1_epi32(static_cast<int32_t>(masks_lo[idx])))
                    );
                    v_hi[tid] = _mm256_and_si256(
                        v_hi[tid],
                        _mm256_or_si256(cmp_le, _mm256_set1_epi32(static_cast<int32_t>(masks_hi[idx])))
                    );
                }
            }

            if (nan_bits != 0) {
                const __m256i not_nan = _mm256_andnot_si256(_mm256_castps_si256(cmp_nan), all_ones);
                for (uint32_t ni = span.nan_begin; ni < span.nan_end; ++ni) {
                    const uint16_t tid = nan_entries[ni].tree_in_block;
                    v_lo[tid] = _mm256_and_si256(
                        v_lo[tid],
                        _mm256_or_si256(not_nan, _mm256_set1_epi32(static_cast<int32_t>(nan_entries[ni].mask_lo)))
                    );
                    v_hi[tid] = _mm256_and_si256(
                        v_hi[tid],
                        _mm256_or_si256(not_nan, _mm256_set1_epi32(static_cast<int32_t>(nan_entries[ni].mask_hi)))
                    );
                }
            }

            if (zero_bits != 0) {
                const __m256i not_zero = _mm256_andnot_si256(_mm256_castps_si256(cmp_zero), all_ones);
                for (uint32_t zi = span.zero_begin; zi < span.zero_end; ++zi) {
                    const uint16_t tid = zero_entries[zi].tree_in_block;
                    v_lo[tid] = _mm256_and_si256(
                        v_lo[tid],
                        _mm256_or_si256(not_zero, _mm256_set1_epi32(static_cast<int32_t>(zero_entries[zi].mask_lo)))
                    );
                    v_hi[tid] = _mm256_and_si256(
                        v_hi[tid],
                        _mm256_or_si256(not_zero, _mm256_set1_epi32(static_cast<int32_t>(zero_entries[zi].mask_hi)))
                    );
                }
            }
        }

        const uint32_t stride = blk.leaf_stride;
        const float* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m256 acc = _mm256_setzero_ps();
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_idx = Ctz64SplitAvx2(v_lo[t], v_hi[t]);
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m256 vals = _mm256_i32gather_ps(tl, leaf_idx, 4);
                acc = _mm256_add_ps(acc, vals);
            }
            float* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), acc));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_idx = Ctz64SplitAvx2(v_lo[t], v_hi[t]);
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m256 vals = _mm256_i32gather_ps(tl, leaf_idx, 4);
                float* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), vals));
            }
        }
    }
}

void EvalCompactTrees_Avx2(
    const CompactTreeMeta* trees,
    size_t num_trees,
    const CompactNode* all_nodes,
    const float* all_leaves,
    const float* tile_col_major,
    size_t n_rows,
    float* tile_acc
) noexcept {
    constexpr size_t W = 8;
    const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i minus_one = _mm256_set1_epi32(-1);
    const __m256i feat_mask = _mm256_set1_epi32(0x00FFFFFF);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 zero_eps = _mm256_set1_ps(1e-35f);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        const __m256i lane_r0 = _mm256_add_epi32(lane_ids, _mm256_set1_epi32(static_cast<int32_t>(r0)));

        for (size_t t = 0; t < num_trees; ++t) {
            const CompactTreeMeta& meta = trees[t];
            const CompactNode* nodes = all_nodes + meta.node_offset;
            const float* leaves = all_leaves + meta.leaf_offset;

            // Unroll root node 0 (all 8 lanes start at root 0 -> contiguous load instead of gather!)
            const CompactNode& root = nodes[0];
            const __m256 vx0 = _mm256_load_ps(
                tile_col_major + (static_cast<size_t>(root.feature_idx()) << 7) + r0
            );
            const __m256 vt0 = _mm256_set1_ps(root.threshold);
            __m256i go_left0 = _mm256_castps_si256(_mm256_cmp_ps(vx0, vt0, _CMP_LE_OQ));
            if (root.nan_goes_left()) {
                __m256i unord0 = _mm256_castps_si256(_mm256_cmp_ps(vx0, vx0, _CMP_UNORD_Q));
                go_left0 = _mm256_or_si256(go_left0, unord0);
            }
            if (root.has_zero_missing()) {
                __m256 abs_x0 = _mm256_andnot_ps(sign_mask, vx0);
                __m256i is_z0 = _mm256_castps_si256(_mm256_cmp_ps(abs_x0, zero_eps, _CMP_LE_OQ));
                __m256i z_dir0 = root.zero_goes_left() ? minus_one : _mm256_setzero_si256();
                go_left0 = _mm256_blendv_epi8(go_left0, z_dir0, is_z0);
            }

            __m256i v_idx = _mm256_blendv_epi8(
                _mm256_set1_epi32(root.right_child),
                _mm256_set1_epi32(root.left_child),
                go_left0
            );

            const float* nodes_f32 = reinterpret_cast<const float*>(nodes);
            const int* nodes_i32 = reinterpret_cast<const int*>(nodes);

            if (!meta.has_zero_missing) {
                while (_mm256_movemask_ps(_mm256_castsi256_ps(v_idx)) != 0xFF) {
                    const __m256i active = _mm256_cmpgt_epi32(v_idx, minus_one);
                    const __m256i safe_idx = _mm256_and_si256(v_idx, active);
                    const __m256i idx4 = _mm256_slli_epi32(safe_idx, 2);

                    const __m256 v_thresh = _mm256_i32gather_ps(nodes_f32, idx4, 4);
                    const __m256i v_ff = _mm256_i32gather_epi32(nodes_i32 + 1, idx4, 4);
                    const __m256i v_left = _mm256_i32gather_epi32(nodes_i32 + 2, idx4, 4);
                    const __m256i v_right = _mm256_i32gather_epi32(nodes_i32 + 3, idx4, 4);

                    const __m256i v_feat = _mm256_and_si256(v_ff, feat_mask);
                    const __m256i feat_off = _mm256_add_epi32(_mm256_slli_epi32(v_feat, 7), lane_r0);
                    const __m256 vx = _mm256_i32gather_ps(tile_col_major, feat_off, 4);

                    const __m256i le_mask = _mm256_castps_si256(_mm256_cmp_ps(vx, v_thresh, _CMP_LE_OQ));
                    const __m256i unord = _mm256_castps_si256(_mm256_cmp_ps(vx, vx, _CMP_UNORD_Q));
                    // Bit 24 -> sign bit 31 -> arithmetic shift right 31
                    const __m256i nan_left = _mm256_srai_epi32(_mm256_slli_epi32(v_ff, 7), 31);
                    const __m256i go_left = _mm256_or_si256(le_mask, _mm256_and_si256(unord, nan_left));

                    const __m256i next_idx = _mm256_blendv_epi8(v_right, v_left, go_left);
                    v_idx = _mm256_blendv_epi8(v_idx, next_idx, active);
                }
            } else {
                while (_mm256_movemask_ps(_mm256_castsi256_ps(v_idx)) != 0xFF) {
                    const __m256i active = _mm256_cmpgt_epi32(v_idx, minus_one);
                    const __m256i safe_idx = _mm256_and_si256(v_idx, active);
                    const __m256i idx4 = _mm256_slli_epi32(safe_idx, 2);

                    const __m256 v_thresh = _mm256_i32gather_ps(nodes_f32, idx4, 4);
                    const __m256i v_ff = _mm256_i32gather_epi32(nodes_i32 + 1, idx4, 4);
                    const __m256i v_left = _mm256_i32gather_epi32(nodes_i32 + 2, idx4, 4);
                    const __m256i v_right = _mm256_i32gather_epi32(nodes_i32 + 3, idx4, 4);

                    const __m256i v_feat = _mm256_and_si256(v_ff, feat_mask);
                    const __m256i feat_off = _mm256_add_epi32(_mm256_slli_epi32(v_feat, 7), lane_r0);
                    const __m256 vx = _mm256_i32gather_ps(tile_col_major, feat_off, 4);

                    const __m256i le_mask = _mm256_castps_si256(_mm256_cmp_ps(vx, v_thresh, _CMP_LE_OQ));
                    const __m256i unord = _mm256_castps_si256(_mm256_cmp_ps(vx, vx, _CMP_UNORD_Q));
                    const __m256i nan_left = _mm256_srai_epi32(_mm256_slli_epi32(v_ff, 7), 31);
                    __m256i go_left = _mm256_or_si256(le_mask, _mm256_and_si256(unord, nan_left));

                    // Bit 25 = has_zero_missing, Bit 26 = zero_goes_left
                    const __m256i has_zm = _mm256_srai_epi32(_mm256_slli_epi32(v_ff, 6), 31);
                    const __m256i zm_left = _mm256_srai_epi32(_mm256_slli_epi32(v_ff, 5), 31);
                    const __m256 abs_vx = _mm256_andnot_ps(sign_mask, vx);
                    const __m256i is_zero = _mm256_castps_si256(_mm256_cmp_ps(abs_vx, zero_eps, _CMP_LE_OQ));
                    const __m256i override_z = _mm256_and_si256(has_zm, is_zero);
                    go_left = _mm256_blendv_epi8(go_left, zm_left, override_z);

                    const __m256i next_idx = _mm256_blendv_epi8(v_right, v_left, go_left);
                    v_idx = _mm256_blendv_epi8(v_idx, next_idx, active);
                }
            }

            const __m256i leaf_idx = _mm256_xor_si256(v_idx, minus_one);
            const __m256 vals = _mm256_i32gather_ps(leaves, leaf_idx, 4);
            float* dst = tile_acc + static_cast<size_t>(meta.class_id) * kTileStride + r0;
            _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), vals));
        }
    }
}

void FinalizeTileOutput_Avx2(
    float* tile_acc,
    size_t n_rows,
    int num_outputs,
    ObjectiveType objective,
    float sigmoid_alpha,
    float scale_factor,
    bool raw_score,
    float* out_row_major
) noexcept {
    constexpr size_t W = 8;
    const bool need_scale = (scale_factor != 1.0f);
    const __m256 v_scale = _mm256_set1_ps(scale_factor);
    const __m256 v_neg_alpha = _mm256_set1_ps(-sigmoid_alpha);

    if (num_outputs == 1) {
        size_t r = 0;
        for (; r + W <= n_rows; r += W) {
            __m256 v = _mm256_load_ps(tile_acc + r);
            if (need_scale) v = _mm256_mul_ps(v, v_scale);
            if (!raw_score) {
                if (objective == ObjectiveType::BinaryLogloss) {
                    v = simd_math::SigmoidPsAvx2(v, v_neg_alpha);
                } else if (objective == ObjectiveType::Exponential) {
                    v = simd_math::ExpPsAvx2(v);
                }
            }
            _mm256_storeu_ps(out_row_major + r, v);
        }
        for (; r < n_rows; ++r) {
            float s = need_scale ? (tile_acc[r] * scale_factor) : tile_acc[r];
            if (!raw_score) {
                if (objective == ObjectiveType::BinaryLogloss) {
                    s = simd_math::SafeSigmoidScalar(s, sigmoid_alpha);
                } else if (objective == ObjectiveType::Exponential) {
                    s = simd_math::SafeExpScalar(s);
                }
            }
            out_row_major[r] = s;
        }
        return;
    }

    // Multi-output K > 1: transform in-place in [K][128] using 8-wide aligned AVX2, then write row-major
    const size_t K = static_cast<size_t>(num_outputs);
    const size_t padded8 = (n_rows + 7u) & ~size_t(7u);

    if (need_scale) {
        for (size_t c = 0; c < K; ++c) {
            float* col = tile_acc + c * kTileStride;
            for (size_t r = 0; r < padded8; r += W) {
                _mm256_store_ps(col + r, _mm256_mul_ps(_mm256_load_ps(col + r), v_scale));
            }
        }
    }

    if (!raw_score) {
        if (objective == ObjectiveType::MulticlassSoftmax) {
            for (size_t r = 0; r < padded8; r += W) {
                __m256 vmax = _mm256_load_ps(tile_acc + r);
                for (size_t c = 1; c < K; ++c) {
                    vmax = _mm256_max_ps(vmax, _mm256_load_ps(tile_acc + c * kTileStride + r));
                }
                __m256 vsum = _mm256_setzero_ps();
                for (size_t c = 0; c < K; ++c) {
                    float* ptr = tile_acc + c * kTileStride + r;
                    __m256 ve = simd_math::ExpPsAvx2(_mm256_sub_ps(_mm256_load_ps(ptr), vmax));
                    _mm256_store_ps(ptr, ve);
                    vsum = _mm256_add_ps(vsum, ve);
                }
                const __m256 vinv = _mm256_div_ps(_mm256_set1_ps(1.0f), vsum);
                for (size_t c = 0; c < K; ++c) {
                    float* ptr = tile_acc + c * kTileStride + r;
                    _mm256_store_ps(ptr, _mm256_mul_ps(_mm256_load_ps(ptr), vinv));
                }
            }
        } else if (objective == ObjectiveType::BinaryLogloss) {
            for (size_t c = 0; c < K; ++c) {
                float* col = tile_acc + c * kTileStride;
                for (size_t r = 0; r < padded8; r += W) {
                    _mm256_store_ps(col + r, simd_math::SigmoidPsAvx2(_mm256_load_ps(col + r), v_neg_alpha));
                }
            }
        } else if (objective == ObjectiveType::Exponential) {
            for (size_t c = 0; c < K; ++c) {
                float* col = tile_acc + c * kTileStride;
                for (size_t r = 0; r < padded8; r += W) {
                    _mm256_store_ps(col + r, simd_math::ExpPsAvx2(_mm256_load_ps(col + r)));
                }
            }
        }
    }

    for (size_t r = 0; r < n_rows; ++r) {
        float* drow = out_row_major + r * K;
        for (size_t c = 0; c < K; ++c) {
            drow[c] = tile_acc[c * kTileStride + r];
        }
    }
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
