#include "fast_scorer/kernels.h"
#include "fast_scorer/math_simd.h"

#if defined(FAST_SCORER_ARCH_X86_64)

#include <immintrin.h>
#include <cstring>
#include <limits>

namespace fast_scorer {

namespace {

inline float HorizontalMax256(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m1 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m1, _mm_movehl_ps(m1, m1));
    __m128 m3 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m3);
}

inline float HorizontalMin256(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m1 = _mm_min_ps(lo, hi);
    __m128 m2 = _mm_min_ps(m1, _mm_movehl_ps(m1, m1));
    __m128 m3 = _mm_min_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m3);
}

// Portable AVX-512F/DQ helpers compatible with older GCC 8 on Oracle Linux 8 / RHEL 8
inline __m512 Abs512(__m512 v) noexcept {
    return _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(v), _mm512_set1_epi32(0x7FFFFFFF)));
}

inline float HorizontalMax512(__m512 v) noexcept {
    const __m256 lo = _mm512_castps512_ps256(v);
    const __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return HorizontalMax256(_mm256_max_ps(lo, hi));
}

inline float HorizontalMin512(__m512 v) noexcept {
    const __m256 lo = _mm512_castps512_ps256(v);
    const __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return HorizontalMin256(_mm256_min_ps(lo, hi));
}

// Hardware VPOPCNTDQ 16-lane 32-bit trailing zero count: tzcnt32(v) = 31 - lzcnt32(v & -v)
inline __m512i Ctz32Avx512(__m512i v) noexcept {
    const __m512i lsb = _mm512_and_si512(v, _mm512_sub_epi32(_mm512_setzero_si512(), v));
    const __m512i lz = _mm512_lzcnt_epi32(lsb);
    return _mm512_sub_epi32(_mm512_set1_epi32(31), lz);
}

// Hardware VPOPCNTDQ 8-lane 64-bit trailing zero count: tzcnt64(v) = 63 - lzcnt64(v & -v)
inline __m256i Ctz64Avx512(__m512i v) noexcept {
    const __m512i lsb = _mm512_and_si512(v, _mm512_sub_epi64(_mm512_setzero_si512(), v));
    const __m512i lz = _mm512_lzcnt_epi64(lsb);
    const __m512i tz64 = _mm512_sub_epi64(_mm512_set1_epi64(63), lz);
    return _mm512_cvtepi64_epi32(tz64);
}

inline uint32_t CountStrictlyLess512(const float* sorted, uint32_t len, float val) noexcept {
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

inline void MinMaxU16x8_512(__m128i vb, uint16_t& out_min, uint16_t& out_max) noexcept {
    out_min = static_cast<uint16_t>(_mm_extract_epi16(_mm_minpos_epu16(vb), 0));
    const __m128i inv = _mm_xor_si128(vb, _mm_set1_epi16(-1));
    out_max = static_cast<uint16_t>(0xFFFFu ^ static_cast<uint16_t>(_mm_extract_epi16(_mm_minpos_epu16(inv), 0)));
}

inline void MinMaxU16x16_512(const uint16_t* ptr, uint16_t& out_min, uint16_t& out_max) noexcept {
    const __m128i v0 = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    const __m128i v1 = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr + 8));
    const __m128i vmin = _mm_min_epu16(v0, v1);
    const __m128i vmax = _mm_max_epu16(v0, v1);
    out_min = static_cast<uint16_t>(_mm_extract_epi16(_mm_minpos_epu16(vmin), 0));
    const __m128i inv = _mm_xor_si128(vmax, _mm_set1_epi16(-1));
    out_max = static_cast<uint16_t>(0xFFFFu ^ static_cast<uint16_t>(_mm_extract_epi16(_mm_minpos_epu16(inv), 0)));
}

void EvalQS32Block_Avx512(
    const QS32Block& blk,
    const float* tile_col_major,
    const uint16_t* tile_bins,
    size_t actual_rows,
    float* tile_acc
) noexcept {
    constexpr size_t W = 16;
    const size_t n_rows = (actual_rows + 15u) & ~size_t(15u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(64) __m512i v_vec[128];
    const float* thresholds = blk.thresholds.data();
    const uint16_t* tree_ids = blk.tree_ids.data();
    const uint32_t* masks = blk.masks.data();
    const uint16_t* bin_offsets = blk.bin_offsets.data();
    const QS32SpecialEntry* nan_entries = blk.nan_entries.data();
    const QS32SpecialEntry* zero_entries = blk.zero_entries.data();

    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 zero_eps = _mm512_set1_ps(1e-35f);
    const __m512i all_ones = _mm512_set1_epi32(-1);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v_vec[t] = all_ones;
        }

        for (const QS32FeatureSpan& span : blk.features) {
            const size_t f_base = (static_cast<size_t>(span.feature_idx) << 7) + r0;

            if (tile_bins != nullptr) {
                uint16_t b_min = 0, b_max = 0;
                MinMaxU16x16_512(tile_bins + f_base, b_min, b_max);
                if (b_max < 0xFFFEu) {
                    if (b_max == 0u) continue;
                    const uint16_t* boff = bin_offsets + span.bin_offset_start;
                    uint32_t idx = span.split_begin;
                    const uint32_t idx_all = boff[b_min];
                    const uint32_t idx_stop = boff[b_max];

                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        v_vec[tid] = _mm512_and_si512(
                            v_vec[tid], _mm512_set1_epi32(static_cast<int32_t>(masks[idx]))
                        );
                    }
                    if (idx < idx_stop) {
                        const __m512 vx = _mm512_load_ps(tile_col_major + f_base);
                        for (; idx < idx_stop; ++idx) {
                            const __mmask16 gt_k = _mm512_cmp_ps_mask(vx, _mm512_set1_ps(thresholds[idx]), _CMP_GT_OQ);
                            const uint16_t tid = tree_ids[idx];
                            v_vec[tid] = _mm512_mask_and_epi32(
                                v_vec[tid], gt_k, v_vec[tid], _mm512_set1_epi32(static_cast<int32_t>(masks[idx]))
                            );
                        }
                    }
                    continue;
                }
            }

            const float* fx_ptr = tile_col_major + f_base;
            const __m512 vx = _mm512_load_ps(fx_ptr);

            const __mmask16 nan_k = _mm512_cmp_ps_mask(vx, vx, _CMP_UNORD_Q);
            __mmask16 zero_k = 0;
            if (span.has_zero_missing) {
                const __m512 abs_vx = Abs512(vx);
                zero_k = _mm512_cmp_ps_mask(abs_vx, zero_eps, _CMP_LE_OQ);
            }

            const __mmask16 special_k = nan_k | zero_k;
            const __m512 vx_clean = special_k ? _mm512_mask_mov_ps(vx, special_k, neg_inf) : vx;

            const float x_max = HorizontalMax512(vx_clean);
            uint32_t idx = span.split_begin;
            const uint32_t idx_stop = idx + CountStrictlyLess512(thresholds + idx, span.split_end - idx, x_max);

            if (idx < idx_stop) {
                if (special_k == 0) {
                    const float x_min = HorizontalMin512(vx_clean);
                    const uint32_t idx_all = idx + CountStrictlyLess512(thresholds + idx, idx_stop - idx, x_min);
                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        v_vec[tid] = _mm512_and_si512(
                            v_vec[tid], _mm512_set1_epi32(static_cast<int32_t>(masks[idx]))
                        );
                    }
                }

                for (; idx < idx_stop; ++idx) {
                    // Embedded broadcast + k-mask predicated VPANDD: 2 instructions for 16 rows, zero loop-exit branch!
                    const __mmask16 gt_k = _mm512_cmp_ps_mask(vx_clean, _mm512_set1_ps(thresholds[idx]), _CMP_GT_OQ);
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
        const float* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m512 acc = _mm512_setzero_ps();
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m512i leaf_idx = Ctz32Avx512(v_vec[t]);
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m512 vals = _mm512_i32gather_ps(leaf_idx, tl, 4);
                acc = _mm512_add_ps(acc, vals);
            }
            float* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm512_store_ps(dst, _mm512_add_ps(_mm512_load_ps(dst), acc));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m512i leaf_idx = Ctz32Avx512(v_vec[t]);
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m512 vals = _mm512_i32gather_ps(leaf_idx, tl, 4);
                float* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm512_store_ps(dst, _mm512_add_ps(_mm512_load_ps(dst), vals));
            }
        }
    }
}

void EvalQS64Block_Avx512(
    const QS64Block& blk,
    const float* tile_col_major,
    const uint16_t* tile_bins,
    size_t actual_rows,
    float* tile_acc
) noexcept {
    constexpr size_t W = 8; // 8x uint64_t in a single __m512i register!
    const size_t n_rows = (actual_rows + 7u) & ~size_t(7u);
    const uint32_t n_trees = blk.num_trees;
    if (n_trees == 0) return;

    alignas(64) __m512i v_vec[64];
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
    const __m512i all_ones = _mm512_set1_epi64(-1LL);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        for (uint32_t t = 0; t < n_trees; ++t) {
            v_vec[t] = all_ones;
        }

        for (const QS64FeatureSpan& span : blk.features) {
            const size_t f_base = (static_cast<size_t>(span.feature_idx) << 7) + r0;

            if (tile_bins != nullptr) {
                const __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(tile_bins + f_base));
                uint16_t b_min = 0, b_max = 0;
                MinMaxU16x8_512(vb, b_min, b_max);
                if (b_max < 0xFFFEu) {
                    if (b_max == 0u) continue;
                    const uint16_t* boff = bin_offsets + span.bin_offset_start;
                    uint32_t idx = span.split_begin;
                    const uint32_t idx_all = boff[b_min];
                    const uint32_t idx_stop = boff[b_max];

                    for (; idx < idx_all; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                        v_vec[tid] = _mm512_and_si512(v_vec[tid], _mm512_set1_epi64(static_cast<int64_t>(m64)));
                    }
                    if (idx < idx_stop) {
                        const __m256 vx = _mm256_load_ps(tile_col_major + f_base);
                        for (; idx < idx_stop; ++idx) {
                            const __mmask8 gt_k = _mm256_cmp_ps_mask(vx, _mm256_set1_ps(thresholds[idx]), _CMP_GT_OQ);
                            const uint16_t tid = tree_ids[idx];
                            const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                            v_vec[tid] = _mm512_mask_and_epi64(
                                v_vec[tid], gt_k, v_vec[tid], _mm512_set1_epi64(static_cast<int64_t>(m64))
                            );
                        }
                    }
                    continue;
                }
            }

            const float* fx_ptr = tile_col_major + f_base;
            const __m256 vx = _mm256_load_ps(fx_ptr);

            const __mmask8 nan_k = _mm256_cmp_ps_mask(vx, vx, _CMP_UNORD_Q);
            __mmask8 zero_k = 0;
            if (span.has_zero_missing) {
                const __m256 abs_vx = _mm256_andnot_ps(sign_mask, vx);
                zero_k = _mm256_cmp_ps_mask(abs_vx, zero_eps, _CMP_LE_OQ);
            }

            const __mmask8 special_k = nan_k | zero_k;
            const __m256 vx_clean = special_k ? _mm256_mask_mov_ps(vx, special_k, neg_inf) : vx;

            const float x_max = HorizontalMax256(vx_clean);
            uint32_t idx = span.split_begin;
            const uint32_t end = span.split_end;

            if (x_max > thresholds[idx]) {
                if (special_k == 0) {
                    const float x_min = HorizontalMin256(vx_clean);
                    for (; idx < end && thresholds[idx] < x_min; ++idx) {
                        const uint16_t tid = tree_ids[idx];
                        const uint64_t m64 = (static_cast<uint64_t>(masks_hi[idx]) << 32) | masks_lo[idx];
                        v_vec[tid] = _mm512_and_si512(v_vec[tid], _mm512_set1_epi64(static_cast<int64_t>(m64)));
                    }
                }

                for (; idx < end; ++idx) {
                    const float th = thresholds[idx];
                    if (th >= x_max) break;
                    const __mmask8 gt_k = _mm256_cmp_ps_mask(vx_clean, _mm256_set1_ps(th), _CMP_GT_OQ);
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
        const float* leaves_base = blk.leaf_values.data();

        if (blk.single_class_id >= 0) {
            __m256 acc = _mm256_setzero_ps();
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_idx = Ctz64Avx512(v_vec[t]);
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m256 vals = _mm256_i32gather_ps(tl, leaf_idx, 4);
                acc = _mm256_add_ps(acc, vals);
            }
            float* dst = tile_acc + static_cast<size_t>(blk.single_class_id) * kTileStride + r0;
            _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), acc));
        } else {
            for (uint32_t t = 0; t < n_trees; ++t) {
                const __m256i leaf_idx = Ctz64Avx512(v_vec[t]);
                const float* tl = leaves_base + static_cast<size_t>(t) * stride;
                const __m256 vals = _mm256_i32gather_ps(tl, leaf_idx, 4);
                float* dst = tile_acc + static_cast<size_t>(blk.tree_class_id[t]) * kTileStride + r0;
                _mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), vals));
            }
        }
    }
}

void EvalCompactTrees_Avx512(
    const CompactTreeMeta* trees,
    size_t num_trees,
    const CompactNode* all_nodes,
    const float* all_leaves,
    const float* tile_col_major,
    size_t n_rows,
    float* tile_acc
) noexcept {
    constexpr size_t W = 16;
    const __m512i lane_ids = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i zero_i = _mm512_setzero_si512();
    const __m512i minus_one = _mm512_set1_epi32(-1);
    const __m512i feat_mask = _mm512_set1_epi32(0x00FFFFFF);
    const __m512 zero_eps = _mm512_set1_ps(1e-35f);

    for (size_t r0 = 0; r0 < n_rows; r0 += W) {
        const __m512i lane_r0 = _mm512_add_epi32(lane_ids, _mm512_set1_epi32(static_cast<int32_t>(r0)));

        for (size_t t = 0; t < num_trees; ++t) {
            const CompactTreeMeta& meta = trees[t];
            const CompactNode* nodes = all_nodes + meta.node_offset;
            const float* leaves = all_leaves + meta.leaf_offset;

            // Unroll root node 0 across 16 rows
            const CompactNode& root = nodes[0];
            const __m512 vx0 = _mm512_load_ps(
                tile_col_major + (static_cast<size_t>(root.feature_idx()) << 7) + r0
            );
            __mmask16 go_left0 = _mm512_cmp_ps_mask(vx0, _mm512_set1_ps(root.threshold), _CMP_LE_OQ);
            if (root.nan_goes_left()) {
                go_left0 |= _mm512_cmp_ps_mask(vx0, vx0, _CMP_UNORD_Q);
            }
            if (root.has_zero_missing()) {
                const __mmask16 is_z0 = _mm512_cmp_ps_mask(Abs512(vx0), zero_eps, _CMP_LE_OQ);
                if (root.zero_goes_left()) {
                    go_left0 |= is_z0;
                } else {
                    go_left0 &= ~is_z0;
                }
            }

            __m512i v_idx = _mm512_mask_blend_epi32(
                go_left0,
                _mm512_set1_epi32(root.right_child),
                _mm512_set1_epi32(root.left_child)
            );

            const float* nodes_f32 = reinterpret_cast<const float*>(nodes);
            const int* nodes_i32 = reinterpret_cast<const int*>(nodes);

            while (true) {
                const __mmask16 active = _mm512_cmpge_epi32_mask(v_idx, zero_i);
                if (active == 0) break;

                const __m512i idx4 = _mm512_slli_epi32(v_idx, 2);
                const __m512 v_thresh = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), active, idx4, nodes_f32, 4);
                const __m512i v_ff = _mm512_mask_i32gather_epi32(zero_i, active, idx4, nodes_i32 + 1, 4);
                const __m512i v_left = _mm512_mask_i32gather_epi32(zero_i, active, idx4, nodes_i32 + 2, 4);
                const __m512i v_right = _mm512_mask_i32gather_epi32(zero_i, active, idx4, nodes_i32 + 3, 4);

                const __m512i v_feat = _mm512_and_si512(v_ff, feat_mask);
                const __m512i feat_off = _mm512_add_epi32(_mm512_slli_epi32(v_feat, 7), lane_r0);
                const __m512 vx = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), active, feat_off, tile_col_major, 4);

                const __mmask16 le_k = _mm512_cmp_ps_mask(vx, v_thresh, _CMP_LE_OQ);
                const __mmask16 unord_k = _mm512_cmp_ps_mask(vx, vx, _CMP_UNORD_Q);
                const __mmask16 nan_left_k = _mm512_test_epi32_mask(v_ff, _mm512_set1_epi32(1 << 24));
                __mmask16 go_left = le_k | (unord_k & nan_left_k);

                if (meta.has_zero_missing) {
                    const __mmask16 has_zm_k = _mm512_test_epi32_mask(v_ff, _mm512_set1_epi32(1 << 25));
                    const __mmask16 zm_left_k = _mm512_test_epi32_mask(v_ff, _mm512_set1_epi32(1 << 26));
                    const __mmask16 is_zero_k = _mm512_cmp_ps_mask(Abs512(vx), zero_eps, _CMP_LE_OQ);
                    const __mmask16 ovr = has_zm_k & is_zero_k;
                    go_left = (go_left & ~ovr) | (zm_left_k & ovr);
                }

                const __m512i next_idx = _mm512_mask_blend_epi32(go_left, v_right, v_left);
                v_idx = _mm512_mask_mov_epi32(v_idx, active, next_idx);
            }

            const __m512i leaf_idx = _mm512_xor_si512(v_idx, minus_one);
            const __m512 vals = _mm512_i32gather_ps(leaf_idx, leaves, 4);
            float* dst = tile_acc + static_cast<size_t>(meta.class_id) * kTileStride + r0;
            _mm512_store_ps(dst, _mm512_add_ps(_mm512_load_ps(dst), vals));
        }
    }
}

void FinalizeTileOutput_Avx512(
    float* tile_acc,
    size_t n_rows,
    int num_outputs,
    ObjectiveType objective,
    float sigmoid_alpha,
    float scale_factor,
    bool raw_score,
    float* out_row_major
) noexcept {
    constexpr size_t W = 16;
    const bool need_scale = (scale_factor != 1.0f);
    const __m512 v_scale = _mm512_set1_ps(scale_factor);
    const __m512 v_neg_alpha = _mm512_set1_ps(-sigmoid_alpha);

    if (num_outputs == 1) {
        size_t r = 0;
        for (; r + W <= n_rows; r += W) {
            __m512 v = _mm512_load_ps(tile_acc + r);
            if (need_scale) v = _mm512_mul_ps(v, v_scale);
            if (!raw_score) {
                if (objective == ObjectiveType::BinaryLogloss) {
                    v = simd_math::SigmoidPsAvx512(v, v_neg_alpha);
                } else if (objective == ObjectiveType::Exponential) {
                    v = simd_math::ExpPsAvx512(v);
                }
            }
            _mm512_storeu_ps(out_row_major + r, v);
        }
        if (r < n_rows) {
            const __mmask16 rem = static_cast<__mmask16>((1u << (n_rows - r)) - 1u);
            __m512 v = _mm512_load_ps(tile_acc + r);
            if (need_scale) v = _mm512_mul_ps(v, v_scale);
            if (!raw_score) {
                if (objective == ObjectiveType::BinaryLogloss) {
                    v = simd_math::SigmoidPsAvx512(v, v_neg_alpha);
                } else if (objective == ObjectiveType::Exponential) {
                    v = simd_math::ExpPsAvx512(v);
                }
            }
            _mm512_mask_storeu_ps(out_row_major + r, rem, v);
        }
        return;
    }

    const size_t K = static_cast<size_t>(num_outputs);
    const size_t padded16 = (n_rows + 15u) & ~size_t(15u);

    if (need_scale) {
        for (size_t c = 0; c < K; ++c) {
            float* col = tile_acc + c * kTileStride;
            for (size_t r = 0; r < padded16; r += W) {
                _mm512_store_ps(col + r, _mm512_mul_ps(_mm512_load_ps(col + r), v_scale));
            }
        }
    }

    if (!raw_score) {
        if (objective == ObjectiveType::MulticlassSoftmax) {
            for (size_t r = 0; r < padded16; r += W) {
                __m512 vmax = _mm512_load_ps(tile_acc + r);
                for (size_t c = 1; c < K; ++c) {
                    vmax = _mm512_max_ps(vmax, _mm512_load_ps(tile_acc + c * kTileStride + r));
                }
                __m512 vsum = _mm512_setzero_ps();
                for (size_t c = 0; c < K; ++c) {
                    float* ptr = tile_acc + c * kTileStride + r;
                    __m512 ve = simd_math::ExpPsAvx512(_mm512_sub_ps(_mm512_load_ps(ptr), vmax));
                    _mm512_store_ps(ptr, ve);
                    vsum = _mm512_add_ps(vsum, ve);
                }
                const __m512 vinv = _mm512_div_ps(_mm512_set1_ps(1.0f), vsum);
                for (size_t c = 0; c < K; ++c) {
                    float* ptr = tile_acc + c * kTileStride + r;
                    _mm512_store_ps(ptr, _mm512_mul_ps(_mm512_load_ps(ptr), vinv));
                }
            }
        } else if (objective == ObjectiveType::BinaryLogloss) {
            for (size_t c = 0; c < K; ++c) {
                float* col = tile_acc + c * kTileStride;
                for (size_t r = 0; r < padded16; r += W) {
                    _mm512_store_ps(col + r, simd_math::SigmoidPsAvx512(_mm512_load_ps(col + r), v_neg_alpha));
                }
            }
        } else if (objective == ObjectiveType::Exponential) {
            for (size_t c = 0; c < K; ++c) {
                float* col = tile_acc + c * kTileStride;
                for (size_t r = 0; r < padded16; r += W) {
                    _mm512_store_ps(col + r, simd_math::ExpPsAvx512(_mm512_load_ps(col + r)));
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
