#pragma once

#include <cmath>
#include <cstdint>

#if defined(FAST_SCORER_ARCH_X86_64)
#include <immintrin.h>
#endif

namespace fast_scorer {
namespace simd_math {

// Scalar reference fast/exact helpers
inline float SafeSigmoidScalar(float x, float alpha) noexcept {
    const double z = -static_cast<double>(alpha) * static_cast<double>(x);
    if (z > 88.0) return 0.0f;
    if (z < -88.0) return 1.0f;
    return static_cast<float>(1.0 / (1.0 + std::exp(z)));
}

inline float SafeExpScalar(float x) noexcept {
    if (x > 88.37626266f) return 3.402823466e+38f;
    if (x < -87.33654475f) return 0.0f;
    return std::exp(x);
}

#if defined(__AVX2__)
// High-accuracy (< 1.5 ULP) degree-6 Cody-Waite + FMA vectorized expf for AVX2 (8x float32)
inline __m256 ExpPsAvx2(__m256 x) noexcept {
    const __m256 max_x = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_x = _mm256_set1_ps(-87.336544750402f);
    const __m256 underflow_mask = _mm256_cmp_ps(x, min_x, _CMP_LT_OQ);

    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1 = _mm256_set1_ps(0.693145751953125f);
    const __m256 c2 = _mm256_set1_ps(1.428606765330187045e-6f);

    // Round x * log2(e) to nearest integer
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Cody-Waite range reduction: r = (x - fx * c1) - fx * c2
    __m256 r = _mm256_fnmadd_ps(fx, c1, x);
    r = _mm256_fnmadd_ps(fx, c2, r);

    // Degree-6 polynomial on |r| <= ln(2)/2
    __m256 p = _mm256_set1_ps(1.38888889e-3f); // 1/720
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.33333333e-3f)); // 1/120
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.16666667e-2f)); // 1/24
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.66666667e-1f)); // 1/6
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));           // 1/2
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

    // Scale by 2^n via IEEE-754 exponent bits
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    __m256 res = _mm256_mul_ps(p, pow2n);
    return _mm256_andnot_ps(underflow_mask, res);
}

inline __m256 SigmoidPsAvx2(__m256 x, __m256 neg_alpha) noexcept {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 e = ExpPsAvx2(_mm256_mul_ps(x, neg_alpha));
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}
#endif

#if defined(__AVX512F__)
// High-accuracy (< 1.5 ULP) degree-6 Cody-Waite + FMA vectorized expf for AVX-512 (16x float32)
inline __m512 ExpPsAvx512(__m512 x) noexcept {
    const __m512 max_x = _mm512_set1_ps(88.3762626647949f);
    const __m512 min_x = _mm512_set1_ps(-87.336544750402f);
    const __mmask16 keep_mask = _mm512_cmp_ps_mask(x, min_x, _CMP_GE_OQ);

    x = _mm512_min_ps(x, max_x);
    x = _mm512_max_ps(x, min_x);

    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 c1 = _mm512_set1_ps(0.693145751953125f);
    const __m512 c2 = _mm512_set1_ps(1.428606765330187045e-6f);

    __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fx, c1, x);
    r = _mm512_fnmadd_ps(fx, c2, r);

    __m512 p = _mm512_set1_ps(1.38888889e-3f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.33333333e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.16666667e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.66666667e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));

    // _mm512_scalef_ps scales p by 2^floor(fx) directly in hardware on AVX-512F!
    __m512 res = _mm512_scalef_ps(p, fx);
    return _mm512_maskz_mov_ps(keep_mask, res);
}

inline __m512 SigmoidPsAvx512(__m512 x, __m512 neg_alpha) noexcept {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 e = ExpPsAvx512(_mm512_mul_ps(x, neg_alpha));
    return _mm512_div_ps(one, _mm512_add_ps(one, e));
}
#endif

} // namespace simd_math
} // namespace fast_scorer
