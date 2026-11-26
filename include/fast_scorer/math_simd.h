#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace fast_scorer {
namespace simd_math {

// Exact IEEE-754 float64 Sigmoid matching LightGBM's 1.0 / (1.0 + std::exp(-alpha * x))
inline double SafeSigmoidF64(double x, double alpha) noexcept {
    const double z = -alpha * x;
    if (z > 709.0) return 0.0;
    if (z < -709.0) return 1.0;
    return 1.0 / (1.0 + std::exp(z));
}

// Exact IEEE-754 float64 Exp matching LightGBM's std::exp(x)
inline double SafeExpF64(double x) noexcept {
    if (x > 709.782712893384) return std::numeric_limits<double>::infinity();
    if (x < -708.3964185322641) return 0.0;
    return std::exp(x);
}

} // namespace simd_math
} // namespace fast_scorer
