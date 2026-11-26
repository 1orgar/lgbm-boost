#pragma once

#include "fast_scorer/model.h"

namespace fast_scorer {

struct CompileOptions {
    uint32_t qs32_block_trees = 64;
    uint32_t qs64_block_trees = 64;
};

CompiledModel CompileModel(const RawModel& raw, const CompileOptions& opts = {});

// Exposed for unit testing exact float64 -> float32 threshold lowering
float LowerThresholdToFloat32(double t64) noexcept;

} // namespace fast_scorer
