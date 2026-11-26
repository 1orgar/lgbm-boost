#pragma once

#include "fast_scorer/model.h"

namespace fast_scorer {

struct CompileOptions {
    uint32_t qs32_block_trees = 64;
    uint32_t qs64_block_trees = 64;
};

CompiledModel CompileModel(const RawModel& raw, const CompileOptions& opts = {});

} // namespace fast_scorer
