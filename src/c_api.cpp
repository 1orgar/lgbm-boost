#include "fast_scorer/c_api.h"
#include "fast_scorer/compiler.h"
#include "fast_scorer/scorer.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

thread_local std::string g_last_error;

int SetError(int code, const char* msg) noexcept {
    try {
        g_last_error = (msg != nullptr) ? msg : "Unknown error";
    } catch (...) {
    }
    return code;
}

int HandleException() noexcept {
    try {
        throw;
    } catch (const std::invalid_argument& ex) {
        return SetError(-1, ex.what());
    } catch (const std::exception& ex) {
        return SetError(-2, ex.what());
    } catch (...) {
        return SetError(-2, "Unknown C++ exception in lgbm_boost");
    }
}

fast_scorer::SimdIsa MapIsaId(int32_t id) {
    switch (id) {
        case 0: return fast_scorer::SimdIsa::Auto;
        case 1: return fast_scorer::SimdIsa::Scalar;
        case 2: return fast_scorer::SimdIsa::Avx2;
        case 3: return fast_scorer::SimdIsa::Avx512;
        default:
            throw std::invalid_argument("Invalid SIMD ISA id");
    }
}

fast_scorer::KernelKind MapKernelId(int32_t id) {
    switch (id) {
        case 0: return fast_scorer::KernelKind::Auto;
        case 1: return fast_scorer::KernelKind::QuickScorer;
        case 2: return fast_scorer::KernelKind::CompactTree;
        default:
            throw std::invalid_argument("Invalid kernel id");
    }
}

fast_scorer::ColumnDType MapColumnDType(uint8_t dt) {
    if (dt > static_cast<uint8_t>(fast_scorer::ColumnDType::UInt32)) {
        throw std::invalid_argument("Unsupported column dtype code");
    }
    return static_cast<fast_scorer::ColumnDType>(dt);
}

const fast_scorer::FastEnsembleScorer* Unwrap(LGBMBoostHandle h) {
    if (h == nullptr) {
        throw std::invalid_argument("Null LGBMBoostHandle");
    }
    return static_cast<const fast_scorer::FastEnsembleScorer*>(h);
}

} // namespace

extern "C" {

const char* lgbm_boost_get_last_error(void) {
    return g_last_error.c_str();
}

const char* lgbm_boost_detect_best_isa(void) {
    return fast_scorer::SimdIsaName(fast_scorer::DetectBestSimdIsa());
}

int lgbm_boost_is_isa_available(int isa_id, int* out_available) {
    try {
        if (out_available == nullptr) {
            throw std::invalid_argument("out_available is null");
        }
        *out_available = fast_scorer::IsSimdIsaAvailable(MapIsaId(isa_id)) ? 1 : 0;
        return 0;
    } catch (...) {
        return HandleException();
    }
}

int lgbm_boost_create_from_string(
    const char* model_str,
    uint64_t model_len,
    uint32_t qs32_block_trees,
    uint32_t qs64_block_trees,
    LGBMBoostHandle* out_handle
) {
    try {
        if (model_str == nullptr || out_handle == nullptr) {
            throw std::invalid_argument("Null argument passed to lgbm_boost_create_from_string");
        }
        *out_handle = nullptr;
        fast_scorer::CompileOptions copts;
        copts.qs32_block_trees = qs32_block_trees;
        copts.qs64_block_trees = qs64_block_trees;

        auto scorer = std::make_unique<fast_scorer::FastEnsembleScorer>(
            fast_scorer::FastEnsembleScorer::FromModelString(
                std::string_view(model_str, static_cast<size_t>(model_len)), copts
            )
        );
        if (!scorer->is_supported()) {
            throw std::invalid_argument(scorer->unsupported_reason());
        }
        *out_handle = static_cast<LGBMBoostHandle>(scorer.release());
        return 0;
    } catch (...) {
        return HandleException();
    }
}

int lgbm_boost_create_from_file(
    const char* filepath,
    uint32_t qs32_block_trees,
    uint32_t qs64_block_trees,
    LGBMBoostHandle* out_handle
) {
    try {
        if (filepath == nullptr || out_handle == nullptr) {
            throw std::invalid_argument("Null argument passed to lgbm_boost_create_from_file");
        }
        *out_handle = nullptr;
        fast_scorer::CompileOptions copts;
        copts.qs32_block_trees = qs32_block_trees;
        copts.qs64_block_trees = qs64_block_trees;

        auto scorer = std::make_unique<fast_scorer::FastEnsembleScorer>(
            fast_scorer::FastEnsembleScorer::FromModelFile(std::string(filepath), copts)
        );
        if (!scorer->is_supported()) {
            throw std::invalid_argument(scorer->unsupported_reason());
        }
        *out_handle = static_cast<LGBMBoostHandle>(scorer.release());
        return 0;
    } catch (...) {
        return HandleException();
    }
}

void lgbm_boost_free(LGBMBoostHandle handle) {
    if (handle != nullptr) {
        delete static_cast<fast_scorer::FastEnsembleScorer*>(handle);
    }
}

int lgbm_boost_get_metadata(
    LGBMBoostHandle handle,
    int32_t* num_features,
    int32_t* num_outputs,
    int32_t* total_iterations,
    int32_t* total_trees,
    int32_t* max_num_leaves,
    uint64_t* num_qs32_blocks,
    uint64_t* num_qs64_blocks,
    uint64_t* num_deep_trees
) {
    try {
        const auto* s = Unwrap(handle);
        const auto& cm = s->compiled_model();
        if (num_features) *num_features = s->num_features();
        if (num_outputs) *num_outputs = s->num_outputs();
        if (total_iterations) *total_iterations = s->total_iterations();
        if (total_trees) *total_trees = s->total_trees();
        if (max_num_leaves) *max_num_leaves = cm.max_num_leaves;
        if (num_qs32_blocks) *num_qs32_blocks = static_cast<uint64_t>(cm.qs32_blocks.size());
        if (num_qs64_blocks) *num_qs64_blocks = static_cast<uint64_t>(cm.qs64_blocks.size());
        if (num_deep_trees) *num_deep_trees = static_cast<uint64_t>(cm.deep_compact_trees.size());
        return 0;
    } catch (...) {
        return HandleException();
    }
}

uint64_t lgbm_boost_num_feature_names(LGBMBoostHandle handle) {
    if (handle == nullptr) return 0;
    const auto* s = static_cast<const fast_scorer::FastEnsembleScorer*>(handle);
    return static_cast<uint64_t>(s->compiled_model().feature_names.size());
}

const char* lgbm_boost_get_feature_name(LGBMBoostHandle handle, uint64_t idx) {
    if (handle == nullptr) return "";
    const auto* s = static_cast<const fast_scorer::FastEnsembleScorer*>(handle);
    const auto& names = s->compiled_model().feature_names;
    if (idx >= names.size()) return "";
    return names[static_cast<size_t>(idx)].c_str();
}

int lgbm_boost_predict_row_major_f32(
    LGBMBoostHandle handle,
    const float* data,
    uint64_t n_rows,
    uint64_t n_cols,
    int64_t row_stride_elems,
    double* out_f64,
    float* out_f32,
    int32_t start_iteration,
    int32_t num_iteration,
    int32_t raw_score,
    int32_t num_threads,
    int32_t isa_id,
    int32_t kernel_id
) {
    try {
        const auto* s = Unwrap(handle);
        fast_scorer::PredictOptions opts;
        opts.start_iteration = start_iteration;
        opts.num_iteration = num_iteration;
        opts.raw_score = (raw_score != 0);
        opts.num_threads = num_threads;
        opts.isa = MapIsaId(isa_id);
        opts.kernel = MapKernelId(kernel_id);

        s->PredictRowMajorF32(
            data,
            static_cast<size_t>(n_rows),
            static_cast<size_t>(n_cols),
            static_cast<ptrdiff_t>(row_stride_elems),
            out_f64,
            out_f32,
            opts
        );
        return 0;
    } catch (...) {
        return HandleException();
    }
}

int lgbm_boost_predict_row_major_f64(
    LGBMBoostHandle handle,
    const double* data,
    uint64_t n_rows,
    uint64_t n_cols,
    int64_t row_stride_elems,
    double* out_f64,
    float* out_f32,
    int32_t start_iteration,
    int32_t num_iteration,
    int32_t raw_score,
    int32_t num_threads,
    int32_t isa_id,
    int32_t kernel_id
) {
    try {
        const auto* s = Unwrap(handle);
        fast_scorer::PredictOptions opts;
        opts.start_iteration = start_iteration;
        opts.num_iteration = num_iteration;
        opts.raw_score = (raw_score != 0);
        opts.num_threads = num_threads;
        opts.isa = MapIsaId(isa_id);
        opts.kernel = MapKernelId(kernel_id);

        s->PredictRowMajorF64(
            data,
            static_cast<size_t>(n_rows),
            static_cast<size_t>(n_cols),
            static_cast<ptrdiff_t>(row_stride_elems),
            out_f64,
            out_f32,
            opts
        );
        return 0;
    } catch (...) {
        return HandleException();
    }
}

int lgbm_boost_predict_columnar(
    LGBMBoostHandle handle,
    const void* const* col_ptrs,
    const uint8_t* col_dtypes,
    uint64_t n_rows,
    uint64_t n_cols,
    double* out_f64,
    float* out_f32,
    int32_t start_iteration,
    int32_t num_iteration,
    int32_t raw_score,
    int32_t num_threads,
    int32_t isa_id,
    int32_t kernel_id
) {
    try {
        const auto* s = Unwrap(handle);
        if (n_cols == 0 || col_ptrs == nullptr || col_dtypes == nullptr) {
            throw std::invalid_argument("Invalid columnar arguments");
        }
        std::vector<fast_scorer::ColumnView> views(static_cast<size_t>(n_cols));
        for (size_t c = 0; c < static_cast<size_t>(n_cols); ++c) {
            views[c].data = col_ptrs[c];
            views[c].dtype = MapColumnDType(col_dtypes[c]);
        }

        fast_scorer::PredictOptions opts;
        opts.start_iteration = start_iteration;
        opts.num_iteration = num_iteration;
        opts.raw_score = (raw_score != 0);
        opts.num_threads = num_threads;
        opts.isa = MapIsaId(isa_id);
        opts.kernel = MapKernelId(kernel_id);

        s->PredictColumnar(
            views.data(),
            static_cast<size_t>(n_rows),
            static_cast<size_t>(n_cols),
            out_f64,
            out_f32,
            opts
        );
        return 0;
    } catch (...) {
        return HandleException();
    }
}

} // extern "C"
