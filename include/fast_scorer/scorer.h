#pragma once

#include "fast_scorer/compiler.h"
#include "fast_scorer/model.h"
#include "fast_scorer/parser.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace fast_scorer {

SimdIsa DetectBestSimdIsa() noexcept;
const char* SimdIsaName(SimdIsa isa) noexcept;
bool IsSimdIsaAvailable(SimdIsa isa) noexcept;

class FastEnsembleScorer {
public:
    FastEnsembleScorer() = default;
    explicit FastEnsembleScorer(CompiledModel compiled);

    static FastEnsembleScorer FromModelString(
        std::string_view content,
        const CompileOptions& opts = {}
    );
    static FastEnsembleScorer FromModelFile(
        const std::string& filepath,
        const CompileOptions& opts = {}
    );
    static FastEnsembleScorer FromRawModel(
        const RawModel& raw,
        const CompileOptions& opts = {}
    );

    bool is_supported() const noexcept { return model_.supported; }
    const std::string& unsupported_reason() const noexcept { return model_.unsupported_reason; }
    const CompiledModel& compiled_model() const noexcept { return model_; }

    int num_features() const noexcept { return model_.num_features; }
    int num_outputs() const noexcept { return model_.num_tree_per_iteration; }
    int total_iterations() const noexcept { return model_.total_iterations; }
    int total_trees() const noexcept { return model_.total_trees; }

    // Batch prediction for row-major float32 matrix -> float64 or float32 output
    void PredictRowMajorF32(
        const float* data,
        size_t n_rows,
        size_t n_cols,
        ptrdiff_t row_stride_elems,
        double* out_scores_f64,
        float* out_scores_f32,
        const PredictOptions& opts = {}
    ) const;

    // Batch prediction for row-major float64 matrix -> float64 or float32 output
    void PredictRowMajorF64(
        const double* data,
        size_t n_rows,
        size_t n_cols,
        ptrdiff_t row_stride_elems,
        double* out_scores_f64,
        float* out_scores_f32,
        const PredictOptions& opts = {}
    ) const;

    // Batch prediction directly from columnar arrays (e.g. pandas.DataFrame / Polars zero-copy)
    void PredictColumnar(
        const ColumnView* columns,
        size_t n_rows,
        size_t n_cols,
        double* out_scores_f64,
        float* out_scores_f32,
        const PredictOptions& opts = {}
    ) const;

    // Convenience overload writing double outputs (for LightGBM C-API drop-in compatibility)
    void PredictRowMajorToF64(
        const double* data_f64,
        const float* data_f32,
        size_t n_rows,
        size_t n_cols,
        ptrdiff_t row_stride_elems,
        double* out_scores_f64,
        const PredictOptions& opts = {}
    ) const;

private:
    enum class InputKind : uint8_t {
        RowMajorF32 = 0,
        RowMajorF64 = 1,
        Columnar = 2
    };

    struct InputDescriptor {
        InputKind kind = InputKind::RowMajorF32;
        const float* f32_ptr = nullptr;
        const double* f64_ptr = nullptr;
        const ColumnView* cols_ptr = nullptr;
        size_t n_rows = 0;
        size_t n_cols = 0;
        ptrdiff_t row_stride = 0;
    };

    void ExecuteBatch(
        const InputDescriptor& input,
        double* out_f64,
        float* out_f32,
        const PredictOptions& opts
    ) const;

    CompiledModel model_;
};

} // namespace fast_scorer
