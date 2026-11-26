#pragma once

// Drop-in integration header for LightGBM's src/c_api.cpp (Stage 4 Fork Integration).
// Lazily compiles FastEnsembleScorer inside LightGBM::Booster and intercepts dense
// row-major batch calls in LGBM_BoosterPredictForMat with 100% transparent fallback
// for categorical trees, linear trees, SHAP (pred_contrib), leaf indices, or early stopping.

#include <LightGBM/boosting.h>
#include <LightGBM/c_api.h>
#include <LightGBM/config.h>
#include <LightGBM/fast_scorer/scorer.h>

#include <memory>
#include <mutex>
#include <string>

namespace LightGBM {

class FastBatchScorerCache {
public:
    bool TryPredictRowMajor(
        const Boosting* boosting,
        const void* data,
        int data_type,
        int32_t nrow,
        int32_t ncol,
        int predict_type,
        int start_iteration,
        int num_iteration,
        const Config& config,
        int64_t* out_len,
        double* out_result
    ) {
        // Only intercept standard normal / raw score batch predictions without pred_early_stop
        if (predict_type != C_API_PREDICT_NORMAL && predict_type != C_API_PREDICT_RAW_SCORE) {
            return false;
        }
        if (config.pred_early_stop || nrow < 32) {
            return false;
        }
        if (data_type != C_API_DTYPE_FLOAT32 && data_type != C_API_DTYPE_FLOAT64) {
            return false;
        }

        const fast_scorer::FastEnsembleScorer* scorer = GetOrCompile(boosting);
        if (scorer == nullptr || !scorer->is_supported()) {
            return false;
        }
        if (ncol != scorer->num_features()) {
            return false;
        }

        fast_scorer::PredictOptions opts;
        opts.start_iteration = start_iteration;
        opts.num_iteration = num_iteration;
        opts.raw_score = (predict_type == C_API_PREDICT_RAW_SCORE);
        opts.num_threads = config.num_threads;

        const float* f32_ptr = (data_type == C_API_DTYPE_FLOAT32)
                                   ? static_cast<const float*>(data)
                                   : nullptr;
        const double* f64_ptr = (data_type == C_API_DTYPE_FLOAT64)
                                    ? static_cast<const double*>(data)
                                    : nullptr;

        scorer->PredictRowMajorToF64(
            f64_ptr,
            f32_ptr,
            static_cast<size_t>(nrow),
            static_cast<size_t>(ncol),
            static_cast<ptrdiff_t>(ncol),
            out_result,
            opts
        );

        *out_len = static_cast<int64_t>(nrow) * static_cast<int64_t>(scorer->num_outputs());
        return true;
    }

    void Invalidate() {
        std::lock_guard<std::mutex> lock(mtx_);
        scorer_.reset();
        compiled_tree_count_ = -1;
    }

private:
    const fast_scorer::FastEnsembleScorer* GetOrCompile(const Boosting* boosting) {
        if (boosting == nullptr) return nullptr;
        const int cur_trees = boosting->NumberOfTotalModel();
        if (cur_trees <= 0) return nullptr;

        std::lock_guard<std::mutex> lock(mtx_);
        if (scorer_ == nullptr || compiled_tree_count_ != cur_trees) {
            std::string model_txt = boosting->SaveModelToString(0, -1, C_API_FEATURE_IMPORTANCE_SPLIT);
            scorer_ = std::make_unique<fast_scorer::FastEnsembleScorer>(
                fast_scorer::FastEnsembleScorer::FromModelString(model_txt)
            );
            compiled_tree_count_ = cur_trees;
        }
        return scorer_.get();
    }

    std::mutex mtx_;
    std::unique_ptr<fast_scorer::FastEnsembleScorer> scorer_;
    int compiled_tree_count_ = -1;
};

} // namespace LightGBM
