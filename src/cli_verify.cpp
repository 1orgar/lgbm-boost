#include "fast_scorer/scorer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

fast_scorer::SimdIsa ParseIsaCli(const std::string& s) {
    if (s == "scalar") return fast_scorer::SimdIsa::Scalar;
    if (s == "avx2") return fast_scorer::SimdIsa::Avx2;
    if (s == "avx512") return fast_scorer::SimdIsa::Avx512;
    return fast_scorer::SimdIsa::Auto;
}

fast_scorer::KernelKind ParseKernelCli(const std::string& s) {
    if (s == "quickscorer" || s == "qs") return fast_scorer::KernelKind::QuickScorer;
    if (s == "compact" || s == "tree") return fast_scorer::KernelKind::CompactTree;
    return fast_scorer::KernelKind::Auto;
}

std::vector<float> ReadFloatBin(const std::string& path, size_t count) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open binary file: " + path);
    }
    std::vector<float> buf(count);
    if (count > 0 && !ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(count * sizeof(float)))) {
        throw std::runtime_error("Short read on binary file: " + path);
    }
    return buf;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(
            stderr,
            "Usage: %s <model.txt> <X_f32.bin> <expected_f32.bin> <n_rows> <n_cols> <n_outputs> <raw_score:0|1> [isa] [kernel] [threads] [tol]\n",
            argv[0]
        );
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const std::string x_path = argv[2];
        const std::string exp_path = argv[3];
        const size_t n_rows = static_cast<size_t>(std::stoull(argv[4]));
        const size_t n_cols = static_cast<size_t>(std::stoull(argv[5]));
        const size_t n_outputs = static_cast<size_t>(std::stoull(argv[6]));
        const bool raw_score = (std::stoi(argv[7]) != 0);
        const std::string isa_str = (argc >= 9) ? argv[8] : "auto";
        const std::string kernel_str = (argc >= 10) ? argv[9] : "auto";
        const int num_threads = (argc >= 11) ? std::stoi(argv[10]) : 0;
        const float tol = (argc >= 12) ? std::stof(argv[11]) : 2e-5f;

        auto scorer = fast_scorer::FastEnsembleScorer::FromModelFile(model_path);
        if (!scorer.is_supported()) {
            std::fprintf(stderr, "Unsupported model: %s\n", scorer.unsupported_reason().c_str());
            return 3;
        }

        const std::vector<float> X = ReadFloatBin(x_path, n_rows * n_cols);
        const std::vector<float> expected = ReadFloatBin(exp_path, n_rows * n_outputs);
        std::vector<float> actual(n_rows * n_outputs, 0.0f);

        fast_scorer::PredictOptions opts;
        opts.raw_score = raw_score;
        opts.isa = ParseIsaCli(isa_str);
        opts.kernel = ParseKernelCli(kernel_str);
        opts.num_threads = num_threads;

        // Warmup + timed run
        scorer.PredictRowMajorF32(X.data(), n_rows, n_cols, static_cast<ptrdiff_t>(n_cols), actual.data(), opts);

        const auto t0 = std::chrono::steady_clock::now();
        constexpr int kRepeats = 5;
        for (int r = 0; r < kRepeats; ++r) {
            scorer.PredictRowMajorF32(X.data(), n_rows, n_cols, static_cast<ptrdiff_t>(n_cols), actual.data(), opts);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / kRepeats;

        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        size_t worst_idx = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float diff = std::fabs(actual[i] - expected[i]);
            const float denom = std::max(1.0f, std::fabs(expected[i]));
            const float rel = diff / denom;
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                worst_idx = i;
            }
            if (rel > max_rel_diff) {
                max_rel_diff = rel;
            }
        }

        std::printf(
            "best_isa=%s requested_isa=%s kernel=%s rows=%zu avg_ms=%.3f throughput=%.2fM_rows/s max_abs_diff=%.3e max_rel_diff=%.3e\n",
            fast_scorer::SimdIsaName(fast_scorer::DetectBestSimdIsa()),
            isa_str.c_str(),
            kernel_str.c_str(),
            n_rows,
            avg_ms,
            (static_cast<double>(n_rows) / (avg_ms * 1e3)),
            max_abs_diff,
            max_rel_diff
        );

        if (max_rel_diff > tol && max_abs_diff > tol) {
            std::fprintf(
                stderr,
                "FAIL: worst_idx=%zu actual=%.8f expected=%.8f max_abs_diff=%.3e max_rel_diff=%.3e > tol=%.3e\n",
                worst_idx,
                actual[worst_idx],
                expected[worst_idx],
                max_abs_diff,
                max_rel_diff,
                tol
            );
            return 1;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Exception: %s\n", ex.what());
        return 1;
    }
}
