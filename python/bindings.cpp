#include "fast_scorer/compiler.h"
#include "fast_scorer/scorer.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

fast_scorer::SimdIsa ParseIsa(const std::string& s) {
    if (s == "auto" || s.empty()) return fast_scorer::SimdIsa::Auto;
    if (s == "scalar") return fast_scorer::SimdIsa::Scalar;
    if (s == "avx2") return fast_scorer::SimdIsa::Avx2;
    if (s == "avx512") return fast_scorer::SimdIsa::Avx512;
    throw std::invalid_argument("Unknown SIMD ISA '" + s + "'. Expected 'auto', 'scalar', 'avx2', or 'avx512'.");
}

fast_scorer::KernelKind ParseKernel(const std::string& s) {
    if (s == "auto" || s.empty()) return fast_scorer::KernelKind::Auto;
    if (s == "quickscorer" || s == "qs") return fast_scorer::KernelKind::QuickScorer;
    if (s == "compact" || s == "tree") return fast_scorer::KernelKind::CompactTree;
    throw std::invalid_argument("Unknown kernel '" + s + "'. Expected 'auto', 'quickscorer', or 'compact'.");
}

fast_scorer::ColumnDType MapDlpackDType(nb::dlpack::dtype dt) {
    // dlpack code: 0 = int, 1 = uint, 2 = float, 4 = bfloat, 6 = bool
    if (dt.lanes != 1) {
        throw std::invalid_argument("Vector-lane dtypes are not supported for columnar input");
    }
    if (dt.code == 2) { // float
        if (dt.bits == 32) return fast_scorer::ColumnDType::Float32;
        if (dt.bits == 64) return fast_scorer::ColumnDType::Float64;
    } else if (dt.code == 0) { // signed int
        if (dt.bits == 8) return fast_scorer::ColumnDType::Int8;
        if (dt.bits == 16) return fast_scorer::ColumnDType::Int16;
        if (dt.bits == 32) return fast_scorer::ColumnDType::Int32;
        if (dt.bits == 64) return fast_scorer::ColumnDType::Int64;
    } else if (dt.code == 1 || dt.code == 6) { // unsigned int / bool
        if (dt.bits == 8) return fast_scorer::ColumnDType::UInt8;
        if (dt.bits == 16) return fast_scorer::ColumnDType::UInt16;
        if (dt.bits == 32) return fast_scorer::ColumnDType::UInt32;
    }
    throw std::invalid_argument("Unsupported column dtype for zero-copy columnar scoring");
}

} // namespace

NB_MODULE(_lgbm_boost_ext, m) {
    m.doc() = "C++17 SIMD batch inference engine for LightGBM tree ensembles";

    m.def("detect_best_isa", []() {
        return std::string(fast_scorer::SimdIsaName(fast_scorer::DetectBestSimdIsa()));
    });

    m.def("is_isa_available", [](const std::string& isa) {
        return fast_scorer::IsSimdIsaAvailable(ParseIsa(isa));
    });

    m.def("lower_threshold_to_f32", &fast_scorer::LowerThresholdToFloat32, "t64"_a);

    nb::class_<fast_scorer::FastEnsembleScorer>(m, "FastEnsembleScorer")
        .def_static(
            "from_model_string",
            [](const std::string& model_str, uint32_t qs32_block_trees, uint32_t qs64_block_trees) {
                fast_scorer::CompileOptions copts;
                copts.qs32_block_trees = qs32_block_trees;
                copts.qs64_block_trees = qs64_block_trees;
                nb::gil_scoped_release release;
                auto scorer = fast_scorer::FastEnsembleScorer::FromModelString(model_str, copts);
                if (!scorer.is_supported()) {
                    throw std::invalid_argument(scorer.unsupported_reason());
                }
                return scorer;
            },
            "model_str"_a,
            "qs32_block_trees"_a = 64,
            "qs64_block_trees"_a = 64
        )
        .def_static(
            "from_model_file",
            [](const std::string& filepath, uint32_t qs32_block_trees, uint32_t qs64_block_trees) {
                fast_scorer::CompileOptions copts;
                copts.qs32_block_trees = qs32_block_trees;
                copts.qs64_block_trees = qs64_block_trees;
                nb::gil_scoped_release release;
                auto scorer = fast_scorer::FastEnsembleScorer::FromModelFile(filepath, copts);
                if (!scorer.is_supported()) {
                    throw std::invalid_argument(scorer.unsupported_reason());
                }
                return scorer;
            },
            "filepath"_a,
            "qs32_block_trees"_a = 64,
            "qs64_block_trees"_a = 64
        )
        .def_prop_ro("num_features", &fast_scorer::FastEnsembleScorer::num_features)
        .def_prop_ro("num_outputs", &fast_scorer::FastEnsembleScorer::num_outputs)
        .def_prop_ro("total_iterations", &fast_scorer::FastEnsembleScorer::total_iterations)
        .def_prop_ro("total_trees", &fast_scorer::FastEnsembleScorer::total_trees)
        .def_prop_ro("max_num_leaves", [](const fast_scorer::FastEnsembleScorer& self) {
            return self.compiled_model().max_num_leaves;
        })
        .def_prop_ro("num_qs32_blocks", [](const fast_scorer::FastEnsembleScorer& self) {
            return self.compiled_model().qs32_blocks.size();
        })
        .def_prop_ro("num_qs64_blocks", [](const fast_scorer::FastEnsembleScorer& self) {
            return self.compiled_model().qs64_blocks.size();
        })
        .def_prop_ro("num_deep_trees", [](const fast_scorer::FastEnsembleScorer& self) {
            return self.compiled_model().deep_compact_trees.size();
        })
        .def_prop_ro("feature_names", [](const fast_scorer::FastEnsembleScorer& self) {
            return self.compiled_model().feature_names;
        })
        .def(
            "predict_f32",
            [](const fast_scorer::FastEnsembleScorer& self,
               nb::ndarray<const float, nb::ndim<2>, nb::device::cpu> X,
               nb::ndarray<float, nb::c_contig, nb::device::cpu> out,
               int start_iteration,
               int num_iteration,
               bool raw_score,
               int num_threads,
               const std::string& isa,
               const std::string& kernel) {
                if (X.shape(1) > 1 && X.stride(1) != 1) {
                    throw std::invalid_argument("X must have contiguous rows (stride(1) == 1)");
                }
                const size_t n_rows = X.shape(0);
                const size_t n_cols = X.shape(1);
                const ptrdiff_t row_stride = static_cast<ptrdiff_t>(X.stride(0));
                const float* in_ptr = X.data();
                if (out.size() != n_rows * static_cast<size_t>(self.num_outputs())) {
                    throw std::invalid_argument("Output buffer size mismatch");
                }
                float* out_ptr = out.data();

                fast_scorer::PredictOptions popts;
                popts.start_iteration = start_iteration;
                popts.num_iteration = num_iteration;
                popts.raw_score = raw_score;
                popts.num_threads = num_threads;
                popts.isa = ParseIsa(isa);
                popts.kernel = ParseKernel(kernel);

                nb::gil_scoped_release release;
                self.PredictRowMajorF32(in_ptr, n_rows, n_cols, row_stride, out_ptr, popts);
            },
            "X"_a,
            "out"_a,
            "start_iteration"_a = 0,
            "num_iteration"_a = -1,
            "raw_score"_a = false,
            "num_threads"_a = 0,
            "isa"_a = "auto",
            "kernel"_a = "auto"
        )
        .def(
            "predict_f64",
            [](const fast_scorer::FastEnsembleScorer& self,
               nb::ndarray<const double, nb::ndim<2>, nb::device::cpu> X,
               nb::ndarray<float, nb::c_contig, nb::device::cpu> out,
               int start_iteration,
               int num_iteration,
               bool raw_score,
               int num_threads,
               const std::string& isa,
               const std::string& kernel) {
                if (X.shape(1) > 1 && X.stride(1) != 1) {
                    throw std::invalid_argument("X must have contiguous rows (stride(1) == 1)");
                }
                const size_t n_rows = X.shape(0);
                const size_t n_cols = X.shape(1);
                const ptrdiff_t row_stride = static_cast<ptrdiff_t>(X.stride(0));
                const double* in_ptr = X.data();
                if (out.size() != n_rows * static_cast<size_t>(self.num_outputs())) {
                    throw std::invalid_argument("Output buffer size mismatch");
                }
                float* out_ptr = out.data();

                fast_scorer::PredictOptions popts;
                popts.start_iteration = start_iteration;
                popts.num_iteration = num_iteration;
                popts.raw_score = raw_score;
                popts.num_threads = num_threads;
                popts.isa = ParseIsa(isa);
                popts.kernel = ParseKernel(kernel);

                nb::gil_scoped_release release;
                self.PredictRowMajorF64(in_ptr, n_rows, n_cols, row_stride, out_ptr, popts);
            },
            "X"_a,
            "out"_a,
            "start_iteration"_a = 0,
            "num_iteration"_a = -1,
            "raw_score"_a = false,
            "num_threads"_a = 0,
            "isa"_a = "auto",
            "kernel"_a = "auto"
        )
        .def(
            "predict_columnar",
            [](const fast_scorer::FastEnsembleScorer& self,
               nb::list columns,
               nb::ndarray<float, nb::c_contig, nb::device::cpu> out,
               int start_iteration,
               int num_iteration,
               bool raw_score,
               int num_threads,
               const std::string& isa,
               const std::string& kernel) {
                const size_t n_cols = nb::len(columns);
                if (n_cols == 0) {
                    throw std::invalid_argument("Empty column list");
                }
                std::vector<nb::ndarray<nb::ro, nb::ndim<1>, nb::c_contig, nb::device::cpu>> holders;
                holders.reserve(n_cols);
                std::vector<fast_scorer::ColumnView> views(n_cols);

                size_t n_rows = 0;
                for (size_t c = 0; c < n_cols; ++c) {
                    auto arr = nb::cast<nb::ndarray<nb::ro, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(columns[c]);
                    if (c == 0) {
                        n_rows = arr.shape(0);
                    } else if (arr.shape(0) != n_rows) {
                        throw std::invalid_argument("All columns must have identical row counts");
                    }
                    views[c].data = arr.data();
                    views[c].dtype = MapDlpackDType(arr.dtype());
                    holders.push_back(std::move(arr));
                }

                if (out.size() != n_rows * static_cast<size_t>(self.num_outputs())) {
                    throw std::invalid_argument("Output buffer size mismatch");
                }
                float* out_ptr = out.data();
                fast_scorer::PredictOptions popts;
                popts.start_iteration = start_iteration;
                popts.num_iteration = num_iteration;
                popts.raw_score = raw_score;
                popts.num_threads = num_threads;
                popts.isa = ParseIsa(isa);
                popts.kernel = ParseKernel(kernel);

                nb::gil_scoped_release release;
                self.PredictColumnar(views.data(), n_rows, n_cols, out_ptr, popts);
            },
            "columns"_a,
            "out"_a,
            "start_iteration"_a = 0,
            "num_iteration"_a = -1,
            "raw_score"_a = false,
            "num_threads"_a = 0,
            "isa"_a = "auto",
            "kernel"_a = "auto"
        );
}
