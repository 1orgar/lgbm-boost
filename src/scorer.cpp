#include "fast_scorer/scorer.h"
#include "fast_scorer/kernels.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(FAST_SCORER_HAVE_OPENMP)
#include <omp.h>
#endif

#if defined(FAST_SCORER_ARCH_X86_64)
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace fast_scorer {

namespace {

// 64-byte aligned buffer for SIMD loads/stores without UB or unaligned traps
class AlignedFloatBuffer {
public:
    AlignedFloatBuffer() = default;
    ~AlignedFloatBuffer() { Release(); }

    AlignedFloatBuffer(const AlignedFloatBuffer&) = delete;
    AlignedFloatBuffer& operator=(const AlignedFloatBuffer&) = delete;

    AlignedFloatBuffer(AlignedFloatBuffer&& other) noexcept
        : ptr_(other.ptr_), capacity_(other.capacity_) {
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }

    AlignedFloatBuffer& operator=(AlignedFloatBuffer&& other) noexcept {
        if (this != &other) {
            Release();
            ptr_ = other.ptr_;
            capacity_ = other.capacity_;
            other.ptr_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    void EnsureCapacity(size_t num_floats) {
        if (num_floats <= capacity_) return;
        Release();
        const size_t bytes = ((num_floats * sizeof(float) + 63u) & ~size_t(63u));
#if defined(_MSC_VER)
        void* p = _aligned_malloc(bytes, 64);
        if (!p) throw std::bad_alloc();
#else
        void* p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0 || p == nullptr) {
            throw std::bad_alloc();
        }
#endif
        ptr_ = static_cast<float*>(p);
        capacity_ = bytes / sizeof(float);
    }

    float* data() noexcept { return ptr_; }
    const float* data() const noexcept { return ptr_; }

private:
    void Release() noexcept {
        if (ptr_) {
#if defined(_MSC_VER)
            _aligned_free(ptr_);
#else
            std::free(ptr_);
#endif
            ptr_ = nullptr;
            capacity_ = 0;
        }
    }

    float* ptr_ = nullptr;
    size_t capacity_ = 0;
};

struct ThreadScratch {
    AlignedFloatBuffer tile_col_major;
    AlignedFloatBuffer tile_bins_storage;
    AlignedFloatBuffer tile_acc;
    AlignedFloatBuffer tile_out_tmp; // Used only when caller requests double* output
};

#if defined(FAST_SCORER_ARCH_X86_64)
inline uint64_t ReadXcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax = 0, edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

struct CpuFeatures {
    bool avx2 = false;
    bool avx512 = false;

    CpuFeatures() noexcept {
        uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7) return;
        __cpuid(info, 1);
        eax = info[0]; ebx = info[1]; ecx = info[2]; edx = info[3];
#else
        unsigned int max_leaf = __get_cpuid_max(0, nullptr);
        if (max_leaf < 7) return;
        __cpuid(1, eax, ebx, ecx, edx);
#endif
        const bool osxsave = (ecx & (1u << 27)) != 0;
        const bool avx = (ecx & (1u << 28)) != 0;
        const bool fma = (ecx & (1u << 12)) != 0;
        if (!osxsave || !avx || !fma) return;

        const uint64_t xcr0 = ReadXcr0();
        const bool ymm_ok = (xcr0 & 0x6ULL) == 0x6ULL;
        if (!ymm_ok) return;

#if defined(_MSC_VER)
        __cpuidex(info, 7, 0);
        eax = info[0]; ebx = info[1]; ecx = info[2]; edx = info[3];
#else
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif
        const bool has_avx2 = (ebx & (1u << 5)) != 0;
        const bool has_bmi1 = (ebx & (1u << 3)) != 0;
        avx2 = has_avx2 && has_bmi1;

        // Note: on macOS Darwin x86_64, the kernel sets ZMM state on demand or bits [7:5] in XCR0;
        // on Linux/Windows XCR0 bits [7:5] (0xE0) + [2:1] (0x06) = 0xE6 indicate ZMM state enabled.
        const bool zmm_ok = (xcr0 & 0xE6ULL) == 0xE6ULL;
        const bool has_512f = (ebx & (1u << 16)) != 0;
        const bool has_512dq = (ebx & (1u << 17)) != 0;
        const bool has_512cd = (ebx & (1u << 28)) != 0;
        const bool has_512bw = (ebx & (1u << 30)) != 0;
        const bool has_512vl = (ebx & (1u << 31)) != 0;
        avx512 = avx2 && zmm_ok && has_512f && has_512dq && has_512cd && has_512bw && has_512vl;
    }
};

const CpuFeatures& GetCpuFeatures() noexcept {
    static const CpuFeatures kFeat;
    return kFeat;
}
#endif

// Persistent lightweight C++17 worker pool used when OpenMP is disabled
#if !defined(FAST_SCORER_HAVE_OPENMP)
class PersistentWorkerPool {
public:
    static PersistentWorkerPool& Instance() {
        static PersistentWorkerPool pool;
        return pool;
    }

    void Run(int num_threads, const std::function<void(int)>& fn) {
        if (num_threads <= 1) {
            fn(0);
            return;
        }
        std::unique_lock<std::mutex> lock(dispatch_mtx_);
        EnsureThreads(static_cast<size_t>(num_threads - 1));

        {
            std::lock_guard<std::mutex> wlock(work_mtx_);
            current_fn_ = &fn;
            active_workers_ = static_cast<size_t>(num_threads - 1);
            remaining_.store(active_workers_, std::memory_order_release);
            ++generation_;
        }
        cv_start_.notify_all();

        // Master thread participates as worker 0
        fn(0);

        // Spin briefly then wait for workers
        while (remaining_.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

    ~PersistentWorkerPool() {
        {
            std::lock_guard<std::mutex> wlock(work_mtx_);
            stop_ = true;
            ++generation_;
        }
        cv_start_.notify_all();
        for (auto& th : threads_) {
            if (th.joinable()) th.join();
        }
    }

private:
    PersistentWorkerPool() = default;

    void EnsureThreads(size_t needed) {
        while (threads_.size() < needed) {
            const size_t worker_idx = threads_.size() + 1;
            threads_.emplace_back([this, worker_idx]() { WorkerLoop(worker_idx); });
        }
    }

    void WorkerLoop(size_t worker_idx) {
        uint64_t seen_gen = 0;
        while (true) {
            const std::function<void(int)>* fn = nullptr;
            bool participate = false;
            {
                std::unique_lock<std::mutex> wlock(work_mtx_);
                cv_start_.wait(wlock, [&]() { return stop_ || generation_ != seen_gen; });
                if (stop_) return;
                seen_gen = generation_;
                if (worker_idx <= active_workers_) {
                    fn = current_fn_;
                    participate = true;
                }
            }
            if (participate && fn != nullptr) {
                (*fn)(static_cast<int>(worker_idx));
                remaining_.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    std::mutex dispatch_mtx_;
    std::mutex work_mtx_;
    std::condition_variable cv_start_;
    std::vector<std::thread> threads_;
    const std::function<void(int)>* current_fn_ = nullptr;
    size_t active_workers_ = 0;
    uint64_t generation_ = 0;
    std::atomic<size_t> remaining_{0};
    bool stop_ = false;
};
#endif

const KernelTable& SelectKernelTable(SimdIsa requested) {
    if (requested == SimdIsa::Auto) {
        requested = DetectBestSimdIsa();
    }
    switch (requested) {
        case SimdIsa::Scalar:
            return GetScalarKernels();
#if defined(FAST_SCORER_ARCH_X86_64)
        case SimdIsa::Avx2:
            if (!IsSimdIsaAvailable(SimdIsa::Avx2)) {
                throw std::runtime_error("Requested AVX2 ISA is not supported by current CPU");
            }
            return GetAvx2Kernels();
        case SimdIsa::Avx512:
            if (!IsSimdIsaAvailable(SimdIsa::Avx512)) {
                throw std::runtime_error("Requested AVX-512 ISA is not supported by current CPU");
            }
            return GetAvx512Kernels();
#else
        case SimdIsa::Avx2:
        case SimdIsa::Avx512:
            throw std::runtime_error("x86_64 AVX2/AVX-512 kernels are not compiled into this non-x86 binary");
#endif
        default:
            return GetScalarKernels();
    }
}

int ResolveThreadCount(int requested) noexcept {
    if (requested > 0) return requested;
#if defined(FAST_SCORER_HAVE_OPENMP)
    int omp_max = omp_get_max_threads();
    return omp_max > 0 ? omp_max : 1;
#else
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
#endif
}

} // namespace

SimdIsa DetectBestSimdIsa() noexcept {
#if defined(FAST_SCORER_ARCH_X86_64)
    const CpuFeatures& f = GetCpuFeatures();
    if (f.avx512) return SimdIsa::Avx512;
    if (f.avx2) return SimdIsa::Avx2;
#endif
    return SimdIsa::Scalar;
}

bool IsSimdIsaAvailable(SimdIsa isa) noexcept {
    if (isa == SimdIsa::Auto || isa == SimdIsa::Scalar) return true;
#if defined(FAST_SCORER_ARCH_X86_64)
    const CpuFeatures& f = GetCpuFeatures();
    if (isa == SimdIsa::Avx2) return f.avx2;
    if (isa == SimdIsa::Avx512) return f.avx512;
#endif
    return false;
}

const char* SimdIsaName(SimdIsa isa) noexcept {
    if (isa == SimdIsa::Auto) isa = DetectBestSimdIsa();
    switch (isa) {
        case SimdIsa::Avx512: return "avx512";
        case SimdIsa::Avx2: return "avx2";
        case SimdIsa::Scalar:
        default: return "scalar";
    }
}

FastEnsembleScorer::FastEnsembleScorer(CompiledModel compiled)
    : model_(std::move(compiled)) {}

FastEnsembleScorer FastEnsembleScorer::FromModelString(
    std::string_view content,
    const CompileOptions& opts
) {
    RawModel raw = ParseModelFromString(content);
    return FastEnsembleScorer(CompileModel(raw, opts));
}

FastEnsembleScorer FastEnsembleScorer::FromModelFile(
    const std::string& filepath,
    const CompileOptions& opts
) {
    RawModel raw = ParseModelFromFile(filepath);
    return FastEnsembleScorer(CompileModel(raw, opts));
}

FastEnsembleScorer FastEnsembleScorer::FromRawModel(
    const RawModel& raw,
    const CompileOptions& opts
) {
    return FastEnsembleScorer(CompileModel(raw, opts));
}

void FastEnsembleScorer::PredictRowMajorF32(
    const float* data,
    size_t n_rows,
    size_t n_cols,
    ptrdiff_t row_stride_elems,
    float* out_scores,
    const PredictOptions& opts
) const {
    InputDescriptor desc;
    desc.kind = InputKind::RowMajorF32;
    desc.f32_ptr = data;
    desc.n_rows = n_rows;
    desc.n_cols = n_cols;
    desc.row_stride = row_stride_elems;
    ExecuteBatch(desc, out_scores, nullptr, opts);
}

void FastEnsembleScorer::PredictRowMajorF64(
    const double* data,
    size_t n_rows,
    size_t n_cols,
    ptrdiff_t row_stride_elems,
    float* out_scores,
    const PredictOptions& opts
) const {
    InputDescriptor desc;
    desc.kind = InputKind::RowMajorF64;
    desc.f64_ptr = data;
    desc.n_rows = n_rows;
    desc.n_cols = n_cols;
    desc.row_stride = row_stride_elems;
    ExecuteBatch(desc, out_scores, nullptr, opts);
}

void FastEnsembleScorer::PredictColumnar(
    const ColumnView* columns,
    size_t n_rows,
    size_t n_cols,
    float* out_scores,
    const PredictOptions& opts
) const {
    InputDescriptor desc;
    desc.kind = InputKind::Columnar;
    desc.cols_ptr = columns;
    desc.n_rows = n_rows;
    desc.n_cols = n_cols;
    ExecuteBatch(desc, out_scores, nullptr, opts);
}

void FastEnsembleScorer::PredictRowMajorToF64(
    const double* data_f64,
    const float* data_f32,
    size_t n_rows,
    size_t n_cols,
    ptrdiff_t row_stride_elems,
    double* out_scores_f64,
    const PredictOptions& opts
) const {
    InputDescriptor desc;
    if (data_f32 != nullptr) {
        desc.kind = InputKind::RowMajorF32;
        desc.f32_ptr = data_f32;
    } else {
        desc.kind = InputKind::RowMajorF64;
        desc.f64_ptr = data_f64;
    }
    desc.n_rows = n_rows;
    desc.n_cols = n_cols;
    desc.row_stride = row_stride_elems;
    ExecuteBatch(desc, nullptr, out_scores_f64, opts);
}

void FastEnsembleScorer::ExecuteBatch(
    const InputDescriptor& input,
    float* out_f32,
    double* out_f64,
    const PredictOptions& opts
) const {
    if (!model_.supported) {
        throw std::runtime_error("Model is not supported by fast_scorer: " + model_.unsupported_reason);
    }
    if (input.n_rows == 0) return;
    if (static_cast<int>(input.n_cols) != model_.num_features) {
        throw std::invalid_argument(
            "Feature count mismatch: got " + std::to_string(input.n_cols) +
            ", expected " + std::to_string(model_.num_features)
        );
    }

    const KernelTable& kernels = SelectKernelTable(opts.isa);
    const int K = model_.num_tree_per_iteration;
    const size_t n_features = static_cast<size_t>(model_.num_features);

    // Resolve iteration slice [start_iter, end_iter)
    const int start_iter = std::max(0, opts.start_iteration);
    int end_iter = model_.total_iterations;
    if (start_iter >= model_.total_iterations) {
        end_iter = start_iter;
    } else if (opts.num_iteration > 0) {
        end_iter = std::min(model_.total_iterations, start_iter + opts.num_iteration);
    }

    const bool is_full_range = (start_iter == 0 && end_iter == model_.total_iterations);
    const int active_iterations = std::max(0, end_iter - start_iter);
    const float scale_factor = (!opts.raw_score && model_.average_output && active_iterations > 0)
                                   ? (1.0f / static_cast<float>(active_iterations))
                                   : 1.0f;

    const KernelKind chosen_kernel = (opts.kernel == KernelKind::Auto)
                                         ? model_.default_kernel
                                         : opts.kernel;

    // Precompute base bias & compact slice range if partial iteration slice is requested
    std::vector<float> init_bias(static_cast<size_t>(K), 0.0f);
    const CompactTreeMeta* compact_slice_ptr = nullptr;
    size_t compact_slice_len = 0;

    if (is_full_range) {
        for (int c = 0; c < K; ++c) {
            init_bias[c] = static_cast<float>(model_.full_base_bias[c]);
        }
        if (chosen_kernel == KernelKind::CompactTree) {
            compact_slice_ptr = model_.all_compact_trees.data();
            compact_slice_len = model_.all_compact_trees.size();
        } else {
            compact_slice_ptr = model_.deep_compact_trees.data();
            compact_slice_len = model_.deep_compact_trees.size();
        }
    } else {
        const uint32_t tree_begin = static_cast<uint32_t>(start_iter * K);
        const uint32_t tree_end = static_cast<uint32_t>(end_iter * K);

        std::vector<double> bias_d(static_cast<size_t>(K), 0.0);
        for (const ConstantTreeEntry& ct : model_.constant_trees) {
            if (ct.global_tree_idx >= tree_begin && ct.global_tree_idx < tree_end) {
                bias_d[ct.class_id] += static_cast<double>(ct.value);
            }
        }
        for (int c = 0; c < K; ++c) {
            init_bias[c] = static_cast<float>(bias_d[c]);
        }

        auto it_begin = std::lower_bound(
            model_.all_compact_trees.begin(),
            model_.all_compact_trees.end(),
            tree_begin,
            [](const CompactTreeMeta& m, uint32_t idx) { return m.global_tree_idx < idx; }
        );
        auto it_end = std::lower_bound(
            it_begin,
            model_.all_compact_trees.end(),
            tree_end,
            [](const CompactTreeMeta& m, uint32_t idx) { return m.global_tree_idx < idx; }
        );
        compact_slice_len = static_cast<size_t>(it_end - it_begin);
        compact_slice_ptr = compact_slice_len > 0 ? &(*it_begin) : nullptr;
    }

    const bool run_qs = is_full_range && (chosen_kernel == KernelKind::QuickScorer);

    auto process_tile = [&](size_t tile_idx, ThreadScratch& scratch) {
        const size_t row_start = tile_idx * kTileStride;
        const size_t tile_rows = std::min(kTileStride, input.n_rows - row_start);
        const size_t padded_rows = (tile_rows + 15u) & ~size_t(15u);

        scratch.tile_col_major.EnsureCapacity(n_features * kTileStride);
        scratch.tile_acc.EnsureCapacity(static_cast<size_t>(K) * kTileStride);

        float* col_buf = scratch.tile_col_major.data();
        float* acc_buf = scratch.tile_acc.data();

        // 1. Stream and transpose/cast tile into L2-resident 64B-aligned column-major buffer
        switch (input.kind) {
            case InputKind::RowMajorF32:
                PackTileRowMajorF32_Scalar(
                    input.f32_ptr + static_cast<ptrdiff_t>(row_start) * input.row_stride,
                    tile_rows,
                    n_features,
                    input.row_stride,
                    col_buf
                );
                break;
            case InputKind::RowMajorF64:
                PackTileRowMajorF64_Scalar(
                    input.f64_ptr + static_cast<ptrdiff_t>(row_start) * input.row_stride,
                    tile_rows,
                    n_features,
                    input.row_stride,
                    col_buf
                );
                break;
            case InputKind::Columnar:
                PackTileColumnar_Scalar(
                    input.cols_ptr,
                    row_start,
                    tile_rows,
                    n_features,
                    col_buf
                );
                break;
        }

        // 2. Initialize class accumulators with constant-tree base bias
        for (int c = 0; c < K; ++c) {
            float* c_acc = acc_buf + static_cast<size_t>(c) * kTileStride;
            const float b = init_bias[c];
            std::fill_n(c_acc, padded_rows, b);
        }

        // 3. Evaluate L1d-sized tree blocks across the tile
        if (run_qs) {
            for (const QS32Block& blk : model_.qs32_blocks) {
                kernels.eval_qs32_block(blk, col_buf, nullptr, tile_rows, acc_buf);
            }
            for (const QS64Block& blk : model_.qs64_blocks) {
                kernels.eval_qs64_block(blk, col_buf, nullptr, tile_rows, acc_buf);
            }
        }

        if (compact_slice_len > 0) {
            kernels.eval_compact_trees(
                compact_slice_ptr,
                compact_slice_len,
                model_.compact_nodes.data(),
                model_.compact_leaves.data(),
                col_buf,
                padded_rows,
                acc_buf
            );
        }

        // 4. Fused vectorized link function (Sigmoid / Softmax / Exp) + store
        if (out_f32 != nullptr) {
            float* dst = out_f32 + row_start * static_cast<size_t>(K);
            kernels.finalize_tile_output(
                acc_buf,
                tile_rows,
                K,
                model_.objective,
                model_.sigmoid_alpha,
                scale_factor,
                opts.raw_score,
                dst
            );
        } else if (out_f64 != nullptr) {
            scratch.tile_out_tmp.EnsureCapacity(kTileStride * static_cast<size_t>(K));
            float* tmp = scratch.tile_out_tmp.data();
            kernels.finalize_tile_output(
                acc_buf,
                tile_rows,
                K,
                model_.objective,
                model_.sigmoid_alpha,
                scale_factor,
                opts.raw_score,
                tmp
            );
            double* dst64 = out_f64 + row_start * static_cast<size_t>(K);
            const size_t total_elems = tile_rows * static_cast<size_t>(K);
            for (size_t i = 0; i < total_elems; ++i) {
                dst64[i] = static_cast<double>(tmp[i]);
            }
        }
    };

    const size_t num_tiles = (input.n_rows + kTileStride - 1u) / kTileStride;
    const int max_threads = ResolveThreadCount(opts.num_threads);
    const int effective_threads = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(max_threads), num_tiles)
    );

    if (effective_threads <= 1) {
        ThreadScratch scratch;
        for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            process_tile(tile_idx, scratch);
        }
        return;
    }

    std::atomic<size_t> next_tile{0};

#if defined(FAST_SCORER_HAVE_OPENMP)
#pragma omp parallel num_threads(effective_threads)
    {
        ThreadScratch scratch;
        while (true) {
            const size_t tile_idx = next_tile.fetch_add(1, std::memory_order_relaxed);
            if (tile_idx >= num_tiles) break;
            process_tile(tile_idx, scratch);
        }
    }
#else
    PersistentWorkerPool::Instance().Run(effective_threads, [&](int) {
        ThreadScratch scratch;
        while (true) {
            const size_t tile_idx = next_tile.fetch_add(1, std::memory_order_relaxed);
            if (tile_idx >= num_tiles) break;
            process_tile(tile_idx, scratch);
        }
    });
#endif
}

} // namespace fast_scorer
