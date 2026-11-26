#pragma once

#include <cstddef>
#include <cstdint>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace fast_scorer {

enum class ObjectiveType : uint8_t {
    Regression = 0,        // identity (regression, l1, huber, fair, quantile, ranking, custom)
    BinaryLogloss = 1,     // 1 / (1 + exp(-sigmoid_alpha * x)) (binary, cross_entropy, multiclassova)
    MulticlassSoftmax = 2, // softmax over num_tree_per_iteration classes
    Exponential = 3        // exp(x) (poisson, gamma, tweedie)
};

enum class SimdIsa : uint8_t {
    Auto = 0,
    Scalar = 1,
    Avx2 = 2,
    Avx512 = 3
};

enum class KernelKind : uint8_t {
    Auto = 0,
    QuickScorer = 1,
    CompactTree = 2
};

enum class ColumnDType : uint8_t {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    Int16 = 4,
    Int8 = 5,
    UInt8 = 6,
    UInt16 = 7,
    UInt32 = 8
};

struct ColumnView {
    const void* data = nullptr;
    ColumnDType dtype = ColumnDType::Float32;
};

struct PredictOptions {
    int start_iteration = 0;
    int num_iteration = -1; // <= 0 means all remaining iterations
    bool raw_score = false;
    int num_threads = 0;    // <= 0 means auto (all available hardware threads)
    SimdIsa isa = SimdIsa::Auto;
    KernelKind kernel = KernelKind::Auto;
};

// ============================================================================
// Raw parsed / imported LightGBM tree & ensemble representation
// (Used both by standalone text parser and direct in-memory import in LightGBM fork)
// ============================================================================

struct RawTree {
    int num_leaves = 0;
    int num_cat = 0;
    bool is_linear = false;
    double shrinkage = 1.0;

    // Size = num_leaves - 1 (for non-categorical numerical trees with num_leaves > 1)
    std::vector<int32_t> split_feature;
    std::vector<double> threshold;
    std::vector<int8_t> decision_type;
    std::vector<int32_t> left_child;
    std::vector<int32_t> right_child;

    // Size = num_leaves
    std::vector<double> leaf_value;
};

struct RawModel {
    std::string version;
    int num_class = 1;
    int num_tree_per_iteration = 1;
    int max_feature_idx = 0;
    ObjectiveType objective = ObjectiveType::Regression;
    float sigmoid_alpha = 1.0f;
    bool average_output = false;
    bool has_categorical = false;
    bool has_linear = false;

    std::vector<std::string> feature_names;
    std::vector<RawTree> trees;

    int num_features() const { return max_feature_idx + 1; }
    int total_iterations() const {
        if (num_tree_per_iteration <= 0) return 0;
        return static_cast<int>(trees.size()) / num_tree_per_iteration;
    }
};

// ============================================================================
// Compiled high-performance tree structures
// ============================================================================

// Compact 16-byte branchless node (4 nodes per 64-byte cache line)
struct alignas(16) CompactNode {
    float threshold;            // Exact downward-rounded float32 threshold
    uint32_t feature_and_flags; // bits [0..23]: feature_idx, bit 24: nan_goes_left,
                                // bit 25: has_zero_missing, bit 26: zero_goes_left
    int32_t left_child;         // >= 0: node index in tree, < 0: ~leaf_idx
    int32_t right_child;        // >= 0: node index in tree, < 0: ~leaf_idx

    inline uint32_t feature_idx() const noexcept { return feature_and_flags & 0x00FFFFFFu; }
    inline bool nan_goes_left() const noexcept { return (feature_and_flags & (1u << 24)) != 0; }
    inline bool has_zero_missing() const noexcept { return (feature_and_flags & (1u << 25)) != 0; }
    inline bool zero_goes_left() const noexcept { return (feature_and_flags & (1u << 26)) != 0; }
};
static_assert(sizeof(CompactNode) == 16, "CompactNode must be exactly 16 bytes");

struct CompactTreeMeta {
    uint32_t node_offset = 0; // Offset into CompiledModel::compact_nodes
    uint32_t leaf_offset = 0; // Offset into CompiledModel::compact_leaves
    uint16_t num_leaves = 0;
    uint16_t class_id = 0;
    uint32_t global_tree_idx = 0;
    bool has_zero_missing = false;
};

// ============================================================================
// V-QuickScorer structures (grouped into L1d-resident TreeBlocks)
// ============================================================================

// Special-value mask update entry (for NaN or 0.0 when special value goes RIGHT)
struct QS32SpecialEntry {
    uint16_t tree_in_block;
    uint16_t pad = 0;
    uint32_t mask;
};

struct QS64SpecialEntry {
    uint16_t tree_in_block;
    uint16_t which_half; // 0 = lo32 only, 1 = hi32 only, 2 = both lo32 & hi32
    uint32_t mask_lo;
    uint32_t mask_hi;
};

// Per-feature range inside a QS32Block (trees with <= 32 leaves, 32-bit bitmasks)
struct QS32FeatureSpan {
    uint32_t feature_idx = 0;
    uint32_t split_begin = 0;
    uint32_t split_end = 0;
    uint32_t bin_offset_start = 0;
    uint32_t nan_begin = 0;
    uint32_t nan_end = 0;
    uint32_t zero_begin = 0;
    uint32_t zero_end = 0;
    bool has_zero_missing = false;
};

struct QS32Block {
    uint32_t num_trees = 0;          // <= 128 trees per block (fits in L1d)
    uint32_t leaf_stride = 32;       // Padded stride per tree in leaf_values (cache-friendly)
    int32_t single_class_id = -1;    // >= 0 if all trees in block contribute to the same class

    // Active features in this block (only features actually split on in this block)
    std::vector<QS32FeatureSpan> features;

    // Jump table mapping global feature bin b -> split index in [split_begin, split_end]
    std::vector<uint16_t> bin_offsets;

    // Structure-of-Arrays (SoA) sorted ascending by threshold within each feature span
    std::vector<float> thresholds;
    std::vector<uint16_t> tree_ids;
    std::vector<uint32_t> masks;

    // Static routing masks for NaN and Zero
    std::vector<QS32SpecialEntry> nan_entries;
    std::vector<QS32SpecialEntry> zero_entries;

    // Per-tree metadata in block (size = num_trees)
    std::vector<uint16_t> tree_class_id;
    std::vector<uint32_t> global_tree_ids;

    // Flattened leaf values in DFS order: size = num_trees * leaf_stride
    std::vector<float> leaf_values;
};

// Per-feature range inside a QS64Block (trees with 33..64 leaves, split lo32/hi32 bitmasks)
struct QS64FeatureSpan {
    uint32_t feature_idx = 0;
    uint32_t split_begin = 0;
    uint32_t split_end = 0;
    uint32_t bin_offset_start = 0;
    uint32_t nan_begin = 0;
    uint32_t nan_end = 0;
    uint32_t zero_begin = 0;
    uint32_t zero_end = 0;
    bool has_zero_missing = false;
};

struct QS64Block {
    uint32_t num_trees = 0;          // <= 64 trees per block (fits in L1d)
    uint32_t leaf_stride = 64;
    int32_t single_class_id = -1;    // >= 0 if all trees in block contribute to the same class

    std::vector<QS64FeatureSpan> features;
    std::vector<uint16_t> bin_offsets;

    // SoA sorted ascending by threshold within each feature span
    std::vector<float> thresholds;
    std::vector<uint16_t> tree_ids;
    std::vector<uint8_t> which_half; // 0 = lo32 only, 1 = hi32 only, 2 = both
    std::vector<uint32_t> masks_lo;
    std::vector<uint32_t> masks_hi;

    std::vector<QS64SpecialEntry> nan_entries;
    std::vector<QS64SpecialEntry> zero_entries;

    std::vector<uint16_t> tree_class_id;
    std::vector<uint32_t> global_tree_ids;
    std::vector<float> leaf_values; // size = num_trees * leaf_stride
};

struct ConstantTreeEntry {
    uint32_t global_tree_idx = 0;
    uint16_t class_id = 0;
    float value = 0.0f;
};

struct CompiledModel {
    int num_features = 0;
    int num_class = 1;
    int num_tree_per_iteration = 1;
    int total_trees = 0;
    int total_iterations = 0;
    int max_num_leaves = 0;

    ObjectiveType objective = ObjectiveType::Regression;
    float sigmoid_alpha = 1.0f;
    bool average_output = false;
    bool supported = true;
    std::string unsupported_reason;

    std::vector<std::string> feature_names;

    // Folded constant bias from all 1-leaf trees across the full ensemble (size = num_tree_per_iteration)
    std::vector<double> full_base_bias;
    // Individual 1-leaf trees (used only when partial start_iteration / num_iteration is requested)
    std::vector<ConstantTreeEntry> constant_trees;

    // Ensemble-wide feature pre-binning tables (amortizes binary search across all tree blocks)
    bool use_prebinning = false;
    std::vector<uint32_t> feature_bin_starts;     // size = num_features + 1
    std::vector<float> shared_bin_thresholds;     // sorted deduplicated thresholds per feature
    std::vector<uint8_t> feature_has_zero_missing; // size = num_features

    // Kernel 1A: V-QuickScorer 32-bit blocks (trees with 2..32 leaves)
    std::vector<QS32Block> qs32_blocks;
    // Kernel 1B: V-QuickScorer 64-bit blocks (trees with 33..64 leaves)
    std::vector<QS64Block> qs64_blocks;

    // Kernel 2: Compact Branchless Trees:
    // - deep_compact_trees: trees with > 64 leaves executed in hybrid/full mode
    // - all_compact_trees: ALL non-constant trees in iteration order (used for partial slice queries
    //   and when KernelKind::CompactTree is explicitly selected)
    std::vector<CompactNode> compact_nodes;
    std::vector<float> compact_leaves;
    std::vector<CompactTreeMeta> deep_compact_trees;
    std::vector<CompactTreeMeta> all_compact_trees;

    // Default auto-selected primary kernel for full-model batch scoring
    KernelKind default_kernel = KernelKind::QuickScorer;
};

} // namespace fast_scorer
