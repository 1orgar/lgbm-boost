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
    ColumnDType dtype = ColumnDType::Float64;
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
// ============================================================================

struct RawTree {
    int num_leaves = 0;
    int num_cat = 0;
    bool is_linear = false;
    double shrinkage = 1.0;

    std::vector<int32_t> split_feature;
    std::vector<double> threshold;
    std::vector<int8_t> decision_type;
    std::vector<int32_t> left_child;
    std::vector<int32_t> right_child;
    std::vector<double> leaf_value;
};

struct RawModel {
    std::string version;
    int num_class = 1;
    int num_tree_per_iteration = 1;
    int max_feature_idx = 0;
    ObjectiveType objective = ObjectiveType::Regression;
    double sigmoid_alpha = 1.0;
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
// Compiled exact float64 high-performance tree structures
// ============================================================================

// Compact 32-byte branchless node (2 nodes per 64-byte cache line, power-of-2 shift << 5)
struct alignas(32) CompactNode {
    double threshold;           // Exact float64 threshold from LightGBM
    uint32_t feature_and_flags; // bits [0..23]: feature_idx, bit 24: nan_goes_left,
                                // bit 25: has_zero_missing, bit 26: zero_goes_left
    int32_t left_child;         // >= 0: node index in tree, < 0: ~leaf_idx
    int32_t right_child;        // >= 0: node index in tree, < 0: ~leaf_idx
    int32_t pad[3] = {0, 0, 0};

    inline uint32_t feature_idx() const noexcept { return feature_and_flags & 0x00FFFFFFu; }
    inline bool nan_goes_left() const noexcept { return (feature_and_flags & (1u << 24)) != 0; }
    inline bool has_zero_missing() const noexcept { return (feature_and_flags & (1u << 25)) != 0; }
    inline bool zero_goes_left() const noexcept { return (feature_and_flags & (1u << 26)) != 0; }
};
static_assert(sizeof(CompactNode) == 32, "CompactNode must be exactly 32 bytes");
static_assert(offsetof(CompactNode, threshold) == 0, "threshold offset must be 0");
static_assert(offsetof(CompactNode, feature_and_flags) == 8, "feature_and_flags offset must be 8");
static_assert(offsetof(CompactNode, left_child) == 12, "left_child offset must be 12");
static_assert(offsetof(CompactNode, right_child) == 16, "right_child offset must be 16");

struct CompactTreeMeta {
    uint32_t node_offset = 0;
    uint32_t leaf_offset = 0;
    uint16_t num_leaves = 0;
    uint16_t class_id = 0;
    uint32_t global_tree_idx = 0;
    bool has_zero_missing = false;
};

// ============================================================================
// V-QuickScorer structures (exact float64 thresholds & leaves + 32/64-bit masks)
// ============================================================================

struct QS32SpecialEntry {
    uint16_t tree_in_block;
    uint16_t pad = 0;
    uint32_t mask;
};

struct QS64SpecialEntry {
    uint16_t tree_in_block;
    uint16_t which_half;
    uint32_t mask_lo;
    uint32_t mask_hi;
};

struct QS32FeatureSpan {
    uint32_t feature_idx = 0;
    uint32_t split_begin = 0;
    uint32_t split_end = 0;
    uint32_t nan_begin = 0;
    uint32_t nan_end = 0;
    uint32_t zero_begin = 0;
    uint32_t zero_end = 0;
    bool has_zero_missing = false;
};

struct QS32Block {
    uint32_t num_trees = 0;
    uint32_t leaf_stride = 32;
    int32_t single_class_id = -1;

    std::vector<QS32FeatureSpan> features;

    // SoA sorted ascending by exact double threshold within each feature span
    std::vector<double> thresholds;
    std::vector<uint16_t> tree_ids;
    std::vector<uint32_t> masks;

    std::vector<QS32SpecialEntry> nan_entries;
    std::vector<QS32SpecialEntry> zero_entries;

    std::vector<uint16_t> tree_class_id;
    std::vector<uint32_t> global_tree_ids;

    // Flattened exact float64 leaf values in DFS order: size = num_trees * leaf_stride
    std::vector<double> leaf_values;
};

struct QS64FeatureSpan {
    uint32_t feature_idx = 0;
    uint32_t split_begin = 0;
    uint32_t split_end = 0;
    uint32_t nan_begin = 0;
    uint32_t nan_end = 0;
    uint32_t zero_begin = 0;
    uint32_t zero_end = 0;
    bool has_zero_missing = false;
};

struct QS64Block {
    uint32_t num_trees = 0;
    uint32_t leaf_stride = 64;
    int32_t single_class_id = -1;

    std::vector<QS64FeatureSpan> features;

    std::vector<double> thresholds;
    std::vector<uint16_t> tree_ids;
    std::vector<uint8_t> which_half;
    std::vector<uint32_t> masks_lo;
    std::vector<uint32_t> masks_hi;

    std::vector<QS64SpecialEntry> nan_entries;
    std::vector<QS64SpecialEntry> zero_entries;

    std::vector<uint16_t> tree_class_id;
    std::vector<uint32_t> global_tree_ids;
    std::vector<double> leaf_values;
};

struct ConstantTreeEntry {
    uint32_t global_tree_idx = 0;
    uint16_t class_id = 0;
    double value = 0.0;
};

struct CompiledModel {
    int num_features = 0;
    int num_class = 1;
    int num_tree_per_iteration = 1;
    int total_trees = 0;
    int total_iterations = 0;
    int max_num_leaves = 0;

    ObjectiveType objective = ObjectiveType::Regression;
    double sigmoid_alpha = 1.0;
    bool average_output = false;
    bool supported = true;
    std::string unsupported_reason;

    std::vector<std::string> feature_names;

    std::vector<double> full_base_bias;
    std::vector<ConstantTreeEntry> constant_trees;

    std::vector<QS32Block> qs32_blocks;
    std::vector<QS64Block> qs64_blocks;

    std::vector<CompactNode> compact_nodes;
    std::vector<double> compact_leaves;
    std::vector<CompactTreeMeta> deep_compact_trees;
    std::vector<CompactTreeMeta> all_compact_trees;

    KernelKind default_kernel = KernelKind::QuickScorer;
};

} // namespace fast_scorer
