#include "fast_scorer/compiler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace fast_scorer {

namespace {

inline bool EvalNaNGoesLeft(int8_t decision_type, double threshold) noexcept {
    const uint8_t missing_type = static_cast<uint8_t>((decision_type >> 2) & 3);
    const bool default_left = (decision_type & 2) != 0;
    if (missing_type == 1 || missing_type == 2) {
        return default_left;
    }
    return (0.0 <= threshold);
}

inline bool HasZeroMissingFlag(int8_t decision_type) noexcept {
    return ((decision_type >> 2) & 3) == 1;
}

inline bool EvalZeroGoesLeft(int8_t decision_type, double threshold) noexcept {
    const uint8_t missing_type = static_cast<uint8_t>((decision_type >> 2) & 3);
    const bool default_left = (decision_type & 2) != 0;
    if (missing_type == 1) {
        return default_left;
    }
    return (0.0 <= threshold);
}

inline uint64_t MakeKeepMask64(int clear_start, int clear_count) noexcept {
    if (clear_count <= 0) return ~0ULL;
    if (clear_count >= 64) return 0ULL;
    const uint64_t clear_bits = ((1ULL << clear_count) - 1ULL) << clear_start;
    return ~clear_bits;
}

struct PreparedNode {
    uint32_t feature_idx = 0;
    double threshold = 0.0;
    bool nan_goes_left = false;
    bool has_zero_missing = false;
    bool zero_goes_left = false;
    int left_clear_start = 0;
    int left_clear_count = 0;
};

struct PreparedTree {
    uint32_t global_tree_idx = 0;
    uint16_t class_id = 0;
    uint16_t num_leaves = 0;
    std::vector<PreparedNode> nodes;
    std::vector<double> dfs_leaves;
};

void BuildPreparedTreeDFS(
    const RawTree& raw,
    int raw_node,
    int& next_dfs_leaf,
    std::vector<int32_t>& orig_leaf_to_dfs,
    std::vector<PreparedNode>& out_nodes
) {
    const size_t cur_idx = out_nodes.size();
    out_nodes.emplace_back();

    const int left_start = next_dfs_leaf;
    const int32_t lc = raw.left_child[raw_node];
    if (lc >= 0) {
        BuildPreparedTreeDFS(raw, lc, next_dfs_leaf, orig_leaf_to_dfs, out_nodes);
    } else {
        const int orig_leaf = ~lc;
        orig_leaf_to_dfs[orig_leaf] = next_dfs_leaf++;
    }
    const int left_end = next_dfs_leaf;

    const int32_t rc = raw.right_child[raw_node];
    if (rc >= 0) {
        BuildPreparedTreeDFS(raw, rc, next_dfs_leaf, orig_leaf_to_dfs, out_nodes);
    } else {
        const int orig_leaf = ~rc;
        orig_leaf_to_dfs[orig_leaf] = next_dfs_leaf++;
    }

    const double t64 = raw.threshold[raw_node];
    const int8_t dt = raw.decision_type[raw_node];

    PreparedNode pn;
    pn.feature_idx = static_cast<uint32_t>(raw.split_feature[raw_node]);
    pn.threshold = t64;
    pn.nan_goes_left = EvalNaNGoesLeft(dt, t64);
    pn.has_zero_missing = HasZeroMissingFlag(dt);
    pn.zero_goes_left = EvalZeroGoesLeft(dt, t64);
    pn.left_clear_start = left_start;
    pn.left_clear_count = left_end - left_start;
    out_nodes[cur_idx] = pn;
}

CompactTreeMeta AppendCompactTree(
    const RawTree& raw,
    uint32_t global_tree_idx,
    uint16_t class_id,
    CompiledModel& out
) {
    const int num_leaves = raw.num_leaves;
    const int num_internal = num_leaves - 1;

    CompactTreeMeta meta;
    meta.node_offset = static_cast<uint32_t>(out.compact_nodes.size());
    meta.leaf_offset = static_cast<uint32_t>(out.compact_leaves.size());
    meta.num_leaves = static_cast<uint16_t>(num_leaves);
    meta.class_id = class_id;
    meta.global_tree_idx = global_tree_idx;
    meta.has_zero_missing = false;

    std::vector<int32_t> raw_to_bfs(num_internal, -1);
    std::vector<int32_t> bfs_to_raw;
    bfs_to_raw.reserve(num_internal);

    std::queue<int32_t> q;
    q.push(0);
    raw_to_bfs[0] = 0;
    bfs_to_raw.push_back(0);

    while (!q.empty()) {
        int32_t u = q.front();
        q.pop();
        int32_t lc = raw.left_child[u];
        if (lc >= 0 && raw_to_bfs[lc] < 0) {
            raw_to_bfs[lc] = static_cast<int32_t>(bfs_to_raw.size());
            bfs_to_raw.push_back(lc);
            q.push(lc);
        }
        int32_t rc = raw.right_child[u];
        if (rc >= 0 && raw_to_bfs[rc] < 0) {
            raw_to_bfs[rc] = static_cast<int32_t>(bfs_to_raw.size());
            bfs_to_raw.push_back(rc);
            q.push(rc);
        }
    }

    std::vector<int32_t> orig_leaf_to_new(num_leaves, -1);
    int32_t next_leaf = 0;

    for (int32_t bfs_idx = 0; bfs_idx < static_cast<int32_t>(bfs_to_raw.size()); ++bfs_idx) {
        const int32_t u = bfs_to_raw[bfs_idx];
        const double t64 = raw.threshold[u];
        const int8_t dt = raw.decision_type[u];

        const uint32_t feat = static_cast<uint32_t>(raw.split_feature[u]);
        const bool nan_left = EvalNaNGoesLeft(dt, t64);
        const bool has_zero = HasZeroMissingFlag(dt);
        const bool zero_left = EvalZeroGoesLeft(dt, t64);
        if (has_zero) meta.has_zero_missing = true;

        uint32_t packed = (feat & 0x00FFFFFFu);
        if (nan_left) packed |= (1u << 24);
        if (has_zero) packed |= (1u << 25);
        if (zero_left) packed |= (1u << 26);

        int32_t lc = raw.left_child[u];
        if (lc >= 0) {
            lc = raw_to_bfs[lc];
        } else {
            int32_t ol = ~lc;
            if (orig_leaf_to_new[ol] < 0) {
                orig_leaf_to_new[ol] = next_leaf++;
                out.compact_leaves.push_back(raw.leaf_value[ol]);
            }
            lc = ~orig_leaf_to_new[ol];
        }

        int32_t rc = raw.right_child[u];
        if (rc >= 0) {
            rc = raw_to_bfs[rc];
        } else {
            int32_t ol = ~rc;
            if (orig_leaf_to_new[ol] < 0) {
                orig_leaf_to_new[ol] = next_leaf++;
                out.compact_leaves.push_back(raw.leaf_value[ol]);
            }
            rc = ~orig_leaf_to_new[ol];
        }

        CompactNode cn;
        cn.threshold = t64;
        cn.feature_and_flags = packed;
        cn.left_child = lc;
        cn.right_child = rc;
        out.compact_nodes.push_back(cn);
    }

    return meta;
}

QS32Block BuildQS32Block(
    const std::vector<PreparedTree>& trees,
    size_t begin_idx,
    size_t end_idx,
    int num_features
) {
    QS32Block blk;
    blk.num_trees = static_cast<uint32_t>(end_idx - begin_idx);

    uint16_t max_leaves = 2;
    int32_t common_class = static_cast<int32_t>(trees[begin_idx].class_id);
    for (size_t i = begin_idx; i < end_idx; ++i) {
        if (trees[i].num_leaves > max_leaves) max_leaves = trees[i].num_leaves;
        if (static_cast<int32_t>(trees[i].class_id) != common_class) {
            common_class = -1;
        }
    }
    blk.single_class_id = common_class;
    blk.leaf_stride = static_cast<uint32_t>((max_leaves + 3u) & ~3u);
    if (blk.leaf_stride > 32u) blk.leaf_stride = 32u;

    blk.tree_class_id.resize(blk.num_trees);
    blk.global_tree_ids.resize(blk.num_trees);
    blk.leaf_values.assign(static_cast<size_t>(blk.num_trees) * blk.leaf_stride, 0.0);

    struct TempSplit {
        double threshold;
        uint16_t tree_in_block;
        uint32_t mask;
    };
    std::vector<std::vector<TempSplit>> per_feat_splits(num_features);
    std::vector<std::vector<QS32SpecialEntry>> per_feat_nan(num_features);
    std::vector<std::vector<QS32SpecialEntry>> per_feat_zero(num_features);
    std::vector<bool> feat_has_zero(num_features, false);

    for (uint32_t t = 0; t < blk.num_trees; ++t) {
        const PreparedTree& pt = trees[begin_idx + t];
        blk.tree_class_id[t] = pt.class_id;
        blk.global_tree_ids[t] = pt.global_tree_idx;
        for (uint16_t l = 0; l < pt.num_leaves; ++l) {
            blk.leaf_values[static_cast<size_t>(t) * blk.leaf_stride + l] = pt.dfs_leaves[l];
        }
        for (const PreparedNode& pn : pt.nodes) {
            if (pn.has_zero_missing) {
                feat_has_zero[pn.feature_idx] = true;
            }
        }
    }

    for (uint32_t t = 0; t < blk.num_trees; ++t) {
        const PreparedTree& pt = trees[begin_idx + t];
        for (const PreparedNode& pn : pt.nodes) {
            const uint32_t f = pn.feature_idx;
            const uint32_t mask32 = static_cast<uint32_t>(
                MakeKeepMask64(pn.left_clear_start, pn.left_clear_count)
            );
            per_feat_splits[f].push_back(TempSplit{
                pn.threshold,
                static_cast<uint16_t>(t),
                mask32
            });

            if (!pn.nan_goes_left) {
                per_feat_nan[f].push_back(QS32SpecialEntry{
                    static_cast<uint16_t>(t), 0, mask32
                });
            }
            if (feat_has_zero[f] && !pn.zero_goes_left) {
                per_feat_zero[f].push_back(QS32SpecialEntry{
                    static_cast<uint16_t>(t), 0, mask32
                });
            }
        }
    }

    for (int f = 0; f < num_features; ++f) {
        auto& sp = per_feat_splits[f];
        if (sp.empty()) continue;

        std::sort(sp.begin(), sp.end(), [](const TempSplit& a, const TempSplit& b) {
            if (a.threshold != b.threshold) return a.threshold < b.threshold;
            return a.tree_in_block < b.tree_in_block;
        });

        size_t write_pos = 0;
        for (size_t read_pos = 0; read_pos < sp.size(); ++read_pos) {
            if (write_pos > 0 &&
                sp[write_pos - 1].threshold == sp[read_pos].threshold &&
                sp[write_pos - 1].tree_in_block == sp[read_pos].tree_in_block) {
                sp[write_pos - 1].mask &= sp[read_pos].mask;
            } else {
                sp[write_pos++] = sp[read_pos];
            }
        }
        sp.resize(write_pos);

        QS32FeatureSpan span;
        span.feature_idx = static_cast<uint32_t>(f);
        span.has_zero_missing = feat_has_zero[f];

        span.split_begin = static_cast<uint32_t>(blk.thresholds.size());
        for (const auto& s : sp) {
            blk.thresholds.push_back(s.threshold);
            blk.tree_ids.push_back(s.tree_in_block);
            blk.masks.push_back(s.mask);
        }
        span.split_end = static_cast<uint32_t>(blk.thresholds.size());

        span.nan_begin = static_cast<uint32_t>(blk.nan_entries.size());
        blk.nan_entries.insert(blk.nan_entries.end(), per_feat_nan[f].begin(), per_feat_nan[f].end());
        span.nan_end = static_cast<uint32_t>(blk.nan_entries.size());

        span.zero_begin = static_cast<uint32_t>(blk.zero_entries.size());
        if (feat_has_zero[f]) {
            blk.zero_entries.insert(blk.zero_entries.end(), per_feat_zero[f].begin(), per_feat_zero[f].end());
        }
        span.zero_end = static_cast<uint32_t>(blk.zero_entries.size());

        blk.features.push_back(span);
    }

    return blk;
}

QS64Block BuildQS64Block(
    const std::vector<PreparedTree>& trees,
    size_t begin_idx,
    size_t end_idx,
    int num_features
) {
    QS64Block blk;
    blk.num_trees = static_cast<uint32_t>(end_idx - begin_idx);

    uint16_t max_leaves = 33;
    int32_t common_class = static_cast<int32_t>(trees[begin_idx].class_id);
    for (size_t i = begin_idx; i < end_idx; ++i) {
        if (trees[i].num_leaves > max_leaves) max_leaves = trees[i].num_leaves;
        if (static_cast<int32_t>(trees[i].class_id) != common_class) {
            common_class = -1;
        }
    }
    blk.single_class_id = common_class;
    blk.leaf_stride = static_cast<uint32_t>((max_leaves + 3u) & ~3u);
    if (blk.leaf_stride > 64u) blk.leaf_stride = 64u;

    blk.tree_class_id.resize(blk.num_trees);
    blk.global_tree_ids.resize(blk.num_trees);
    blk.leaf_values.assign(static_cast<size_t>(blk.num_trees) * blk.leaf_stride, 0.0);

    struct TempSplit64 {
        double threshold;
        uint16_t tree_in_block;
        uint8_t which_half;
        uint32_t mask_lo;
        uint32_t mask_hi;
    };
    std::vector<std::vector<TempSplit64>> per_feat_splits(num_features);
    std::vector<std::vector<QS64SpecialEntry>> per_feat_nan(num_features);
    std::vector<std::vector<QS64SpecialEntry>> per_feat_zero(num_features);
    std::vector<bool> feat_has_zero(num_features, false);

    for (uint32_t t = 0; t < blk.num_trees; ++t) {
        const PreparedTree& pt = trees[begin_idx + t];
        blk.tree_class_id[t] = pt.class_id;
        blk.global_tree_ids[t] = pt.global_tree_idx;
        for (uint16_t l = 0; l < pt.num_leaves; ++l) {
            blk.leaf_values[static_cast<size_t>(t) * blk.leaf_stride + l] = pt.dfs_leaves[l];
        }
        for (const PreparedNode& pn : pt.nodes) {
            if (pn.has_zero_missing) {
                feat_has_zero[pn.feature_idx] = true;
            }
        }
    }

    for (uint32_t t = 0; t < blk.num_trees; ++t) {
        const PreparedTree& pt = trees[begin_idx + t];
        for (const PreparedNode& pn : pt.nodes) {
            const uint32_t f = pn.feature_idx;
            const uint64_t mask64 = MakeKeepMask64(pn.left_clear_start, pn.left_clear_count);
            const uint32_t m_lo = static_cast<uint32_t>(mask64 & 0xFFFFFFFFULL);
            const uint32_t m_hi = static_cast<uint32_t>(mask64 >> 32);
            uint8_t wh = 2;
            if (m_hi == 0xFFFFFFFFu) {
                wh = 0;
            } else if (m_lo == 0xFFFFFFFFu) {
                wh = 1;
            }

            per_feat_splits[f].push_back(TempSplit64{
                pn.threshold,
                static_cast<uint16_t>(t),
                wh,
                m_lo,
                m_hi
            });

            if (!pn.nan_goes_left) {
                per_feat_nan[f].push_back(QS64SpecialEntry{
                    static_cast<uint16_t>(t), wh, m_lo, m_hi
                });
            }
            if (feat_has_zero[f] && !pn.zero_goes_left) {
                per_feat_zero[f].push_back(QS64SpecialEntry{
                    static_cast<uint16_t>(t), wh, m_lo, m_hi
                });
            }
        }
    }

    for (int f = 0; f < num_features; ++f) {
        auto& sp = per_feat_splits[f];
        if (sp.empty()) continue;

        std::sort(sp.begin(), sp.end(), [](const TempSplit64& a, const TempSplit64& b) {
            if (a.threshold != b.threshold) return a.threshold < b.threshold;
            return a.tree_in_block < b.tree_in_block;
        });

        size_t write_pos = 0;
        for (size_t read_pos = 0; read_pos < sp.size(); ++read_pos) {
            if (write_pos > 0 &&
                sp[write_pos - 1].threshold == sp[read_pos].threshold &&
                sp[write_pos - 1].tree_in_block == sp[read_pos].tree_in_block) {
                sp[write_pos - 1].mask_lo &= sp[read_pos].mask_lo;
                sp[write_pos - 1].mask_hi &= sp[read_pos].mask_hi;
                if (sp[write_pos - 1].mask_hi == 0xFFFFFFFFu) {
                    sp[write_pos - 1].which_half = 0;
                } else if (sp[write_pos - 1].mask_lo == 0xFFFFFFFFu) {
                    sp[write_pos - 1].which_half = 1;
                } else {
                    sp[write_pos - 1].which_half = 2;
                }
            } else {
                sp[write_pos++] = sp[read_pos];
            }
        }
        sp.resize(write_pos);

        QS64FeatureSpan span;
        span.feature_idx = static_cast<uint32_t>(f);
        span.has_zero_missing = feat_has_zero[f];

        span.split_begin = static_cast<uint32_t>(blk.thresholds.size());
        for (const auto& s : sp) {
            blk.thresholds.push_back(s.threshold);
            blk.tree_ids.push_back(s.tree_in_block);
            blk.which_half.push_back(s.which_half);
            blk.masks_lo.push_back(s.mask_lo);
            blk.masks_hi.push_back(s.mask_hi);
        }
        span.split_end = static_cast<uint32_t>(blk.thresholds.size());

        span.nan_begin = static_cast<uint32_t>(blk.nan_entries.size());
        blk.nan_entries.insert(blk.nan_entries.end(), per_feat_nan[f].begin(), per_feat_nan[f].end());
        span.nan_end = static_cast<uint32_t>(blk.nan_entries.size());

        span.zero_begin = static_cast<uint32_t>(blk.zero_entries.size());
        if (feat_has_zero[f]) {
            blk.zero_entries.insert(blk.zero_entries.end(), per_feat_zero[f].begin(), per_feat_zero[f].end());
        }
        span.zero_end = static_cast<uint32_t>(blk.zero_entries.size());

        blk.features.push_back(span);
    }

    return blk;
}

} // namespace

CompiledModel CompileModel(const RawModel& raw, const CompileOptions& opts) {
    CompiledModel out;
    out.num_features = raw.num_features();
    out.num_class = raw.num_class;
    out.num_tree_per_iteration = raw.num_tree_per_iteration;
    out.total_trees = static_cast<int>(raw.trees.size());
    out.total_iterations = raw.total_iterations();
    out.objective = raw.objective;
    out.sigmoid_alpha = raw.sigmoid_alpha;
    out.average_output = raw.average_output;
    out.feature_names = raw.feature_names;

    if (raw.has_categorical) {
        out.supported = false;
        out.unsupported_reason = "Categorical splits are not supported in fast_scorer";
        return out;
    }
    if (raw.has_linear) {
        out.supported = false;
        out.unsupported_reason = "Linear trees (is_linear=1) are not supported in fast_scorer";
        return out;
    }
    if (out.num_features <= 0 || out.num_features > 0x00FFFFFF) {
        out.supported = false;
        out.unsupported_reason = "Feature count exceeds 24-bit CompactNode limit";
        return out;
    }

    const int k = out.num_tree_per_iteration;
    out.full_base_bias.assign(static_cast<size_t>(k), 0.0);

    std::vector<PreparedTree> qs32_candidates;
    std::vector<PreparedTree> qs64_candidates;

    for (int t = 0; t < out.total_trees; ++t) {
        const RawTree& rt = raw.trees[t];
        const uint16_t class_id = static_cast<uint16_t>(t % k);
        if (rt.num_leaves > out.max_num_leaves) {
            out.max_num_leaves = rt.num_leaves;
        }

        if (rt.num_leaves <= 0) continue;

        if (rt.num_leaves == 1) {
            const double val = rt.leaf_value.empty() ? 0.0 : rt.leaf_value[0];
            out.full_base_bias[class_id] += val;
            out.constant_trees.push_back(ConstantTreeEntry{
                static_cast<uint32_t>(t),
                class_id,
                val
            });
            continue;
        }

        if (rt.num_leaves > 32768) {
            out.supported = false;
            out.unsupported_reason = "Tree num_leaves exceeds 32768 limit";
            return out;
        }

        CompactTreeMeta cmeta = AppendCompactTree(rt, static_cast<uint32_t>(t), class_id, out);
        out.all_compact_trees.push_back(cmeta);

        if (rt.num_leaves > 64) {
            out.deep_compact_trees.push_back(cmeta);
            continue;
        }

        PreparedTree pt;
        pt.global_tree_idx = static_cast<uint32_t>(t);
        pt.class_id = class_id;
        pt.num_leaves = static_cast<uint16_t>(rt.num_leaves);
        pt.nodes.reserve(static_cast<size_t>(rt.num_leaves - 1));

        std::vector<int32_t> orig_leaf_to_dfs(rt.num_leaves, -1);
        int next_dfs_leaf = 0;
        BuildPreparedTreeDFS(rt, 0, next_dfs_leaf, orig_leaf_to_dfs, pt.nodes);

        pt.dfs_leaves.resize(rt.num_leaves);
        for (int orig_l = 0; orig_l < rt.num_leaves; ++orig_l) {
            int dfs_l = orig_leaf_to_dfs[orig_l];
            if (dfs_l >= 0 && dfs_l < rt.num_leaves) {
                pt.dfs_leaves[dfs_l] = rt.leaf_value[orig_l];
            }
        }

        if (rt.num_leaves <= 32) {
            qs32_candidates.push_back(std::move(pt));
        } else {
            qs64_candidates.push_back(std::move(pt));
        }
    }

    auto by_class_then_idx = [](const PreparedTree& a, const PreparedTree& b) {
        if (a.class_id != b.class_id) return a.class_id < b.class_id;
        return a.global_tree_idx < b.global_tree_idx;
    };
    std::sort(qs32_candidates.begin(), qs32_candidates.end(), by_class_then_idx);
    std::sort(qs64_candidates.begin(), qs64_candidates.end(), by_class_then_idx);

    const size_t b32 = std::clamp<uint32_t>(opts.qs32_block_trees, 1u, 128u);
    for (size_t i = 0; i < qs32_candidates.size(); i += b32) {
        size_t end = std::min(qs32_candidates.size(), i + b32);
        out.qs32_blocks.push_back(BuildQS32Block(qs32_candidates, i, end, out.num_features));
    }

    const size_t b64 = std::clamp<uint32_t>(opts.qs64_block_trees, 1u, 64u);
    for (size_t i = 0; i < qs64_candidates.size(); i += b64) {
        size_t end = std::min(qs64_candidates.size(), i + b64);
        out.qs64_blocks.push_back(BuildQS64Block(qs64_candidates, i, end, out.num_features));
    }

    if (out.qs32_blocks.empty() && out.qs64_blocks.empty()) {
        out.default_kernel = KernelKind::CompactTree;
    } else {
        out.default_kernel = KernelKind::QuickScorer;
    }

    return out;
}

} // namespace fast_scorer
