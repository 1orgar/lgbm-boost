#include "fast_scorer/parser.h"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fast_scorer {

namespace {

inline std::string_view TrimRightCR(std::string_view sv) {
    while (!sv.empty() && (sv.back() == '\r' || sv.back() == ' ' || sv.back() == '\t')) {
        sv.remove_suffix(1);
    }
    return sv;
}

template <typename IntT>
void ParseIntArray(std::string_view sv, std::vector<IntT>& out, size_t expected_hint = 0) {
    out.clear();
    if (expected_hint > 0) out.reserve(expected_hint);
    const char* p = sv.data();
    const char* end = sv.data() + sv.size();
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;
        if (p >= end) break;
        IntT val = 0;
        auto [next_p, ec] = std::from_chars(p, end, val);
        if (ec != std::errc{}) {
            throw std::runtime_error("Failed to parse integer token in LightGBM model");
        }
        out.push_back(val);
        p = next_p;
    }
}

void ParseDoubleArray(std::string_view sv, std::vector<double>& out, size_t expected_hint = 0) {
    out.clear();
    if (expected_hint > 0) out.reserve(expected_hint);
    const char* p = sv.data();
    const char* end = sv.data() + sv.size();
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;
        if (p >= end) break;
        // Fast temporary null-terminated stack copy per token to avoid strtod scanning whole buffer
        const char* tok_start = p;
        while (p < end && *p != ' ' && *p != '\t') ++p;
        size_t len = static_cast<size_t>(p - tok_start);
        char buf[64];
        double val = 0.0;
        if (len < sizeof(buf)) {
            std::memcpy(buf, tok_start, len);
            buf[len] = '\0';
            char* endptr = nullptr;
            val = std::strtod(buf, &endptr);
        } else {
            std::string tmp(tok_start, len);
            val = std::stod(tmp);
        }
        out.push_back(val);
    }
}

void ParseStringTokens(std::string_view sv, std::vector<std::string>& out) {
    out.clear();
    const char* p = sv.data();
    const char* end = sv.data() + sv.size();
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;
        if (p >= end) break;
        const char* start = p;
        while (p < end && *p != ' ' && *p != '\t') ++p;
        out.emplace_back(start, static_cast<size_t>(p - start));
    }
}

void ParseObjectiveConfig(std::string_view val, RawModel& model) {
    std::vector<std::string> tokens;
    ParseStringTokens(val, tokens);
    if (tokens.empty()) return;

    const std::string& name = tokens[0];
    if (name == "binary" || name == "cross_entropy" || name == "xentropy" ||
        name == "cross_entropy_lambda" || name == "xentlambda" || name == "multiclassova" ||
        name == "multiclass_ova" || name == "ova" || name == "ovr") {
        model.objective = ObjectiveType::BinaryLogloss;
        model.sigmoid_alpha = 1.0f;
    } else if (name == "multiclass" || name == "softmax") {
        model.objective = ObjectiveType::MulticlassSoftmax;
    } else if (name == "poisson" || name == "gamma" || name == "tweedie") {
        model.objective = ObjectiveType::Exponential;
    } else {
        model.objective = ObjectiveType::Regression;
    }

    for (size_t i = 1; i < tokens.size(); ++i) {
        const std::string& tok = tokens[i];
        if (tok.rfind("sigmoid:", 0) == 0) {
            model.sigmoid_alpha = std::stof(tok.substr(8));
        }
    }
}

} // namespace

RawModel ParseModelFromString(std::string_view content) {
    RawModel model;
    RawTree* cur_tree = nullptr;
    bool in_trees = false;

    size_t pos = 0;
    const size_t total_len = content.size();

    while (pos < total_len) {
        size_t nl = content.find('\n', pos);
        std::string_view line = (nl == std::string_view::npos)
                                    ? content.substr(pos)
                                    : content.substr(pos, nl - pos);
        pos = (nl == std::string_view::npos) ? total_len : nl + 1;

        line = TrimRightCR(line);
        if (line.empty()) continue;

        if (line == "end of trees") {
            break;
        }
        if (line == "average_output") {
            model.average_output = true;
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string_view::npos) {
            continue;
        }

        std::string_view key = line.substr(0, eq);
        std::string_view val = line.substr(eq + 1);

        if (key == "Tree") {
            in_trees = true;
            model.trees.emplace_back();
            cur_tree = &model.trees.back();
            continue;
        }

        if (!in_trees) {
            if (key == "version") {
                model.version = std::string(val);
            } else if (key == "num_class") {
                std::from_chars(val.data(), val.data() + val.size(), model.num_class);
            } else if (key == "num_tree_per_iteration") {
                std::from_chars(val.data(), val.data() + val.size(), model.num_tree_per_iteration);
            } else if (key == "max_feature_idx") {
                std::from_chars(val.data(), val.data() + val.size(), model.max_feature_idx);
            } else if (key == "objective") {
                ParseObjectiveConfig(val, model);
            } else if (key == "average_output") {
                model.average_output = (val != "0" && val != "false");
            } else if (key == "feature_names") {
                ParseStringTokens(val, model.feature_names);
            }
        } else if (cur_tree != nullptr) {
            if (key == "num_leaves") {
                std::from_chars(val.data(), val.data() + val.size(), cur_tree->num_leaves);
            } else if (key == "num_cat") {
                std::from_chars(val.data(), val.data() + val.size(), cur_tree->num_cat);
                if (cur_tree->num_cat > 0) model.has_categorical = true;
            } else if (key == "is_linear") {
                int v = 0;
                std::from_chars(val.data(), val.data() + val.size(), v);
                cur_tree->is_linear = (v != 0);
                if (cur_tree->is_linear) model.has_linear = true;
            } else if (key == "shrinkage") {
                std::vector<double> tmp;
                ParseDoubleArray(val, tmp, 1);
                if (!tmp.empty()) cur_tree->shrinkage = tmp[0];
            } else if (key == "split_feature") {
                size_t hint = cur_tree->num_leaves > 1 ? static_cast<size_t>(cur_tree->num_leaves - 1) : 0;
                ParseIntArray(val, cur_tree->split_feature, hint);
            } else if (key == "threshold") {
                size_t hint = cur_tree->num_leaves > 1 ? static_cast<size_t>(cur_tree->num_leaves - 1) : 0;
                ParseDoubleArray(val, cur_tree->threshold, hint);
            } else if (key == "decision_type") {
                std::vector<int32_t> tmp;
                size_t hint = cur_tree->num_leaves > 1 ? static_cast<size_t>(cur_tree->num_leaves - 1) : 0;
                ParseIntArray(val, tmp, hint);
                cur_tree->decision_type.resize(tmp.size());
                for (size_t i = 0; i < tmp.size(); ++i) {
                    cur_tree->decision_type[i] = static_cast<int8_t>(tmp[i]);
                    if ((cur_tree->decision_type[i] & 1) != 0) {
                        model.has_categorical = true;
                    }
                }
            } else if (key == "left_child") {
                size_t hint = cur_tree->num_leaves > 1 ? static_cast<size_t>(cur_tree->num_leaves - 1) : 0;
                ParseIntArray(val, cur_tree->left_child, hint);
            } else if (key == "right_child") {
                size_t hint = cur_tree->num_leaves > 1 ? static_cast<size_t>(cur_tree->num_leaves - 1) : 0;
                ParseIntArray(val, cur_tree->right_child, hint);
            } else if (key == "leaf_value") {
                size_t hint = cur_tree->num_leaves > 0 ? static_cast<size_t>(cur_tree->num_leaves) : 0;
                ParseDoubleArray(val, cur_tree->leaf_value, hint);
            }
        }
    }

    if (model.trees.empty()) {
        throw std::runtime_error("Invalid or empty LightGBM model: no trees found");
    }
    if (model.num_tree_per_iteration <= 0) {
        model.num_tree_per_iteration = (model.num_class > 1) ? model.num_class : 1;
    }

    return model;
}

RawModel ParseModelFromFile(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open LightGBM model file: " + filepath);
    }
    const std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string buffer(static_cast<size_t>(size), '\0');
    if (size > 0 && !ifs.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read LightGBM model file: " + filepath);
    }
    return ParseModelFromString(buffer);
}

} // namespace fast_scorer
