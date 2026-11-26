#pragma once

#include "fast_scorer/model.h"
#include <string>
#include <string_view>

namespace fast_scorer {

RawModel ParseModelFromString(std::string_view content);
RawModel ParseModelFromFile(const std::string& filepath);

} // namespace fast_scorer
