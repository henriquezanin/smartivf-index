#include "smartivf/utils/scalar.hpp"

#include <charconv>
#include <cstddef>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace smartivf::utils {

namespace {

[[nodiscard]] std::string_view trim(std::string_view s) noexcept {
    auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
    while (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_ws(s.back()))  s.remove_suffix(1);
    return s;
}

[[nodiscard]] std::vector<std::string_view> split_csv_line(std::string_view line) {
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            fields.push_back(trim(line.substr(start, i - start)));
            start = i + 1;
        }
    }
    return fields;
}

}  // namespace

Result<std::vector<std::vector<float>>>
read_scalar_csv(const std::string& path, bool has_header) {
    std::ifstream f(path);
    if (!f) {
        return std::unexpected(IndexError{std::format("failed to open {}", path)});
    }

    std::vector<std::vector<float>> out;
    std::string line;
    std::size_t row_idx = 0;
    while (std::getline(f, line)) {
        if (has_header && row_idx == 0) {
            ++row_idx;
            continue;
        }
        auto fields = split_csv_line(line);
        if (fields.size() == 1 && fields[0].empty()) {
            ++row_idx;
            continue;
        }
        std::vector<float> row(fields.size());
        for (std::size_t i = 0; i < fields.size(); ++i) {
            float v{};
            const auto* first = fields[i].data();
            const auto* last = fields[i].data() + fields[i].size();
            auto [ptr, ec] = std::from_chars(first, last, v);
            if (ec != std::errc{} || ptr != last) {
                return std::unexpected(IndexError{std::format(
                    "invalid float at row {} col {}: {:?}",
                    row_idx, i, std::string(fields[i]))});
            }
            row[i] = v;
        }
        out.push_back(std::move(row));
        ++row_idx;
    }
    return out;
}

}  // namespace smartivf::utils
