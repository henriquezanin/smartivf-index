#include "smartivf/utils/ranges.hpp"

#include <charconv>
#include <cstddef>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace smartivf::utils {

namespace {

// Trim ASCII whitespace from both ends.
[[nodiscard]] std::string_view trim(std::string_view s) noexcept {
    auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
    while (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_ws(s.back()))  s.remove_suffix(1);
    return s;
}

// Parse a single CSV line: split on commas, trim each field. Empty input → empty result.
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

[[nodiscard]] Result<float> parse_f32(std::string_view s, std::size_t row, std::size_t col) {
    float v{};
    const auto* first = s.data();
    const auto* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, v);
    if (ec != std::errc{} || ptr != last) {
        return std::unexpected(IndexError{std::format(
            "invalid float at row {} col {}: {:?}", row, col, std::string(s))});
    }
    return v;
}

}  // namespace

Result<std::vector<QueryRange>> read_ranges_csv(const std::string& path, bool has_header) {
    std::ifstream f(path);
    if (!f) {
        return std::unexpected(IndexError{std::format("failed to open {}", path)});
    }

    std::vector<QueryRange> out;
    std::string line;
    std::size_t row_idx = 0;
    while (std::getline(f, line)) {
        if (has_header && row_idx == 0) {
            ++row_idx;
            continue;
        }
        auto fields = split_csv_line(line);
        // Skip blank trailing line(s).
        if (fields.size() == 1 && fields[0].empty()) {
            ++row_idx;
            continue;
        }
        if (fields.size() % 2 != 0) {
            return std::unexpected(IndexError{std::format(
                "row {}: expected even column count, got {}", row_idx, fields.size())});
        }
        const std::size_t dims = fields.size() / 2;
        QueryRange ranges(dims);
        for (std::size_t d = 0; d < dims; ++d) {
            auto lo = parse_f32(fields[d * 2],     row_idx, d * 2);
            if (!lo) return std::unexpected(lo.error());
            auto hi = parse_f32(fields[d * 2 + 1], row_idx, d * 2 + 1);
            if (!hi) return std::unexpected(hi.error());
            ranges[d] = {*lo, *hi};
        }
        out.push_back(std::move(ranges));
        ++row_idx;
    }
    return out;
}

}  // namespace smartivf::utils
