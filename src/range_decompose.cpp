#include "smartivf/range_decompose.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

namespace smartivf {

namespace {

struct Region {
    std::vector<std::uint64_t> min;
    std::vector<std::uint64_t> max;
};

}  // namespace

Result<std::vector<SFCInterval>>
range_decompose(std::span<const std::array<float, 2>> filter_ranges,
                const sfc::SFC& curve,
                std::uint8_t bits,
                std::size_t budget) {
    const std::size_t dims = filter_ranges.size();
    if (dims == 0) return std::vector<SFCInterval>{};

    const std::uint64_t max_coord =
        (bits >= 64) ? std::numeric_limits<std::uint64_t>::max()
                     : ((std::uint64_t{1} << bits) - 1);

    // Quantise the normalised filter ranges to integer coordinates.
    std::vector<std::array<std::uint64_t, 2>> quantised(dims);
    for (std::size_t d = 0; d < dims; ++d) {
        double lo = static_cast<double>(filter_ranges[d][0]);
        double hi = static_cast<double>(filter_ranges[d][1]);
        if (lo < 0.0) lo = 0.0;
        if (hi > 1.0) hi = 1.0;
        quantised[d][0] = static_cast<std::uint64_t>(lo * static_cast<double>(max_coord));
        quantised[d][1] = static_cast<std::uint64_t>(hi * static_cast<double>(max_coord));
    }

    std::vector<SFCInterval> intervals;

    Region root{std::vector<std::uint64_t>(dims, 0),
                std::vector<std::uint64_t>(dims, max_coord)};

    std::deque<Region> queue;
    queue.push_back(std::move(root));

    while (!queue.empty()) {
        Region current = std::move(queue.front());
        queue.pop_front();

        bool disjoint = false;
        for (std::size_t d = 0; d < dims; ++d) {
            if (current.max[d] < quantised[d][0] || current.min[d] > quantised[d][1]) {
                disjoint = true;
                break;
            }
        }
        if (disjoint) continue;

        bool fully_contained = true;
        for (std::size_t d = 0; d < dims; ++d) {
            if (current.min[d] < quantised[d][0] || current.max[d] > quantised[d][1]) {
                fully_contained = false;
                break;
            }
        }
        bool atomic = true;
        for (std::size_t d = 0; d < dims; ++d) {
            if (current.min[d] != current.max[d]) {
                atomic = false;
                break;
            }
        }
        const bool budget_exceeded =
            intervals.size() + queue.size() + 1 >= budget;

        if (fully_contained || atomic || budget_exceeded) {
            const auto start = curve.encode(current.min);
            const auto end   = curve.encode(current.max);
            intervals.push_back(SFCInterval{start, end});
            continue;
        }

        // Bisect along the widest dimension.
        std::uint64_t max_width = 0;
        std::size_t widest = 0;
        for (std::size_t d = 0; d < dims; ++d) {
            const auto w = current.max[d] - current.min[d];
            if (w > max_width) {
                max_width = w;
                widest = d;
            }
        }
        const std::uint64_t mid =
            (current.min[widest] + current.max[widest]) / 2;

        Region left  = current;
        Region right = std::move(current);
        left.max[widest]  = mid;
        right.min[widest] = mid + 1;
        queue.push_back(std::move(left));
        queue.push_back(std::move(right));
    }

    // Sort by start.
    std::ranges::sort(intervals, {}, &SFCInterval::start);

    // First pass: merge overlapping or contiguous intervals.
    std::vector<SFCInterval> merged;
    if (!intervals.empty()) {
        SFCInterval cur = intervals.front();
        for (std::size_t i = 1; i < intervals.size(); ++i) {
            const SFCInterval& nxt = intervals[i];
            if (nxt.start <= cur.end) {
                if (nxt.end > cur.end) cur.end = nxt.end;
            } else if (nxt.start == cur.end + 1) {
                if (nxt.end > cur.end) cur.end = nxt.end;
            } else {
                merged.push_back(cur);
                cur = nxt;
            }
        }
        merged.push_back(cur);
    }
    intervals = std::move(merged);

    // Second pass: collapse the smallest gap until the budget is met.
    while (intervals.size() > budget) {
        std::uint64_t min_gap = std::numeric_limits<std::uint64_t>::max();
        std::size_t min_idx = 0;
        for (std::size_t i = 0; i + 1 < intervals.size(); ++i) {
            const std::uint64_t gap = intervals[i + 1].start - intervals[i].end;
            if (gap < min_gap) {
                min_gap = gap;
                min_idx = i;
            }
        }
        intervals[min_idx].end = intervals[min_idx + 1].end;
        intervals.erase(intervals.begin() + static_cast<std::ptrdiff_t>(min_idx + 1));
    }

    return intervals;
}

}  // namespace smartivf
