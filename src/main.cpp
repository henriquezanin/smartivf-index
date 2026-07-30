// SmartIVF CLI: the build, search and run subcommands.

#include "smartivf/smartivf.hpp"
#include "smartivf/store.hpp"
#include "smartivf/utils/fvecs.hpp"
#include "smartivf/utils/ivecs.hpp"
#include "smartivf/utils/ranges.hpp"

#include <CLI/CLI.hpp>
#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct BuildArgs {
    std::string base_vecs = "datasets/sift/sift_base.fvecs";
    std::string base_scalars = "datasets/scalar/uniform_scalars.csv";
    std::string out_index = "index.bin";
    std::string sfc = "z-order";
    int scalar_dims = 4;
    int scalar_bits = 8;
    int partition_threshold = 1000;
    int partitions = 50;
    std::string metric = "l2";
    std::uint64_t seed = 42;
    int threads = 0;
};

struct SearchArgs {
    std::string in_index = "index.bin";
    std::string query_vecs = "datasets/sift/sift_query.fvecs";
    std::string query_ranges = "datasets/scalar/query_ranges.csv";
    std::string ground_truth;
    int k = 100;
    int sfc_budget = 1000;
    int min_probe = 10;
    int max_probe = 100;
    int candidate_target = 1000;
    std::string sfc = "z-order";
    int scalar_dims = 4;
    int scalar_bits = 8;
    int partition_threshold = 1000;
    int partitions = 50;
    std::string metric = "l2";
    int threads = 0;
};

struct RunArgs {
    BuildArgs b;
    SearchArgs s;
};

[[nodiscard]] smartivf::MetricType parse_metric(std::string_view s) {
    if (s == "cosine") return smartivf::MetricType::Cosine;
    return smartivf::MetricType::L2;  // default
}

[[nodiscard]] smartivf::SFCType parse_sfc(std::string_view s) {
    if (s == "z-order" || s == "zorder") return smartivf::SFCType::ZOrder;
    throw std::invalid_argument(std::string("unsupported SFC: ") + std::string(s));
}

[[nodiscard]] smartivf::SmartIVFConfig
make_cfg(const std::string& sfc, int sdims, int sbits, int pt, int parts,
         const std::string& metric, std::uint64_t seed, int threads) {
    return smartivf::SmartIVFConfig{
        .sfc                        = parse_sfc(sfc),
        .scalar_dimensions          = sdims,
        .scalar_bits_per_dimension  = static_cast<std::uint8_t>(sbits),
        .partition_threshold        = pt,
        .partitions                 = parts,
        .metric                     = parse_metric(metric),
        .seed                       = seed,
        .threads                    = threads,
    };
}

int do_build(const BuildArgs& a) {
    if (a.threads > 0) omp_set_num_threads(a.threads);
    std::println("Executing build phase...");
    smartivf::SmartIVF s(make_cfg(a.sfc, a.scalar_dims, a.scalar_bits,
                                  a.partition_threshold, a.partitions,
                                  a.metric, a.seed, a.threads));
    auto idx = s.build_index(a.base_vecs, a.base_scalars);
    if (!idx) { std::println("Error building index: {}", idx.error().message); return 1; }
    std::println("Saving index to '{}'...", a.out_index);
    auto store = smartivf::store_index(*idx, a.out_index);
    if (!store) { std::println("Error saving index: {}", store.error().message); return 1; }
    std::println("Build Phase completed successfully.");
    return 0;
}

int do_search(const SearchArgs& a) {
    if (a.threads > 0) omp_set_num_threads(a.threads);
    std::println("Loading index from '{}'...", a.in_index);
    auto idx = smartivf::load_index(a.in_index);
    if (!idx) { std::println("Error loading index: {}", idx.error().message); return 1; }
    std::println("Index loaded successfully with {} partitions.", idx->partitions.size());

    smartivf::SmartIVF s(make_cfg(a.sfc, a.scalar_dims, a.scalar_bits,
                                  a.partition_threshold, a.partitions,
                                  a.metric, /*seed=*/42, a.threads));

    std::println("Loading query objects...");
    auto qv = smartivf::utils::read_fvecs(a.query_vecs);
    if (!qv) { std::println("Error loading query vectors: {}", qv.error().message); return 1; }
    auto qr = smartivf::utils::read_ranges_csv(a.query_ranges);
    if (!qr) { std::println("Error loading query ranges: {}", qr.error().message); return 1; }

    std::vector<std::vector<std::int32_t>> gt;
    if (!a.ground_truth.empty()) {
        auto g = smartivf::utils::read_ivecs(a.ground_truth);
        if (!g) { std::println("Error loading ground truth: {}", g.error().message); return 1; }
        gt = std::move(*g);
    }

    std::println("Starting Search Phase...");
    const std::size_t totalQ = std::min(qv->size(), qr->size());

    // Per-thread accumulators avoid OpenMP atomics on the hot path.
    const int T = (a.threads > 0) ? a.threads : omp_get_max_threads();
    std::vector<std::int64_t> totalDCO_per(static_cast<std::size_t>(T), 0);
    std::vector<std::int64_t> correct_per(static_cast<std::size_t>(T), 0);
    std::vector<std::int64_t> totalGT_per(static_cast<std::size_t>(T), 0);

    const auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 64)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(totalQ); ++i) {
        const int tid = omp_get_thread_num();
        const std::size_t qi = static_cast<std::size_t>(i);
        auto r = s.search(*idx,
                          (*qv)[qi],
                          std::span<const std::array<float, 2>>{(*qr)[qi]},
                          a.k, static_cast<std::size_t>(a.sfc_budget),
                          a.min_probe, a.max_probe, a.candidate_target);
        if (!r) continue;
        totalDCO_per[static_cast<std::size_t>(tid)] += r->dco;
        if (!gt.empty() && qi < gt.size()) {
            std::unordered_set<std::int32_t> hits;
            hits.reserve(r->results.size());
            for (const auto& o : r->results) hits.insert(o->id);
            for (std::int32_t gid : gt[qi]) {
                if (gid == -1) continue;
                ++totalGT_per[static_cast<std::size_t>(tid)];
                if (hits.contains(gid)) ++correct_per[static_cast<std::size_t>(tid)];
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const auto total_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);

    std::int64_t totalDCO = 0, correct = 0, totalGT = 0;
    for (int t = 0; t < T; ++t) {
        totalDCO += totalDCO_per[static_cast<std::size_t>(t)];
        correct  += correct_per[static_cast<std::size_t>(t)];
        totalGT  += totalGT_per[static_cast<std::size_t>(t)];
    }

    const double avg_ns = static_cast<double>(total_time.count()) / static_cast<double>(totalQ);
    std::println("");
    std::println("Search Phase completed.");
    std::println("Total queries processed: {}", totalQ);
    std::println("Average search time per query: {} ns ({:.3f} ms)", static_cast<std::int64_t>(avg_ns), avg_ns / 1e6);
    std::println("Distance computations per query: {:.2f}",
                 static_cast<double>(totalDCO) / static_cast<double>(totalQ));
    if (totalGT > 0) {
        const double recall = static_cast<double>(correct) / static_cast<double>(totalGT);
        std::println("Recall@{}: {:.4f}", a.k, recall);
    }
    return 0;
}

int do_run(const RunArgs& a) {
    if (a.b.threads > 0) omp_set_num_threads(a.b.threads);
    std::println("Starting Build Phase...");
    const auto t0 = std::chrono::steady_clock::now();
    smartivf::SmartIVF s(make_cfg(a.b.sfc, a.b.scalar_dims, a.b.scalar_bits,
                                  a.b.partition_threshold, a.b.partitions,
                                  a.b.metric, a.b.seed, a.b.threads));
    auto idx = s.build_index(a.b.base_vecs, a.b.base_scalars);
    if (!idx) { std::println("Error building index: {}", idx.error().message); return 1; }
    const auto t1 = std::chrono::steady_clock::now();
    const auto build_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::println("Build Phase completed in {} ns ({:.3f} s)\n",
                 build_ns, static_cast<double>(build_ns) / 1e9);

    SearchArgs sa = a.s;
    sa.in_index = "";  // no index file: the in-memory idx is used directly
    sa.scalar_dims = a.b.scalar_dims;
    sa.scalar_bits = a.b.scalar_bits;
    sa.partition_threshold = a.b.partition_threshold;
    sa.partitions = a.b.partitions;
    sa.metric = a.b.metric;
    sa.sfc = a.b.sfc;

    std::println("Loading query objects...");
    auto qv = smartivf::utils::read_fvecs(sa.query_vecs);
    if (!qv) { std::println("Error loading query vectors: {}", qv.error().message); return 1; }
    auto qr = smartivf::utils::read_ranges_csv(sa.query_ranges);
    if (!qr) { std::println("Error loading query ranges: {}", qr.error().message); return 1; }

    std::vector<std::vector<std::int32_t>> gt;
    if (!sa.ground_truth.empty()) {
        auto g = smartivf::utils::read_ivecs(sa.ground_truth);
        if (!g) { std::println("Error loading ground truth: {}", g.error().message); return 1; }
        gt = std::move(*g);
    }

    std::println("Starting Search Phase...");
    const std::size_t totalQ = std::min(qv->size(), qr->size());

    const int T = (sa.threads > 0) ? sa.threads : omp_get_max_threads();
    std::vector<std::int64_t> totalDCO_per(static_cast<std::size_t>(T), 0);
    std::vector<std::int64_t> correct_per(static_cast<std::size_t>(T), 0);
    std::vector<std::int64_t> totalGT_per(static_cast<std::size_t>(T), 0);

    const auto s0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic, 64)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(totalQ); ++i) {
        const int tid = omp_get_thread_num();
        const std::size_t qi = static_cast<std::size_t>(i);
        auto r = s.search(*idx,
                          (*qv)[qi],
                          std::span<const std::array<float, 2>>{(*qr)[qi]},
                          sa.k, static_cast<std::size_t>(sa.sfc_budget),
                          sa.min_probe, sa.max_probe, sa.candidate_target);
        if (!r) continue;
        totalDCO_per[static_cast<std::size_t>(tid)] += r->dco;
        if (!gt.empty() && qi < gt.size()) {
            std::unordered_set<std::int32_t> hits;
            hits.reserve(r->results.size());
            for (const auto& o : r->results) hits.insert(o->id);
            for (std::int32_t gid : gt[qi]) {
                if (gid == -1) continue;
                ++totalGT_per[static_cast<std::size_t>(tid)];
                if (hits.contains(gid)) ++correct_per[static_cast<std::size_t>(tid)];
            }
        }
    }

    const auto s1 = std::chrono::steady_clock::now();
    const auto total_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0);

    std::int64_t totalDCO = 0, correct = 0, totalGT = 0;
    for (int t = 0; t < T; ++t) {
        totalDCO += totalDCO_per[static_cast<std::size_t>(t)];
        correct  += correct_per[static_cast<std::size_t>(t)];
        totalGT  += totalGT_per[static_cast<std::size_t>(t)];
    }

    const double avg_ns = static_cast<double>(total_time.count()) / static_cast<double>(totalQ);
    std::println("");
    std::println("Search Phase completed.");
    std::println("Total queries processed: {}", totalQ);
    std::println("Average search time per query: {} ns ({:.3f} ms)", static_cast<std::int64_t>(avg_ns), avg_ns / 1e6);
    std::println("Distance computations per query: {:.2f}",
                 static_cast<double>(totalDCO) / static_cast<double>(totalQ));
    if (totalGT > 0) {
        const double recall = static_cast<double>(correct) / static_cast<double>(totalGT);
        std::println("Recall@{}: {:.4f}", sa.k, recall);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"SmartIVF: range-filtered approximate nearest-neighbor search"};
    app.require_subcommand(1);

    BuildArgs ba;
    auto* bcmd = app.add_subcommand("build", "construct and save the index");
    bcmd->add_option("--base-vecs",            ba.base_vecs);
    bcmd->add_option("--base-scalars",         ba.base_scalars);
    bcmd->add_option("--index-out",            ba.out_index);
    bcmd->add_option("--sfc",                  ba.sfc);
    bcmd->add_option("--scalar-dims",          ba.scalar_dims);
    bcmd->add_option("--scalar-bits",          ba.scalar_bits);
    bcmd->add_option("--partition-threshold",  ba.partition_threshold);
    bcmd->add_option("--partitions",           ba.partitions);
    bcmd->add_option("--metric",               ba.metric);
    bcmd->add_option("--seed",                 ba.seed,    "k-means RNG seed");
    bcmd->add_option("--threads",              ba.threads, "0 = max available");

    SearchArgs sa;
    auto* scmd = app.add_subcommand("search", "query a pre-built index");
    scmd->add_option("--index-in",             sa.in_index);
    scmd->add_option("--query-vecs",           sa.query_vecs);
    scmd->add_option("--query-ranges",         sa.query_ranges);
    scmd->add_option("--ground-truth",         sa.ground_truth);
    scmd->add_option("--k",                    sa.k);
    scmd->add_option("--sfc-budget",           sa.sfc_budget);
    scmd->add_option("--min-probe",            sa.min_probe);
    scmd->add_option("--max-probe",            sa.max_probe);
    scmd->add_option("--candidate-target",     sa.candidate_target);
    scmd->add_option("--sfc",                  sa.sfc);
    scmd->add_option("--scalar-dims",          sa.scalar_dims);
    scmd->add_option("--scalar-bits",          sa.scalar_bits);
    scmd->add_option("--partition-threshold",  sa.partition_threshold);
    scmd->add_option("--partitions",           sa.partitions);
    scmd->add_option("--metric",               sa.metric);
    scmd->add_option("--threads",              sa.threads, "0 = max available");

    RunArgs ra;
    auto* rcmd = app.add_subcommand("run", "build and search in one shot");
    rcmd->add_option("--base-vecs",            ra.b.base_vecs);
    rcmd->add_option("--base-scalars",         ra.b.base_scalars);
    rcmd->add_option("--query-vecs",           ra.s.query_vecs);
    rcmd->add_option("--query-ranges",         ra.s.query_ranges);
    rcmd->add_option("--ground-truth",         ra.s.ground_truth);
    rcmd->add_option("--sfc",                  ra.b.sfc);
    rcmd->add_option("--scalar-dims",          ra.b.scalar_dims);
    rcmd->add_option("--scalar-bits",          ra.b.scalar_bits);
    rcmd->add_option("--partition-threshold",  ra.b.partition_threshold);
    rcmd->add_option("--partitions",           ra.b.partitions);
    rcmd->add_option("--k",                    ra.s.k);
    rcmd->add_option("--sfc-budget",           ra.s.sfc_budget);
    rcmd->add_option("--min-probe",            ra.s.min_probe);
    rcmd->add_option("--max-probe",            ra.s.max_probe);
    rcmd->add_option("--candidate-target",     ra.s.candidate_target);
    rcmd->add_option("--metric",               ra.b.metric);
    rcmd->add_option("--seed",                 ra.b.seed,    "k-means RNG seed");
    rcmd->add_option("--threads",              ra.b.threads, "0 = max available");

    CLI11_PARSE(app, argc, argv);

    if      (bcmd->parsed()) return do_build(ba);
    else if (scmd->parsed()) return do_search(sa);
    else if (rcmd->parsed()) return do_run(ra);

    return 1;
}
