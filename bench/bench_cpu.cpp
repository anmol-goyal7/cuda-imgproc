// bench_cpu — CPU-only benchmark (single-thread + OpenMP), runnable on any
// machine with g++: `make bench-cpu`. Useful for getting CPU baselines early
// and for machines without a GPU; the *canonical* results CSV, with GPU
// columns filled and all numbers from one machine, is produced by bench_gpu
// on the benchmark GPU (see colab/).
//
// Methodology matches bench_gpu: seeded random inputs generated outside timed
// regions, 10 warm-up + 100 timed runs per measurement, medians to the CSV,
// stddev on stdout. Batch rows: 1 warm-up + 5 timed runs (each run already
// covers 100 images).
//
// Output: results/cpu_baseline_<hostname>.csv — same schema as the canonical
// CSV with the GPU and speedup columns left empty. The README renderer
// ignores these files by design (it globs results/results_*.csv).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <omp.h>
#include <unistd.h>  // gethostname

#include "cuda_imgproc.hpp"

namespace {

constexpr int kWarmup = 10;
constexpr int kRuns = 100;
constexpr int kBatchWarmup = 1;
constexpr int kBatchRuns = 5;

struct Stats {
    double median_ms = 0.0;
    double stddev_ms = 0.0;
};

Stats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    Stats s;
    s.median_ms = (n % 2 == 1) ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (double v : samples) var += (v - mean) * (v - mean);
    s.stddev_ms = std::sqrt(var / static_cast<double>(n));
    return s;
}

template <typename F>
Stats time_host_ms(F&& fn, int warmup = kWarmup, int runs = kRuns) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return summarize(std::move(samples));
}

struct Row {
    std::string op;
    std::string resolution;
    Stats cpu_1t, omp;
};
std::vector<Row> g_rows;

void add_row(Row r) {
    std::printf("%-24s %-22s 1t %9.3f ±%7.3f | omp %9.3f ±%7.3f  [ms]\n", r.op.c_str(),
                r.resolution.c_str(), r.cpu_1t.median_ms, r.cpu_1t.stddev_ms, r.omp.median_ms,
                r.omp.stddev_ms);
    std::fflush(stdout);
    g_rows.push_back(std::move(r));
}

std::string res(int w, int h) { return std::to_string(w) + "x" + std::to_string(h); }

void bench_all() {
    const std::vector<int> sizes = {256, 512, 1024, 2048, 4096};

    for (int s : sizes) {
        const cig::Image img = cig::random_image(s, s, 3, 42);
        std::vector<std::uint8_t> out(img.n_pixels());
        Row r{"rgb_to_gray", res(s, s), {}, {}};
        r.cpu_1t =
            time_host_ms([&] { cig::cpu::rgb_to_gray(img.data.data(), out.data(), s, s); });
        r.omp =
            time_host_ms([&] { cig::cpu_omp::rgb_to_gray(img.data.data(), out.data(), s, s); });
        add_row(std::move(r));
    }

    for (int s : sizes) {
        const cig::Image img = cig::random_image(s, s, 1, 42);
        std::vector<std::uint8_t> out(img.n_pixels());
        Row r{"gaussian5x5", res(s, s), {}, {}};
        r.cpu_1t = time_host_ms([&] {
            cig::cpu::convolve2d(img.data.data(), out.data(), s, s, cig::Filter::gaussian5x5);
        });
        r.omp = time_host_ms([&] {
            cig::cpu_omp::convolve2d(img.data.data(), out.data(), s, s,
                                     cig::Filter::gaussian5x5);
        });
        add_row(std::move(r));
    }

    for (int s : sizes) {
        const cig::Image img = cig::random_image(s, s, 1, 42);
        std::vector<std::uint8_t> out(img.n_pixels());
        Row r{"equalize_hist", res(s, s), {}, {}};
        r.cpu_1t = time_host_ms(
            [&] { cig::cpu::equalize_hist(img.data.data(), out.data(), img.n_pixels()); });
        r.omp = time_host_ms(
            [&] { cig::cpu_omp::equalize_hist(img.data.data(), out.data(), img.n_pixels()); });
        add_row(std::move(r));
    }

    {
        const int sw = 4096, sh = 4096, dw = 1024, dh = 1024, ch = 3;
        const cig::Image img = cig::random_image(sw, sh, ch, 42);
        std::vector<std::uint8_t> out(static_cast<std::size_t>(dw) * dh * ch);
        Row r{"resize_bilinear", "4096x4096->1024x1024", {}, {}};
        r.cpu_1t = time_host_ms(
            [&] { cig::cpu::resize(img.data.data(), sw, sh, out.data(), dw, dh, ch); });
        r.omp = time_host_ms(
            [&] { cig::cpu_omp::resize(img.data.data(), sw, sh, out.data(), dw, dh, ch); });
        add_row(std::move(r));
    }

    // Batch rows: per-image medians over a 100-image batch.
    {
        const int w = 1024, h = 1024, count = 100;
        std::vector<cig::Image> rgbs, grays;
        rgbs.reserve(count);
        grays.reserve(count);
        for (int i = 0; i < count; ++i) rgbs.push_back(cig::random_image(w, h, 3, 42 + i));
        for (int i = 0; i < count; ++i) grays.push_back(cig::random_image(w, h, 1, 142 + i));
        std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * h);

        Row rg{"rgb_to_gray_batch100", res(w, h), {}, {}};
        Stats s1 = time_host_ms(
            [&] {
                for (const auto& im : rgbs) cig::cpu::rgb_to_gray(im.data.data(), out.data(), w, h);
            },
            kBatchWarmup, kBatchRuns);
        Stats so = time_host_ms(
            [&] {
                for (const auto& im : rgbs)
                    cig::cpu_omp::rgb_to_gray(im.data.data(), out.data(), w, h);
            },
            kBatchWarmup, kBatchRuns);
        rg.cpu_1t = {s1.median_ms / count, s1.stddev_ms / count};
        rg.omp = {so.median_ms / count, so.stddev_ms / count};
        add_row(std::move(rg));

        Row rc{"gaussian5x5_batch100", res(w, h), {}, {}};
        s1 = time_host_ms(
            [&] {
                for (const auto& im : grays)
                    cig::cpu::convolve2d(im.data.data(), out.data(), w, h,
                                         cig::Filter::gaussian5x5);
            },
            kBatchWarmup, kBatchRuns);
        so = time_host_ms(
            [&] {
                for (const auto& im : grays)
                    cig::cpu_omp::convolve2d(im.data.data(), out.data(), w, h,
                                             cig::Filter::gaussian5x5);
            },
            kBatchWarmup, kBatchRuns);
        rc.cpu_1t = {s1.median_ms / count, s1.stddev_ms / count};
        rc.omp = {so.median_ms / count, so.stddev_ms / count};
        add_row(std::move(rc));
    }
}

}  // namespace

int main() try {
    std::printf("cpu-only benchmark | OpenMP threads: %d\n\n", omp_get_max_threads());
    bench_all();

    std::filesystem::create_directories("results");
    // gethostname, not getenv("HOSTNAME"): HOSTNAME is a shell variable most
    // shells never export, so the env lookup almost always came up empty.
    char hostbuf[256] = {};
    std::string host = "local";
    if (gethostname(hostbuf, sizeof(hostbuf) - 1) == 0 && hostbuf[0] != '\0') {
        host = hostbuf;
    }
    const std::string path = "results/cpu_baseline_" + host + ".csv";
    std::ofstream csv(path);
    csv << "# cpu-only run; omp_threads: " << omp_get_max_threads() << "\n";
    csv << "op,resolution,cpu_1t_ms,omp_ms,gpu_kernel_ms,gpu_wall_ms,speedup_vs_1t,"
           "speedup_vs_omp\n";
    char buf[256];
    for (const Row& r : g_rows) {
        std::snprintf(buf, sizeof(buf), "%s,%s,%.6f,%.6f,,,,\n", r.op.c_str(),
                      r.resolution.c_str(), r.cpu_1t.median_ms, r.omp.median_ms);
        csv << buf;
    }
    csv.close();
    std::printf("\nwrote %s\n", path.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_cpu failed: %s\n", e.what());
    return 1;
}
