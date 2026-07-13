// bench_gpu — the canonical benchmark harness. Runs on a CUDA machine
// (Colab T4: `make bench-gpu ARCH=sm_75`) and writes
// results/results_<gpu_name>.csv with CPU single-thread, OpenMP, and GPU
// timings in one shot, so every number in a row comes from the same machine,
// same build, same inputs.
//
// Methodology (mirrored in the README):
//   * inputs are seeded pseudo-random images, generated before timing —
//     no disk I/O anywhere near a timed region
//   * every measurement: 10 warm-up iterations, then 100 timed runs;
//     the CSV records the median, stdout also shows the standard deviation
//     (exception: the CPU sides of the 100-image batch rows use 1 warm-up +
//     5 timed runs — a single run already aggregates 100 images, and 100
//     runs of a multi-second batch would take tens of minutes on 2 vCPUs)
//   * GPU kernel-only time comes from cudaEvent pairs bracketing the launch,
//     which timestamp on the GPU itself and exclude all host overhead and
//     transfers
//   * GPU wall time (host-to-device copy + kernel + device-to-host copy) is
//     measured with std::chrono around the full sequence — the honest
//     "what does one isolated image cost end to end" number
//   * CPU times are std::chrono; OpenMP uses all logical cores
//   * speedup columns = cpu_median / gpu_kernel_median (the kernel-only
//     figure; compare against gpu_wall_ms yourself to see transfer overhead)
//
// Batch rows (op suffix "_batch100"): per-image figures over a 100-image
// batch. The kernel column launches 100 kernels back to back over 100
// *distinct* device-resident images (events around the whole burst, /100) —
// launch-overhead amortization without letting L2 replay a single hot input;
// the wall column re-uploads and downloads every image.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <omp.h>

#include "cuda_imgproc.hpp"

namespace {

constexpr int kWarmup = 10;
constexpr int kRuns = 100;
constexpr int kBatchCpuWarmup = 1;
constexpr int kBatchCpuRuns = 5;

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

// Kernel-only timing: cudaEvents timestamp on the device's own clock, so the
// interval covers exactly the GPU work between them — no launch latency, no
// host jitter, no transfers. One synchronize per sample keeps samples
// independent (at the cost of a launch gap *between* samples, which the
// events don't see).
template <typename L>
Stats time_gpu_kernel_ms(L&& launch, int warmup = kWarmup, int runs = kRuns) {
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventRecord(beg));
        launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        samples.push_back(static_cast<double>(ms));
    }
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return summarize(std::move(samples));
}

struct Row {
    std::string op;
    std::string resolution;
    Stats cpu_1t, omp, gpu_kernel, gpu_wall;
};
std::vector<Row> g_rows;

void print_row(const Row& r) {
    std::printf("%-24s %-22s 1t %9.3f ±%7.3f | omp %9.3f ±%7.3f | gpu %8.4f ±%7.4f | wall %8.3f ±%7.3f  [ms]\n",
                r.op.c_str(), r.resolution.c_str(), r.cpu_1t.median_ms, r.cpu_1t.stddev_ms,
                r.omp.median_ms, r.omp.stddev_ms, r.gpu_kernel.median_ms,
                r.gpu_kernel.stddev_ms, r.gpu_wall.median_ms, r.gpu_wall.stddev_ms);
    std::fflush(stdout);
}

void add_row(Row r) {
    print_row(r);
    g_rows.push_back(std::move(r));
}

// ------------------------------------------------------------- op benches

void bench_rgb_to_gray(int w, int h) {
    const cig::Image img = cig::random_image(w, h, 3, 42);
    const std::size_t n = img.n_pixels();
    std::vector<std::uint8_t> out(n);

    Row r;
    r.op = "rgb_to_gray";
    r.resolution = std::to_string(w) + "x" + std::to_string(h);
    r.cpu_1t = time_host_ms([&] { cig::cpu::rgb_to_gray(img.data.data(), out.data(), w, h); });
    r.omp = time_host_ms([&] { cig::cpu_omp::rgb_to_gray(img.data.data(), out.data(), w, h); });

    cig::GpuBuffer<std::uint8_t> d_in(img.size()), d_out(n);
    d_in.copy_from_host(img.data.data(), img.size());
    r.gpu_kernel = time_gpu_kernel_ms([&] { cig::rgb_to_gray(d_in, d_out, w, h); });
    r.gpu_wall = time_host_ms([&] {
        d_in.copy_from_host(img.data.data(), img.size());
        cig::rgb_to_gray(d_in, d_out, w, h);
        d_out.copy_to_host(out.data(), n);
    });
    add_row(std::move(r));
}

void bench_gaussian(int w, int h) {
    const cig::Image img = cig::random_image(w, h, 1, 42);
    const std::size_t n = img.n_pixels();
    std::vector<std::uint8_t> out(n);
    const int k = cig::upload_filter(cig::Filter::gaussian5x5);

    // CPU numbers are shared by the tiled and naive rows: the CPU reference
    // has no tiled/naive distinction (there is one loop nest).
    const Stats cpu_1t = time_host_ms(
        [&] { cig::cpu::convolve2d(img.data.data(), out.data(), w, h, cig::Filter::gaussian5x5); });
    const Stats omp = time_host_ms([&] {
        cig::cpu_omp::convolve2d(img.data.data(), out.data(), w, h, cig::Filter::gaussian5x5);
    });

    cig::GpuBuffer<std::uint8_t> d_in(n), d_out(n);
    d_in.copy_from_host(img.data.data(), n);

    Row rt;
    rt.op = "gaussian5x5_tiled";
    rt.resolution = std::to_string(w) + "x" + std::to_string(h);
    rt.cpu_1t = cpu_1t;
    rt.omp = omp;
    rt.gpu_kernel = time_gpu_kernel_ms([&] { cig::convolve2d_tiled(d_in, d_out, w, h, k); });
    rt.gpu_wall = time_host_ms([&] {
        d_in.copy_from_host(img.data.data(), n);
        cig::convolve2d_tiled(d_in, d_out, w, h, k);
        d_out.copy_to_host(out.data(), n);
    });
    add_row(std::move(rt));

    Row rn;
    rn.op = "gaussian5x5_naive";
    rn.resolution = std::to_string(w) + "x" + std::to_string(h);
    rn.cpu_1t = cpu_1t;
    rn.omp = omp;
    rn.gpu_kernel = time_gpu_kernel_ms([&] { cig::convolve2d_naive(d_in, d_out, w, h, k); });
    rn.gpu_wall = time_host_ms([&] {
        d_in.copy_from_host(img.data.data(), n);
        cig::convolve2d_naive(d_in, d_out, w, h, k);
        d_out.copy_to_host(out.data(), n);
    });
    add_row(std::move(rn));
}

void bench_resize() {
    const int sw = 4096, sh = 4096, dw = 1024, dh = 1024, ch = 3;
    const cig::Image img = cig::random_image(sw, sh, ch, 42);
    const std::size_t out_bytes = static_cast<std::size_t>(dw) * dh * ch;
    std::vector<std::uint8_t> out(out_bytes);

    Row r;
    r.op = "resize_bilinear";
    r.resolution = "4096x4096->1024x1024";
    r.cpu_1t = time_host_ms(
        [&] { cig::cpu::resize(img.data.data(), sw, sh, out.data(), dw, dh, ch); });
    r.omp = time_host_ms(
        [&] { cig::cpu_omp::resize(img.data.data(), sw, sh, out.data(), dw, dh, ch); });

    cig::GpuBuffer<std::uint8_t> d_in(img.size()), d_out(out_bytes);
    d_in.copy_from_host(img.data.data(), img.size());
    r.gpu_kernel = time_gpu_kernel_ms([&] { cig::resize(d_in, sw, sh, d_out, dw, dh, ch); });
    r.gpu_wall = time_host_ms([&] {
        d_in.copy_from_host(img.data.data(), img.size());
        cig::resize(d_in, sw, sh, d_out, dw, dh, ch);
        d_out.copy_to_host(out.data(), out_bytes);
    });
    add_row(std::move(r));
}

void bench_equalize(int w, int h) {
    const cig::Image img = cig::random_image(w, h, 1, 42);
    const std::size_t n = img.n_pixels();
    std::vector<std::uint8_t> out(n);

    Row r;
    r.op = "equalize_hist";
    r.resolution = std::to_string(w) + "x" + std::to_string(h);
    r.cpu_1t = time_host_ms([&] { cig::cpu::equalize_hist(img.data.data(), out.data(), n); });
    r.omp = time_host_ms([&] { cig::cpu_omp::equalize_hist(img.data.data(), out.data(), n); });

    cig::GpuBuffer<std::uint8_t> d_in(n), d_out(n);
    cig::GpuBuffer<unsigned int> d_hist(256);
    cig::GpuBuffer<std::uint8_t> d_lut(256);
    d_in.copy_from_host(img.data.data(), n);
    // Kernel-only time covers the whole 3-launch algorithm (memset +
    // histogram + scan + remap) — that IS the op.
    r.gpu_kernel =
        time_gpu_kernel_ms([&] { cig::equalize_hist(d_in, d_out, n, d_hist, d_lut); });
    r.gpu_wall = time_host_ms([&] {
        d_in.copy_from_host(img.data.data(), n);
        cig::equalize_hist(d_in, d_out, n, d_hist, d_lut);
        d_out.copy_to_host(out.data(), n);
    });
    add_row(std::move(r));
}

// Batch throughput: 100 distinct 1024x1024 images. Answers "how fast is a
// preprocessing *pipeline*", where the GPU gets to hide launch overhead and
// keep its caches warm, versus the per-image numbers above.
void bench_batch() {
    const int w = 1024, h = 1024, count = 100;
    const std::size_t n = static_cast<std::size_t>(w) * h;

    // --- rgb_to_gray over 100 RGB images ---
    {
        std::vector<cig::Image> imgs;
        imgs.reserve(count);
        for (int i = 0; i < count; ++i) imgs.push_back(cig::random_image(w, h, 3, 42 + i));
        std::vector<std::uint8_t> out(n);

        Row r;
        r.op = "rgb_to_gray_batch100";
        r.resolution = std::to_string(w) + "x" + std::to_string(h);
        const Stats c1 = time_host_ms(
            [&] {
                for (const auto& im : imgs) cig::cpu::rgb_to_gray(im.data.data(), out.data(), w, h);
            },
            kBatchCpuWarmup, kBatchCpuRuns);
        const Stats co = time_host_ms(
            [&] {
                for (const auto& im : imgs)
                    cig::cpu_omp::rgb_to_gray(im.data.data(), out.data(), w, h);
            },
            kBatchCpuWarmup, kBatchCpuRuns);
        r.cpu_1t = {c1.median_ms / count, c1.stddev_ms / count};
        r.omp = {co.median_ms / count, co.stddev_ms / count};

        // All 100 inputs resident on the device (~300 MB), so every launch in
        // the timed burst reads a *different* image. Cycling one buffer would
        // let the whole working set live in L2 across launches and flatter
        // the per-kernel figure — a pipeline over distinct images gets no
        // such replay.
        std::vector<cig::GpuBuffer<std::uint8_t>> d_ins;
        d_ins.reserve(count);
        for (const auto& im : imgs) {
            d_ins.emplace_back(n * 3);
            d_ins.back().copy_from_host(im.data.data(), n * 3);
        }
        cig::GpuBuffer<std::uint8_t> d_in(n * 3), d_out(n);
        const Stats gk = time_gpu_kernel_ms([&] {
            for (int i = 0; i < count; ++i) cig::rgb_to_gray(d_ins[i], d_out, w, h);
        });
        const Stats gw = time_host_ms([&] {
            for (const auto& im : imgs) {
                d_in.copy_from_host(im.data.data(), n * 3);
                cig::rgb_to_gray(d_in, d_out, w, h);
                d_out.copy_to_host(out.data(), n);
            }
        });
        r.gpu_kernel = {gk.median_ms / count, gk.stddev_ms / count};
        r.gpu_wall = {gw.median_ms / count, gw.stddev_ms / count};
        add_row(std::move(r));
    }

    // --- gaussian5x5 (tiled) over 100 grayscale images ---
    {
        std::vector<cig::Image> imgs;
        imgs.reserve(count);
        for (int i = 0; i < count; ++i) imgs.push_back(cig::random_image(w, h, 1, 142 + i));
        std::vector<std::uint8_t> out(n);
        const int k = cig::upload_filter(cig::Filter::gaussian5x5);

        Row r;
        r.op = "gaussian5x5_batch100";
        r.resolution = std::to_string(w) + "x" + std::to_string(h);
        const Stats c1 = time_host_ms(
            [&] {
                for (const auto& im : imgs)
                    cig::cpu::convolve2d(im.data.data(), out.data(), w, h,
                                         cig::Filter::gaussian5x5);
            },
            kBatchCpuWarmup, kBatchCpuRuns);
        const Stats co = time_host_ms(
            [&] {
                for (const auto& im : imgs)
                    cig::cpu_omp::convolve2d(im.data.data(), out.data(), w, h,
                                             cig::Filter::gaussian5x5);
            },
            kBatchCpuWarmup, kBatchCpuRuns);
        r.cpu_1t = {c1.median_ms / count, c1.stddev_ms / count};
        r.omp = {co.median_ms / count, co.stddev_ms / count};

        // Distinct device-resident inputs, same reasoning as the rgb_to_gray
        // batch above — with 1 MB grayscale images the L2-replay flattery
        // would otherwise be at its worst (input + output fit entirely).
        std::vector<cig::GpuBuffer<std::uint8_t>> d_ins;
        d_ins.reserve(count);
        for (const auto& im : imgs) {
            d_ins.emplace_back(n);
            d_ins.back().copy_from_host(im.data.data(), n);
        }
        cig::GpuBuffer<std::uint8_t> d_in(n), d_out(n);
        const Stats gk = time_gpu_kernel_ms([&] {
            for (int i = 0; i < count; ++i) cig::convolve2d_tiled(d_ins[i], d_out, w, h, k);
        });
        const Stats gw = time_host_ms([&] {
            for (const auto& im : imgs) {
                d_in.copy_from_host(im.data.data(), n);
                cig::convolve2d_tiled(d_in, d_out, w, h, k);
                d_out.copy_to_host(out.data(), n);
            }
        });
        r.gpu_kernel = {gk.median_ms / count, gk.stddev_ms / count};
        r.gpu_wall = {gw.median_ms / count, gw.stddev_ms / count};
        add_row(std::move(r));
    }
}

std::string sanitize(const char* name) {
    std::string s(name);
    for (char& c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (!ok) c = '_';
    }
    return s;
}

}  // namespace

int main() try {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // memoryClockRate left cudaDeviceProp in CUDA 13; query the attribute
    // there instead (and shrug if the platform no longer reports it).
    int mem_clock_khz = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    if (cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev) != cudaSuccess) {
        mem_clock_khz = 0;
    }
#else
    mem_clock_khz = prop.memoryClockRate;
#endif

    std::printf("device: %s | SMs: %d | mem clock: %.0f MHz | sm_%d%d | OpenMP threads: %d\n\n",
                prop.name, prop.multiProcessorCount, mem_clock_khz / 1000.0, prop.major,
                prop.minor, omp_get_max_threads());

    const std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    for (int s : sizes) bench_rgb_to_gray(s, s);
    for (int s : sizes) bench_gaussian(s, s);
    for (int s : sizes) bench_equalize(s, s);
    bench_resize();
    bench_batch();

    std::filesystem::create_directories("results");
    const std::string path = "results/results_" + sanitize(prop.name) + ".csv";
    std::ofstream csv(path);
    csv << "# device: " << prop.name << "\n";
    csv << "# sms: " << prop.multiProcessorCount << "\n";
    csv << "# mem_clock_mhz: " << mem_clock_khz / 1000 << "\n";
    csv << "# omp_threads: " << omp_get_max_threads() << "\n";
    csv << "# methodology: 10 warmup + 100 timed runs, medians; gpu_kernel_ms via cudaEvent "
           "(kernel only), gpu_wall_ms via std::chrono incl. H2D+D2H; *_batch100 rows are "
           "per-image over a 100-image batch (CPU batch paths: 1 warmup + 5 timed runs)\n";
    csv << "op,resolution,cpu_1t_ms,omp_ms,gpu_kernel_ms,gpu_wall_ms,speedup_vs_1t,"
           "speedup_vs_omp\n";
    char buf[512];
    for (const Row& r : g_rows) {
        std::snprintf(buf, sizeof(buf), "%s,%s,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f\n", r.op.c_str(),
                      r.resolution.c_str(), r.cpu_1t.median_ms, r.omp.median_ms,
                      r.gpu_kernel.median_ms, r.gpu_wall.median_ms,
                      r.cpu_1t.median_ms / r.gpu_kernel.median_ms,
                      r.omp.median_ms / r.gpu_kernel.median_ms);
        csv << buf;
    }
    csv.close();
    std::printf("\nwrote %s\n", path.c_str());
    return 0;
} catch (const std::exception& e) {
    // CUDA_CHECK and the filesystem/CSV writes throw; fail with the message
    // rather than a std::terminate backtrace.
    std::fprintf(stderr, "bench_gpu failed: %s\n", e.what());
    return 1;
}
