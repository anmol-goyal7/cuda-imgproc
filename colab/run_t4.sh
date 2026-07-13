#!/bin/bash
# cuda-imgproc — Colab T4 test + benchmark runner.
#
# Paste into a single Colab cell (GPU runtime!) as:
#
#     !bash <(curl -fsSL https://raw.githubusercontent.com/anmol-goyal7/cuda-imgproc/main/colab/run_t4.sh)
#
# or clone first and run `!bash cuda-imgproc/colab/run_t4.sh`. Roughly 5-10
# minutes end to end (the CPU baselines at 4096x4096 dominate; Colab gives
# the GPU runtime only ~2 vCPUs).
set -euo pipefail

echo "== 1/5 GPU check =========================================================="
if ! nvidia-smi; then
    echo ""
    echo "ERROR: no GPU visible. In Colab: Runtime -> Change runtime type ->"
    echo "Hardware accelerator: T4 GPU, then run this cell again."
    exit 1
fi
nvcc --version | tail -1

# Build arch: honour an ARCH from the environment (`!ARCH=sm_90 bash <(...)`),
# else read the compute capability off the device (T4 -> "7.5" -> sm_75), so
# any GPU runtime gets native SASS. Falls back to the Makefile's default
# fatbin if detection fails.
if [ -z "${ARCH:-}" ]; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    case "$cc" in
        '' | *[!0-9]*) ARCH="" ;;
        *) ARCH="sm_${cc}" ;;
    esac
fi
echo "build arch: ${ARCH:-Makefile default fatbin (sm_75 + sm_86 + PTX)}"

echo ""
echo "== 2/5 clone =============================================================="
rm -rf cuda-imgproc
git clone --depth 1 https://github.com/anmol-goyal7/cuda-imgproc.git
cd cuda-imgproc

echo ""
echo "== 3/5 correctness: GPU vs CPU reference =================================="
# Aborts this script (set -e) if any op disagrees with the CPU reference.
make test-gpu ${ARCH:+ARCH=$ARCH}

echo ""
echo "== 4/5 benchmark (writes results/*.csv) ==================================="
make bench-gpu ${ARCH:+ARCH=$ARCH}

echo ""
echo "== 5/5 results ============================================================"
for f in results/results_*.csv; do
    echo "--- $f ---"
    cat "$f"
done

echo ""
echo "DONE. Download the CSV via the Colab file browser (folder icon, left"
echo "sidebar): $(pwd)/results/$(ls results | grep '^results_.*\.csv$' | head -1)"
echo "Then, in your local clone:"
echo "    cp ~/Downloads/results_*.csv results/"
echo "    make readme-results"
