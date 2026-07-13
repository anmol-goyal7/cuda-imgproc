# Running the GPU suite on a free Colab T4

The library's GPU tests and benchmarks need an NVIDIA GPU; a free Google
Colab T4 (sm_75) is the reference target. Total hands-on time: ~2 minutes,
plus 5–10 minutes of runtime.

## Procedure

1. **Open a T4 notebook.** Go to <https://colab.research.google.com>, create a
   new notebook, then *Runtime → Change runtime type → Hardware accelerator:
   **T4 GPU** → Save*.

2. **Paste this into a cell and run it:**

   ```
   !bash <(curl -fsSL https://raw.githubusercontent.com/anmol-goyal7/cuda-imgproc/main/colab/run_t4.sh)
   ```

   The script checks the GPU, detects its architecture (sm_75 on a T4; prefix
   the command with `ARCH=sm_xx ` to override), clones this repo, runs
   `make test-gpu` (aborting if any op disagrees with the CPU reference —
   **check that the printed table is all PASS**), then `make bench-gpu`, and
   finally prints the CSV it wrote.

3. **Download the CSV.** Open the file browser (folder icon in the left
   sidebar), navigate to `cuda-imgproc/results/`, and download
   `results_Tesla_T4.csv` (the name embeds whatever GPU actually ran).

4. **Publish the numbers.** In your local clone of this repo:

   ```sh
   cp ~/Downloads/results_Tesla_T4.csv results/
   make readme-results     # renders the CSV into README.md + results/summary.md
   git add README.md results/ && git commit -m "results: t4 benchmark run" && git push
   ```

## Notes

* Colab pairs the T4 with ~2 vCPUs, so the CPU baseline columns are honest
  but slow — most of the 5–10 minutes is the CPU, not the GPU.
* A different GPU runtime (L4, A100) works too: the script reads the compute
  capability off `nvidia-smi` and builds native SASS for whatever device it
  finds (override with `ARCH=sm_xx` in front of the command), and the
  CSV/README will be labeled with the actual device name.
