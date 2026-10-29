# CUDA-Accelerated FP-Growth — C++17 / NVIDIA CUDA

**Project:** frequent itemset mining by FP-Growth, with GPU-accelerated weighted support counting at **every FP-tree level**, including root and all recursively built conditional trees. This is a real CPU/CUDA *hybrid*, not a GPU-resident pointer-based FP-tree: the CPU constructs the trees, builds conditional pattern bases, recurses, and writes the results; the GPU runs the parallel counting kernel throughout mining.

**Current verification:** C++ CPU miner passed 7 automated test groups (including 100 random datasets checked against a brute-force oracle). The CUDA-specific *host code path* passed 34 exact CPU-versus-emulated-backend comparisons, including random cases. This preparation environment does **not** have an NVIDIA GPU or the CUDA compiler `nvcc`. **The actual `.cu` GPU kernel has not yet been compiled, run, or benchmarked. Run the lab commands below before submitting your final report.**

## Tomorrow in the lab: do this first

Open a Linux terminal in the folder containing this README:

```bash
nvcc --version
nvidia-smi    # works on many desktop GPUs; may be absent on Jetson
make
make cuda
./fp_growth_cuda data/toy_transactions.txt 3 --output results/toy_cuda.csv --summary results/toy_cuda.json
python3 scripts/verify_cuda.py --random-cases 20
python3 scripts/benchmark_cuda.py --repeats 3
```

**Quick expected values:** toy dataset: 5 transactions, 8 frequent itemsets, `bread;milk` support = 3, `beer;diaper` support = 3. The output JSON's `backend` should say `cuda-hybrid-all-level-counting`; `cuda_device` should name a real NVIDIA GPU, **not** `HOST SIMULATOR`; and `support_conditional_calls` should be >0. The toy case needs 1 root kernel plus 4 conditional histogram kernels, so `cuda_kernel_calls` should be **5**.

If `nvcc` isn't on the PATH, try `/usr/local/cuda/bin/nvcc --version` and then `make cuda NVCC=/usr/local/cuda/bin/nvcc` (if installed). If running on **Jetson Orin Nano**, you may use `make cuda NVCCFLAGS='-std=c++17 -O2 -arch=sm_87'` when its installed CUDA version supports this architecture. Don't hard-code that arch for other GPUs. NVIDIA lists Jetson Orin Nano with compute capability 8.7: https://developer.nvidia.com/cuda/gpus

If the GPU test fails, copy the **full** terminal error and `nvcc --version`; do not replace real GPU verification with the emulator. `make cuda` needs the CUDA toolkit and an actual NVIDIA GPU at runtime.

## Source code and architecture

| File | Role |
|---|---|
| `src/fp_growth.cpp` | Actual FP-Growth recursion, weighted conditional pattern bases, FP-tree construction, CLI and output; builds as CPU reference or CUDA host driver |
| `src/cuda_histogram.hpp` | Shared CUDA histogram interface and GPU measurement statistics |
| `src/cuda_histogram.cu` | `weighted_support_kernel<<<...>>>` weighted integer `atomicAdd`, device memory allocation/reuse, copies in/out, GPU identification |
| `tests/cuda_histogram_emulator.cpp` | Host-only mock for validating the CUDA calling path (NOT a GPU implementation) |
| `tests/test_fp_growth.py` | Independent CPU correctness tests |
| `scripts/verify_cuda.py` | Real GPU versus CPU **complete itemset and support** comparison; random exhaustive oracle |
| `scripts/benchmark_cuda.py` | Fresh matched CPU/GPU runtime and peak RAM/device-buffer statistics, CSV and JSON |
| `scripts/get_uci.py` | Downloads and converts additional original UCI benchmarks when network is available |
| `report/FP_Growth_CUDA_Project_Report.pdf` | Printable research report: algorithm, tests, real CPU results, CUDA design, GPU validation plan |

### Where CUDA is executed

1. **Root tree:** parse/normalize transactions on CPU, flatten item occurrences to `(item_id, weight=1)` on host, GPU computes all singleton supports.
2. **Build FP-tree on CPU:** filter infrequent items, order by support, compress shared prefixes and build header node pointers.
3. **Recursive conditional mining:** for each suffix item, follow header node links and collect ancestor paths, weighted by the header node's count.
4. **Every conditional tree:** flatten `(item_id, path_weight)` on the host, send to device, clear output, launch a parallel 64-bit atomic support histogram, copy counts back, filter infrequent items and construct the next FP-tree.
5. Recursively enumerate and stream **all** qualifying frequent itemsets and their exact support values.

The GPU is not only invoked during root preprocessing. You can confirm this in the JSON stats: `support_root_calls`, `support_conditional_calls`, `cuda_kernel_calls`, `cuda_entries_processed`, and `cuda_buffer_high_water_bytes`.

**Performance caveat:** numerous small conditional trees require many kernel launches and PCIe/unified-memory transfers. Hybrid CUDA execution can be slower than the CPU baseline on these instances. Measure both rather than claiming speedup beforehand.

## Normal usage

```bash
# CPU reference (g++):
make
./fp_growth data/toy_transactions.txt 3 --output results/cpu_toy.csv --summary results/cpu_toy.json

# CUDA GPU miner (nvcc):
make cuda
./fp_growth_cuda data/toy_transactions.txt 3 --output results/toy_cuda.csv --summary results/toy_cuda.json

# Support can be absolute, fraction, or percentage:
./fp_growth_cuda data/uci_wdbc_569.txt 50% --no-output --summary results/wdbc_cuda.json
./fp_growth_cuda data/uci_digits_1797.txt 0.8 --no-output --summary results/digits_cuda.json

# CPU tests + strict GPU equality + CPU/GPU benchmark:
make test
make verify-cuda
make bench-cuda
```

**Output formats:** `patterns.csv` contains `support,itemset` with sorted semicolon-separated item names. JSON summarizes transaction count, support threshold, itemset count, maximum length, FP-tree counts, elapsed time, process peak resident RAM and (on CUDA) GPU statistics. Each input line is one whitespace-delimited transaction; repeated items per transaction count only once. Comments begin with `#`; blank lines are ignored. `--no-output` avoids writing potentially huge result CSVs but still mines the full pattern set.

## Dataset files and provenance

The ZIP includes the following datasets so the project can run offline tomorrow:

| File | Transactions | Source / conversion |
|---|---:|---|
| `toy_transactions.txt` | 5 | Small illustrated market baskets for exact hand validation |
| `uci_digits_1797.txt` | 1,797 | UCI-origin handwritten-digit data via scikit-learn, pixel-nonzero item tokens; no class label |
| `uci_wdbc_569.txt` | 569 | UCI-origin Wisconsin breast-cancer diagnostic data via scikit-learn; median-bin feature tokens, no class label |
| `uci_digits_repeated_53910.txt` | 53,910 | **Exactly 30 copies** of digits, strictly a scaling experiment |
| `synthetic_retail_50000.txt` | 50,000 | Seeded synthetic data, **not UCI** |

The assignment also requests examples from the UCI archive: `python3 scripts/get_uci.py all` attempts original Mushroom, Chess, Connect-4 downloads. **These original files were not obtained/benchmarked here**, and the downloader's network behavior has not been lab verified. Existing offline inputs include actual UCI-origin copies plus clearly labeled synthetic/replicated data.

## Benchmark output and report finalization

The previous CPU-only benchmark is saved in `results/benchmark_results.csv` and `.json` with real 5-run measurements from the preparation machine. **Do not use those CPU numbers as lab GPU measurements.** On lab hardware, `python3 scripts/benchmark_cuda.py --repeats 3` creates `results/cuda_benchmark_results.csv` and `.json` containing CPU and CUDA medians, process RAM, GPU buffer allocation, device name and kernel-call counts. Open the CSV and use its measured values to update the PDF before hardcopy/ZIP submission. The report generator is `python3 scripts/generate_report.py` and needs `reportlab`, `matplotlib` for plots; its CPU benchmarks are loaded from the supplied JSON. Its lab-performance section is intentionally marked as **pending** until the real measurements are available.

The assignment screenshot specifies a hardcopy report and emailed source/report ZIP to `alex.doboli@stonybrook.edu`; confirm the actual deadline with the instructor given the date confusion. For bonus credit, describe the implementation accurately as **CUDA-accelerated hybrid FP-Growth**. Do not claim it is fully GPU-resident or faster until verified.

## Optional offline simulator (NOT GPU testing)

If you want to verify the new CUDA calling path on a PC without CUDA:

```bash
make cuda_sim
python3 scripts/verify_cuda.py --cuda-bin fp_growth_cuda_sim --random-cases 30
```

The mock `fp_growth_cuda_sim` uses a CPU histogram via the same interface, so its success **does not prove** the NVIDIA `.cu` kernel compiles or runs. The 34 comparisons were already completed; see `results/host_sim_validation.txt`.

## References

- Han, Pei, & Yin (2000), *Mining Frequent Patterns without Candidate Generation*. https://doi.org/10.1145/342009.335372
- NVIDIA CUDA C++ Programming Guide (atomic functions): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA Jetson Orin CUDA setup: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_cuda.html
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
