CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
NVCCFLAGS ?= -std=c++17 -O2

all: fp_growth

fp_growth: src/fp_growth.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Real CUDA miner: GPU counts weighted item supports at every FP-tree level,
# not just the root. CPU constructs the tree topology and controls recursion.
cuda: fp_growth_cuda
fp_growth_cuda: src/fp_growth.cpp src/cuda_histogram.cu src/cuda_histogram.hpp
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -x cu src/fp_growth.cpp src/cuda_histogram.cu -o $@

# Host-only emulator helps validate the CUDA-specific orchestration without
# nvcc/GPU. It is not GPU testing; the GPU kernel is not compiled here.
cuda_sim: fp_growth_cuda_sim
fp_growth_cuda_sim: src/fp_growth.cpp tests/cuda_histogram_emulator.cpp src/cuda_histogram.hpp
	$(CXX) $(CXXFLAGS) -DUSE_CUDA src/fp_growth.cpp tests/cuda_histogram_emulator.cpp -o $@

test: fp_growth
	python3 -m unittest discover -s tests -v

verify-cuda: fp_growth fp_growth_cuda
	python3 scripts/verify_cuda.py

bench-cuda: fp_growth fp_growth_cuda
	python3 scripts/benchmark_cuda.py --repeats 3

clean:
	rm -f fp_growth fp_growth_cuda fp_growth_cuda_sim

.PHONY: all cuda cuda_sim test verify-cuda bench-cuda clean
