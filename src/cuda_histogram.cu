// CUDA-accelerated weighted support histograms for EVERY FP-tree level.
// The host performs FP-tree structural construction / recursive traversal.
// The GPU evaluates weighted item supports at the root AND within each
// recursively generated conditional pattern base.
#include "cuda_histogram.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + operation + ": " +
                                 cudaGetErrorString(result));
}

// Global-memory histogram: one thread per weighted item occurrence.
// atomicAdd on 64-bit integer weights requires an appropriate CUDA GPU.
__global__ void weighted_support_kernel(const int* item_ids,
                                        const unsigned long long* weights,
                                        std::size_t n,
                                        unsigned long long* counts) {
    std::size_t index = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        atomicAdd(&counts[item_ids[index]], weights[index]);
    }
}

std::size_t grow_capacity(std::size_t current, std::size_t required) {
    if (required <= current) return current;
    if (current > std::numeric_limits<std::size_t>::max() / 2) return required;
    return required < current * 2 ? current * 2 : required;
}
}

struct CudaHistogram::Impl {
    int* d_items = nullptr;
    unsigned long long* d_weights = nullptr;
    unsigned long long* d_counts = nullptr;
    std::size_t item_capacity = 0;
    std::size_t count_capacity = 0;
    CudaHistogramStats measurements;

    ~Impl() {
        if (d_items) cudaFree(d_items);
        if (d_weights) cudaFree(d_weights);
        if (d_counts) cudaFree(d_counts);
    }

    void reserve(std::size_t items, std::size_t counts) {
        if (items > item_capacity) {
            // Preserve previously allocated buffers on allocation failure.
            const std::size_t capacity = grow_capacity(item_capacity, items);
            if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(unsigned long long))
                throw std::length_error("Too many GPU histogram entries");
            int* new_items = nullptr;
            unsigned long long* new_weights = nullptr;
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&new_items), capacity * sizeof(int)), "alloc items");
            try {
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&new_weights),
                                      capacity * sizeof(unsigned long long)), "alloc weights");
            } catch (...) {
                cudaFree(new_items);
                throw;
            }
            if (d_items) cudaFree(d_items);
            if (d_weights) cudaFree(d_weights);
            d_items = new_items;
            d_weights = new_weights;
            item_capacity = capacity;
        }
        if (counts > count_capacity) {
            const std::size_t capacity = grow_capacity(count_capacity, counts);
            if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(unsigned long long))
                throw std::length_error("Too many GPU histogram buckets");
            unsigned long long* new_counts = nullptr;
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&new_counts),
                                  capacity * sizeof(unsigned long long)), "alloc counts");
            if (d_counts) cudaFree(d_counts);
            d_counts = new_counts;
            count_capacity = capacity;
        }
        const std::uint64_t reserved =
            item_capacity * (sizeof(int) + sizeof(unsigned long long)) +
            count_capacity * sizeof(unsigned long long);
        if (reserved > measurements.max_reserved_bytes)
            measurements.max_reserved_bytes = reserved;
    }
};

CudaHistogram::CudaHistogram(): impl_(new Impl) {
    try {
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "device discovery");
        if (devices == 0) throw std::runtime_error("No CUDA device available");
        cudaDeviceProp props{};
        cuda_check(cudaGetDeviceProperties(&props, 0), "device properties");
        // Native 64-bit integer atomicAdd on global memory needs CC >= 3.5.
        if (props.major < 3 || (props.major == 3 && props.minor < 5))
            throw std::runtime_error("CUDA GPU compute capability 3.5 or higher is required");
        cuda_check(cudaSetDevice(0), "select device 0");
        impl_->measurements.device_name = props.name;
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

CudaHistogram::~CudaHistogram() { delete impl_; }

std::vector<std::uint64_t> CudaHistogram::count(
    const std::vector<int>& ids, const std::vector<std::uint64_t>& weights,
    std::size_t num_items) {
    if (ids.size() != weights.size())
        throw std::invalid_argument("CUDA histogram input lengths do not match");
    if (num_items == 0) {
        if (!ids.empty()) throw std::invalid_argument("Zero histogram buckets for nonempty input");
        return {};
    }
    std::vector<std::uint64_t> host_counts(num_items, 0);
    if (ids.empty()) return host_counts;
    if (ids.size() > std::numeric_limits<std::size_t>::max() / sizeof(unsigned long long))
        throw std::length_error("CUDA weighted input size overflow");
    // Source parser and FP-tree construction guarantee valid item IDs. Check
    // anyway to avoid invalid GPU memory accesses if this API is reused.
    for (int id : ids)
        if (id < 0 || static_cast<std::size_t>(id) >= num_items)
            throw std::invalid_argument("Item ID outside GPU histogram bounds");
    impl_->reserve(ids.size(), num_items);
    cuda_check(cudaMemcpy(impl_->d_items, ids.data(), ids.size() * sizeof(int),
                          cudaMemcpyHostToDevice), "copy items H2D");
    cuda_check(cudaMemcpy(impl_->d_weights, weights.data(),
                          weights.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice), "copy weights H2D");
    cuda_check(cudaMemset(impl_->d_counts, 0, num_items * sizeof(std::uint64_t)), "clear supports");
    constexpr unsigned int THREADS = 256;
    const std::size_t blocks = (ids.size() + THREADS - 1) / THREADS;
    if (blocks > 2147483647ULL)
        throw std::length_error("CUDA grid dimensions exceed max grid X");
    weighted_support_kernel<<<static_cast<unsigned int>(blocks), THREADS>>>(
        impl_->d_items, impl_->d_weights, ids.size(), impl_->d_counts);
    cuda_check(cudaGetLastError(), "kernel launch");
    // The following synchronous D2H copy waits for the kernel's completion
    // and reports asynchronous device failures through its return code.
    cuda_check(cudaMemcpy(host_counts.data(), impl_->d_counts,
                          num_items * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost), "copy counts D2H");
    ++impl_->measurements.kernel_calls;
    impl_->measurements.entries_processed += ids.size();
    return host_counts;
}

CudaHistogramStats CudaHistogram::stats() const { return impl_->measurements; }
