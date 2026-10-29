#ifndef FP_GROWTH_CUDA_HISTOGRAM_HPP
#define FP_GROWTH_CUDA_HISTOGRAM_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// GPU implementation is in cuda_histogram.cu. This interface deliberately uses
// only standard C++ types, allowing the host algorithm to compile with g++.
struct CudaHistogramStats {
    std::string device_name;
    std::uint64_t kernel_calls = 0;
    std::uint64_t entries_processed = 0;
    std::uint64_t max_reserved_bytes = 0; // allocation high-water for the CUDA buffers
};

class CudaHistogram {
public:
    CudaHistogram();
    ~CudaHistogram();
    CudaHistogram(const CudaHistogram&) = delete;
    CudaHistogram& operator=(const CudaHistogram&) = delete;

    // Each flattened item occurrence contributes its associated path weight.
    // All item IDs are in [0, num_items) and counts are unsigned 64-bit.
    std::vector<std::uint64_t> count(const std::vector<int>& item_ids,
                                      const std::vector<std::uint64_t>& weights,
                                      std::size_t num_items);
    CudaHistogramStats stats() const;
private:
    struct Impl;
    Impl* impl_;
};

#endif
