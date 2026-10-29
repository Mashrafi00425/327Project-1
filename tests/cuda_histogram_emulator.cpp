// Testing ONLY: exercises the CUDA-enabled host call path with a CPU histogram.
// NOT an alternative to testing CUDA hardware / compiling cuda_histogram.cu.
#include "../src/cuda_histogram.hpp"
#include <stdexcept>
struct CudaHistogram::Impl { CudaHistogramStats s; };
CudaHistogram::CudaHistogram(): impl_(new Impl) { impl_->s.device_name = "HOST SIMULATOR (NOT CUDA)"; }
CudaHistogram::~CudaHistogram() { delete impl_; }
std::vector<std::uint64_t> CudaHistogram::count(const std::vector<int>& ids,
        const std::vector<std::uint64_t>& weights, std::size_t n) {
    if (ids.size() != weights.size()) throw std::invalid_argument("length mismatch");
    std::vector<std::uint64_t> counts(n, 0);
    for (std::size_t i=0;i<ids.size();++i) {
        if (ids[i] < 0 || (std::size_t)ids[i] >= n) throw std::invalid_argument("bad ID");
        counts[ids[i]] += weights[i];
    }
    if (!ids.empty()) { ++impl_->s.kernel_calls; impl_->s.entries_processed += ids.size(); }
    return counts;
}
CudaHistogramStats CudaHistogram::stats() const { return impl_->s; }
