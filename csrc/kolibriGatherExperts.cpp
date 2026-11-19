#include "kolibriGatherExperts.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace kolibri::host {
namespace {

constexpr size_t kMaxSlabs = 16;
constexpr size_t kSmallTensorBytesThreshold = 4096;
constexpr size_t kOmpParallelBytesThreshold = 512 * 1024;
constexpr size_t kAlignBytes = 64;

struct CpuTopologyInfo {
    cpu_set_t pcore_mask;
    int64_t pcore_count = 1;
    bool has_hetero_cores = false;
};

CpuTopologyInfo const& get_cpu_topology() {
    static CpuTopologyInfo const info = []() {
        CpuTopologyInfo out;
        CPU_ZERO(&out.pcore_mask);

        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
            out.pcore_count = std::max<int64_t>(1, omp_get_num_procs());
            for (int c = 0; c < out.pcore_count && c < CPU_SETSIZE; ++c) {
                CPU_SET(c, &out.pcore_mask);
            }
            return out;
        }

        struct CpuFreqEntry {
            int cpu;
            int64_t max_khz;
        };
        std::vector<CpuFreqEntry> entries;
        int64_t global_max_khz = 0;

        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) {
                continue;
            }
            int64_t khz = 0;
            std::string path =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq";
            std::ifstream f(path);
            if (f.is_open()) {
                f >> khz;
            }
            entries.push_back({cpu, khz});
            global_max_khz = std::max(global_max_khz, khz);
        }

        if (entries.empty()) {
            out.pcore_count = 1;
            CPU_SET(0, &out.pcore_mask);
            return out;
        }

        if (global_max_khz <= 0) {
            for (auto const& e : entries) {
                CPU_SET(e.cpu, &out.pcore_mask);
            }
            out.pcore_count = static_cast<int64_t>(entries.size());
            out.has_hetero_cores = false;
            return out;
        }

        // Distinguish P-cores from E-cores (e.g., 4.9-5.0 GHz vs 4.6 GHz -> 95% cutoff).
        int64_t const pcore_cutoff = (global_max_khz * 95) / 100;
        int64_t p_cnt = 0;
        int64_t e_cnt = 0;
        for (auto const& e : entries) {
            if (e.max_khz >= pcore_cutoff) {
                CPU_SET(e.cpu, &out.pcore_mask);
                ++p_cnt;
            } else {
                ++e_cnt;
            }
        }

        if (p_cnt == 0) {
            for (auto const& e : entries) {
                CPU_SET(e.cpu, &out.pcore_mask);
            }
            out.pcore_count = static_cast<int64_t>(entries.size());
            out.has_hetero_cores = false;
        } else {
            out.pcore_count = p_cnt;
            out.has_hetero_cores = (e_cnt > 0);
        }
        return out;
    }();
    return info;
}

struct TensorSliceDesc {
    uint8_t const* src_ptr;
    uint8_t* dst_ptr;
    size_t row_bytes;
    size_t prefix_start;
    size_t prefix_end;
};

}  // namespace

int64_t detected_pcore_count() {
    return get_cpu_topology().pcore_count;
}

void gather_experts_omp(
    std::vector<torch::Tensor> const& src_tensors,
    std::vector<torch::Tensor> const& dst_tensors,
    torch::Tensor const& expert_ids,
    int64_t num_threads
) {
    TORCH_CHECK(src_tensors.size() == dst_tensors.size(), "src_tensors and dst_tensors must match in length");
    TORCH_CHECK(src_tensors.size() <= kMaxSlabs, "Too many tensors in gather_experts_omp");
    TORCH_CHECK(expert_ids.device().is_cpu(), "expert_ids must be on CPU");
    TORCH_CHECK(expert_ids.scalar_type() == torch::kLong, "expert_ids must be int64");
    TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");

    int64_t const m = expert_ids.numel();
    if (m == 0 || src_tensors.empty()) {
        return;
    }

    int64_t const* idx_ptr = expert_ids.data_ptr<int64_t>();
    size_t const num_slabs = src_tensors.size();

    std::array<TensorSliceDesc, kMaxSlabs> large_slabs{};
    std::array<TensorSliceDesc, kMaxSlabs> small_slabs{};
    size_t num_large = 0;
    size_t num_small = 0;
    size_t large_bytes_per_expert = 0;
    int64_t max_experts = -1;

    for (size_t k = 0; k < num_slabs; ++k) {
        auto const& src = src_tensors[k];
        auto const& dst = dst_tensors[k];
        TORCH_CHECK(src.device().is_cpu() && dst.device().is_cpu(), "src/dst tensors must be on CPU");
        TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst tensors must be contiguous");
        TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src/dst dtypes must match");
        TORCH_CHECK(src.dim() >= 1 && dst.dim() >= 1, "src/dst tensors must have >= 1 dimension");
        TORCH_CHECK(dst.size(0) >= m, "dst tensor has fewer rows than expert_ids");

        int64_t const num_e = src.size(0);
        if (max_experts < 0) {
            max_experts = num_e;
        } else {
            TORCH_CHECK(max_experts == num_e, "All src tensors must share dim 0 size");
        }

        size_t const src_row_bytes = static_cast<size_t>(src.stride(0)) * src.element_size();
        size_t const dst_row_bytes = static_cast<size_t>(dst.stride(0)) * dst.element_size();
        TORCH_CHECK(src_row_bytes == dst_row_bytes, "src/dst row byte sizes must match");

        TensorSliceDesc desc{
            static_cast<uint8_t const*>(src.data_ptr()),
            static_cast<uint8_t*>(dst.data_ptr()),
            src_row_bytes,
            0,
            0,
        };
        if (src_row_bytes < kSmallTensorBytesThreshold) {
            small_slabs[num_small++] = desc;
        } else {
            desc.prefix_start = large_bytes_per_expert;
            large_bytes_per_expert += src_row_bytes;
            desc.prefix_end = large_bytes_per_expert;
            large_slabs[num_large++] = desc;
        }
    }

    for (int64_t e = 0; e < m; ++e) {
        int64_t const eid = idx_ptr[e];
        TORCH_CHECK(eid >= 0 && eid < max_experts, "Expert index out of range");
    }

    pybind11::gil_scoped_release release_gil;

    // Copy tiny tensors (e.g., scalar global scales) without OpenMP overhead.
    for (size_t k = 0; k < num_small; ++k) {
        auto const& s = small_slabs[k];
        size_t const rb = s.row_bytes;
        for (int64_t e = 0; e < m; ++e) {
            std::memcpy(s.dst_ptr + e * rb, s.src_ptr + idx_ptr[e] * rb, rb);
        }
    }

    if (num_large == 0) {
        return;
    }

    CpuTopologyInfo const& topo = get_cpu_topology();
    int64_t const requested = num_threads > 0 ? num_threads : topo.pcore_count;
    int const eff_threads = static_cast<int>(
        std::clamp<int64_t>(requested, 1, std::max<int64_t>(1, topo.pcore_count))
    );
    size_t const total_bytes = static_cast<size_t>(m) * large_bytes_per_expert;

    #pragma omp parallel num_threads(eff_threads) if (total_bytes >= kOmpParallelBytesThreshold && eff_threads > 1)
    {
        int const tid = omp_get_thread_num();
        int const nt = omp_get_num_threads();

        cpu_set_t prev_mask;
        bool restore_affinity = false;
        if (topo.has_hetero_cores) {
            pthread_t const self = pthread_self();
            if (pthread_getaffinity_np(self, sizeof(prev_mask), &prev_mask) == 0 &&
                !CPU_EQUAL(&prev_mask, &topo.pcore_mask)) {
                if (pthread_setaffinity_np(self, sizeof(topo.pcore_mask), &topo.pcore_mask) == 0) {
                    restore_affinity = true;
                }
            }
        }

        size_t t_begin = (total_bytes * static_cast<size_t>(tid)) / static_cast<size_t>(nt);
        size_t t_end = (total_bytes * static_cast<size_t>(tid + 1)) / static_cast<size_t>(nt);
        if (tid > 0) {
            t_begin = (t_begin + kAlignBytes - 1) & ~(kAlignBytes - 1);
        }
        if (tid + 1 < nt) {
            t_end = (t_end + kAlignBytes - 1) & ~(kAlignBytes - 1);
        }
        t_begin = std::min(t_begin, total_bytes);
        t_end = std::min(t_end, total_bytes);

        size_t pos = t_begin;
        while (pos < t_end) {
            size_t const e = pos / large_bytes_per_expert;
            size_t const rem = pos % large_bytes_per_expert;
            int64_t const eid = idx_ptr[e];

            size_t k = 0;
            while (k + 1 < num_large && rem >= large_slabs[k].prefix_end) {
                ++k;
            }
            auto const& s = large_slabs[k];
            size_t const in_row_off = rem - s.prefix_start;
            size_t const avail = s.prefix_end - rem;
            size_t const chunk = std::min(avail, t_end - pos);

            std::memcpy(
                s.dst_ptr + e * s.row_bytes + in_row_off,
                s.src_ptr + static_cast<size_t>(eid) * s.row_bytes + in_row_off,
                chunk
            );
            pos += chunk;
        }

        if (restore_affinity) {
            pthread_setaffinity_np(pthread_self(), sizeof(prev_mask), &prev_mask);
        }
    }
}

}  // namespace kolibri::host
