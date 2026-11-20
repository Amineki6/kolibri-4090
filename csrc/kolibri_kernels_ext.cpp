#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kolibriCacheOps.h"
#include "kolibriGatherExperts.h"
#include "kolibriRouterKernel.h"
#include "tensorrt_llm/kernels/moe/post_norm/kolibriPostMoeNormKernel.h"
#include "tensorrt_llm/kernels/moe/sandwich_norm/kolibriSandwichNormKernel.h"

namespace th = torch;

std::tuple<th::Tensor, th::Tensor> fused_sandwich_norm(
    th::Tensor attn_out,
    th::Tensor residual_in,
    th::Tensor gamma1,
    th::Tensor gamma2,
    double eps
) {
    TORCH_CHECK(attn_out.is_cuda() && residual_in.is_cuda(), "attn_out and residual_in must be CUDA tensors");
    TORCH_CHECK(gamma1.is_cuda() && gamma2.is_cuda(), "gamma1 and gamma2 must be CUDA tensors");
    TORCH_CHECK(
        attn_out.is_contiguous() && residual_in.is_contiguous() && gamma1.is_contiguous() && gamma2.is_contiguous(),
        "fused_sandwich_norm expects contiguous inputs"
    );
    TORCH_CHECK(attn_out.sizes() == residual_in.sizes(), "attn_out and residual_in shapes must match");
    TORCH_CHECK(
        attn_out.scalar_type() == residual_in.scalar_type()
            && attn_out.scalar_type() == gamma1.scalar_type()
            && attn_out.scalar_type() == gamma2.scalar_type(),
        "fused_sandwich_norm inputs must share dtype"
    );

    int const hidden_dim = attn_out.size(-1);
    TORCH_CHECK(hidden_dim == 2560, "fused_sandwich_norm requires hidden_dim == 2560");
    TORCH_CHECK(gamma1.numel() == hidden_dim && gamma2.numel() == hidden_dim, "gamma sizes must match hidden_dim");
    int const tokens = attn_out.numel() / hidden_dim;

    c10::cuda::CUDAGuard device_guard(attn_out.device());
    auto stream = at::cuda::getCurrentCUDAStream(attn_out.device().index()).stream();

    auto updated_residual = th::empty_like(residual_in);
    auto moe_in = th::empty_like(attn_out);
    if (tokens == 0) {
        return {updated_residual, moe_in};
    }

    auto const dtype = attn_out.scalar_type();
    if (dtype == th::kBFloat16) {
        tensorrt_llm::kernels::invokeFusedSandwichNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(attn_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(residual_in.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma1.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma2.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(updated_residual.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(moe_in.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    } else if (dtype == th::kHalf) {
        tensorrt_llm::kernels::invokeFusedSandwichNorm<half>(
            reinterpret_cast<half const*>(attn_out.data_ptr()),
            reinterpret_cast<half const*>(residual_in.data_ptr()),
            reinterpret_cast<half const*>(gamma1.data_ptr()),
            reinterpret_cast<half const*>(gamma2.data_ptr()),
            reinterpret_cast<half*>(updated_residual.data_ptr()),
            reinterpret_cast<half*>(moe_in.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    } else if (dtype == th::kFloat) {
        tensorrt_llm::kernels::invokeFusedSandwichNorm<float>(
            reinterpret_cast<float const*>(attn_out.data_ptr()),
            reinterpret_cast<float const*>(residual_in.data_ptr()),
            reinterpret_cast<float const*>(gamma1.data_ptr()),
            reinterpret_cast<float const*>(gamma2.data_ptr()),
            reinterpret_cast<float*>(updated_residual.data_ptr()),
            reinterpret_cast<float*>(moe_in.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused_sandwich_norm (expected bf16, fp16, or fp32)");
    }

    return {updated_residual, moe_in};
}

th::Tensor fused_add_post_moe_norm(
    th::Tensor routed_out,
    th::Tensor shared_out,
    th::Tensor gamma,
    double eps
) {
    TORCH_CHECK(routed_out.is_cuda() && shared_out.is_cuda() && gamma.is_cuda(), "All inputs must be CUDA tensors");
    TORCH_CHECK(
        routed_out.is_contiguous() && shared_out.is_contiguous() && gamma.is_contiguous(),
        "fused_add_post_moe_norm expects contiguous inputs"
    );
    TORCH_CHECK(routed_out.sizes() == shared_out.sizes(), "routed_out and shared_out shapes must match");
    TORCH_CHECK(
        routed_out.scalar_type() == shared_out.scalar_type() && routed_out.scalar_type() == gamma.scalar_type(),
        "fused_add_post_moe_norm inputs must share dtype"
    );

    int const hidden_dim = routed_out.size(-1);
    TORCH_CHECK(hidden_dim == 2560, "fused_add_post_moe_norm requires hidden_dim == 2560");
    TORCH_CHECK(gamma.numel() == hidden_dim, "gamma size must match hidden_dim");
    int const tokens = routed_out.numel() / hidden_dim;

    c10::cuda::CUDAGuard device_guard(routed_out.device());
    auto stream = at::cuda::getCurrentCUDAStream(routed_out.device().index()).stream();

    auto moe_out = th::empty_like(routed_out);
    if (tokens == 0) {
        return moe_out;
    }

    auto const dtype = routed_out.scalar_type();
    if (dtype == th::kBFloat16) {
        tensorrt_llm::kernels::invokeFusedAddPostMoeNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(routed_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(shared_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(gamma.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(moe_out.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    } else if (dtype == th::kHalf) {
        tensorrt_llm::kernels::invokeFusedAddPostMoeNorm<half>(
            reinterpret_cast<half const*>(routed_out.data_ptr()),
            reinterpret_cast<half const*>(shared_out.data_ptr()),
            reinterpret_cast<half const*>(gamma.data_ptr()),
            reinterpret_cast<half*>(moe_out.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    } else if (dtype == th::kFloat) {
        tensorrt_llm::kernels::invokeFusedAddPostMoeNorm<float>(
            reinterpret_cast<float const*>(routed_out.data_ptr()),
            reinterpret_cast<float const*>(shared_out.data_ptr()),
            reinterpret_cast<float const*>(gamma.data_ptr()),
            reinterpret_cast<float*>(moe_out.data_ptr()),
            tokens,
            hidden_dim,
            static_cast<float>(eps),
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused_add_post_moe_norm (expected bf16, fp16, or fp32)");
    }

    return moe_out;
}

std::tuple<th::Tensor, th::Tensor> fused_kolibri_router(
    th::Tensor scores_in,
    c10::optional<th::Tensor> bias_opt
) {
    TORCH_CHECK(scores_in.is_cuda() && scores_in.is_contiguous(), "scores_in must be a contiguous CUDA tensor");
    TORCH_CHECK(scores_in.dim() == 2 && scores_in.size(1) == 384, "scores_in must have shape [tokens, 384]");

    int const num_tokens = static_cast<int>(scores_in.size(0));
    float const* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto const& b = *bias_opt;
        TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.numel() == 384, "bias must be contiguous CUDA [384]");
        TORCH_CHECK(b.scalar_type() == th::kFloat, "bias must be float32");
        bias_ptr = b.data_ptr<float>();
    }

    c10::cuda::CUDAGuard device_guard(scores_in.device());
    auto stream = at::cuda::getCurrentCUDAStream(scores_in.device().index()).stream();

    auto out_indices = th::empty({num_tokens, 6}, scores_in.options().dtype(th::kInt32));
    auto out_weights = th::empty({num_tokens, 6}, scores_in.options().dtype(th::kFloat));
    if (num_tokens == 0) {
        return {out_indices, out_weights};
    }

    auto const dtype = scores_in.scalar_type();
    if (dtype == th::kBFloat16) {
        kolibri::kernels::invokeKolibriRouter<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 const*>(scores_in.data_ptr()),
            bias_ptr,
            out_indices.data_ptr<int32_t>(),
            out_weights.data_ptr<float>(),
            num_tokens,
            stream
        );
    } else if (dtype == th::kHalf) {
        kolibri::kernels::invokeKolibriRouter<half>(
            reinterpret_cast<half const*>(scores_in.data_ptr()),
            bias_ptr,
            out_indices.data_ptr<int32_t>(),
            out_weights.data_ptr<float>(),
            num_tokens,
            stream
        );
    } else if (dtype == th::kFloat) {
        kolibri::kernels::invokeKolibriRouter<float>(
            scores_in.data_ptr<float>(),
            bias_ptr,
            out_indices.data_ptr<int32_t>(),
            out_weights.data_ptr<float>(),
            num_tokens,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused_kolibri_router (expected bf16, fp16, or fp32)");
    }

    return {out_indices, out_weights};
}

uint64_t configure_cuda_mempool(int64_t device_idx) {
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_idx));
    cudaMemPool_t mempool = nullptr;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&mempool, static_cast<int>(device_idx));
    TORCH_CHECK(err == cudaSuccess && mempool != nullptr,
                "cudaDeviceGetDefaultMemPool failed: ", cudaGetErrorString(err));
    uint64_t threshold = UINT64_MAX;
    err = cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    TORCH_CHECK(err == cudaSuccess,
                "cudaMemPoolSetAttribute failed: ", cudaGetErrorString(err));
    uint64_t readback = 0;
    err = cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &readback);
    TORCH_CHECK(err == cudaSuccess,
                "cudaMemPoolGetAttribute failed: ", cudaGetErrorString(err));
    return readback;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sandwich_norm", &fused_sandwich_norm, "Kolibri fused sandwich RMSNorm");
    m.def("fused_add_post_moe_norm", &fused_add_post_moe_norm, "Kolibri fused add + post-MoE RMSNorm");
    m.def("fused_kolibri_router", &fused_kolibri_router, "Kolibri fused Top-6 sigmoid router");
    m.def("configure_cuda_mempool", &configure_cuda_mempool, "Set default CUDA mempool release threshold to UINT64_MAX");
    m.def("detected_pcore_count", &kolibri::host::detected_pcore_count, "Detected Linux P-core count");
    m.def(
        "gather_experts_omp",
        &kolibri::host::gather_experts_omp,
        "Gather expert rows into pinned host bounce buffers via P-core pinned OpenMP",
        py::arg("src_tensors"),
        py::arg("dst_tensors"),
        py::arg("expert_ids"),
        py::arg("num_threads") = 0
    );
    m.def(
        "dma_slabs_async",
        &kolibri::cache::dma_slabs_async,
        "Issue async H2D cudaMemcpyAsync across all slabs and record copy event",
        py::arg("src_slabs"),
        py::arg("dst_slabs"),
        py::arg("count"),
        py::arg("dst_offset"),
        py::arg("stream_ptr"),
        py::arg("event_ptr")
    );
    m.def(
        "promote_experts_cuda",
        &kolibri::cache::promote_experts_cuda,
        "Promote missed experts from scratch to LRU victim slots and update LUT in 1 CUDA kernel",
        py::arg("cache_slabs"),
        py::arg("gpu_expert_to_slot"),
        py::arg("src_slots"),
        py::arg("dst_slots"),
        py::arg("old_eids"),
        py::arg("new_eids")
    );
}
