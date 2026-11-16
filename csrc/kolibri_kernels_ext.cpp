#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sandwich_norm", &fused_sandwich_norm, "Kolibri fused sandwich RMSNorm");
    m.def("fused_add_post_moe_norm", &fused_add_post_moe_norm, "Kolibri fused add + post-MoE RMSNorm");
}
