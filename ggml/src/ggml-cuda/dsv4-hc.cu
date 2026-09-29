#include "common.cuh"
#include "dsv4-hc.cuh"
#include "norm.cuh"
#include "quantize.cuh"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <vector>


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        block_q8_1 * dst_q8_1,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        if constexpr (gated) {
            sum = ggml_cuda_dsv4_hc_pre_gated_step(sum, xv, weights[i0*sw0 + ih*sw1 + it*sw2]);
        } else {
            const float wv = weights[ih*sw0 + it*sw1];
            sum += xv * wv;
        }
    }

    const float out = scale * sum;
    dst[i0*sd0 + it*sd1] = out;
    if (dst_q8_1 != nullptr) {
        quantize_q8_1_element(out, dst_q8_1, ir);
    }
}

struct dsv4_hc_post_gate {
    float scale;
    float bias;
    float scale2;
    float bias2;
};

static __device__ __forceinline__ float dsv4_hc_post_gate_weight(const dsv4_hc_post_gate gate, const float logit) {
    return gate.scale2 * (1.0f / (1.0f + expf(-(gate.scale * logit + gate.bias)))) + gate.bias2;
}

// identity-comb gated hc_post of one stream of one token, then the RMS norm of that stream times its gamma; the norm
// repeats rms_norm_f32<block_size, true>'s per-thread sums and block reduction, so both outputs match the unfused ops
template <int block_size>
static __global__ void dsv4_hc_post_rms_norm_f32(
        const dsv4_hc_post_gate gate,
        const float * x,
        const float * residual,
        const float * logits,
        const float * gamma,
        float * dst_post,
        float * dst_norm,
        block_q8_1 * dst_q8_1,
        const int n_embd,
        const int64_t sx1,
        const int64_t sr1,
        const int64_t sr2,
        const int64_t sp0,
        const int64_t sp1,
        const int64_t sg1,
        const float eps) {
    ggml_cuda_pdl_lc();
    const int hc     = gridDim.x;
    const int stream = blockIdx.x;
    const int token  = blockIdx.y;
    const int tid    = threadIdx.x;

    const int64_t dst_row = ((int64_t) token*hc + stream)*n_embd;
    x        += token*sx1;
    residual += stream*sr1 + token*sr2;
    gamma    += stream*sg1;
    dst_post += dst_row;
    dst_norm += dst_row;

    ggml_cuda_pdl_sync();
    const float p = dsv4_hc_post_gate_weight(gate, logits[stream*sp0 + token*sp1]);

    float tmp = 0.0f;
    for (int col = tid; col < n_embd; col += block_size) {
        float sum = x[col] * p;
        sum += residual[col];
        dst_post[col] = sum;
        tmp += sum * sum;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < n_embd; col += block_size) {
        const float out = scale * dst_post[col] * gamma[col];
        dst_norm[col] = out;
        if (dst_q8_1) {
            quantize_q8_1_element(out, dst_q8_1, dst_row + col);
        }
    }
}

template <bool has_comb, bool gated_post>
static __global__ void dsv4_hc_post_f32(
        const dsv4_hc_post_gate gate,
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float p = post[idst*sp0 + it*sp1];
    if constexpr (gated_post) {
        p = dsv4_hc_post_gate_weight(gate, p);
    }

    float sum = x[i0*sx0 + it*sx1] * p;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    block_q8_1 * dst_q8_1 = ctx.q8_1_prequantize_dst(dst);

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data, dst_q8_1,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_dsv4_hc_pre_gated_raw(const float * x, const float * gate, float * dst, block_q8_1 * dst_q8_1,
        int64_t n_embd, int64_t hc, int64_t n_tokens, int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2, int64_t sd0, int64_t sd1, float scale, cudaStream_t stream) {
    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32<true>, launch_params,
            x, gate, dst, dst_q8_1, n_embd, hc, n_tokens, sx0, sx1, sx2, sw0, sw1, sw2, sd0, sd1, scale);
}

static void dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * post, const dsv4_hc_post_gate * gate) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? (gate ? dsv4_hc_post_f32<true, true>  : dsv4_hc_post_f32<true, false>)
                       : (gate ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            gate ? *gate : dsv4_hc_post_gate{},
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dsv4_hc_post(ctx, dst, dst->src[2], nullptr);
}

void ggml_cuda_op_dsv4_hc_post_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_node, const ggml_tensor * scale2_node) {
    const float * scale  = (const float *) scale_node->op_params;
    const float * scale2 = (const float *) scale2_node->op_params;
    const dsv4_hc_post_gate gate = { scale[0], scale[1], scale2[0], scale2[1] };
    dsv4_hc_post(ctx, dst, scale_node->src[0], &gate);
}

static void dsv4_hc_post_gated_rms_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
        const ggml_tensor * scale_node, const ggml_tensor * scale2_node, const ggml_tensor * norm_node, const ggml_tensor * mul_node,
        float * post_d, float * norm_d, block_q8_1 * q8_1_d) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * logits   = scale_node->src[0];
    const ggml_tensor * gamma    = mul_node->src[0] == norm_node ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(dst->src[3] == nullptr);
    GGML_ASSERT(x->nb[0] == sizeof(float) && residual->nb[0] == sizeof(float) && gamma->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(mul_node));

    const float * scale  = (const float *) scale_node->op_params;
    const float * scale2 = (const float *) scale2_node->op_params;
    const dsv4_hc_post_gate gate = { scale[0], scale[1], scale2[0], scale2[1] };

    float eps;
    memcpy(&eps, norm_node->op_params, sizeof(float));

    const int     n_embd   = (int) x->ne[0];
    const int64_t hc       = residual->ne[1];
    const int64_t n_tokens = x->ne[1];

    const dim3 grid_dims(hc, n_tokens, 1);
    auto launch = [&](auto kernel, const int block_size) {
        const dim3 block_dims(block_size, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 32*sizeof(float), ctx.stream());
        ggml_cuda_kernel_launch(kernel, launch_params,
                gate, (const float *) x->data, (const float *) residual->data, (const float *) logits->data,
                (const float *) gamma->data, post_d, norm_d, q8_1_d,
                n_embd,
                x->nb[1] / sizeof(float),
                residual->nb[1] / sizeof(float), residual->nb[2] / sizeof(float),
                logits->nb[0] / sizeof(float), logits->nb[1] / sizeof(float),
                gamma->nb[1] / sizeof(float),
                eps);
    };
    // the block size rms_norm_mul_f32_cuda picks, which fixes the reduction order
    if (n_embd < 1024) {
        launch(dsv4_hc_post_rms_norm_f32<256>, 256);
    } else {
        launch(dsv4_hc_post_rms_norm_f32<1024>, 1024);
    }
}

void ggml_cuda_op_dsv4_hc_post_gated_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_node, const ggml_tensor * scale2_node, const ggml_tensor * norm_node, ggml_tensor * mul_node) {
    dsv4_hc_post_gated_rms_norm(ctx, dst, scale_node, scale2_node, norm_node, mul_node,
            (float *) dst->data, (float *) mul_node->data, ctx.q8_1_prequantize_dst(mul_node));
}

void ggml_cuda_op_dsv4_hc_post_gated_rms_norm_to(ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
        const ggml_tensor * scale_node, const ggml_tensor * scale2_node, const ggml_tensor * norm_node, const ggml_tensor * mul_node,
        float * post_d, float * norm_d, block_q8_1 * q8_1_d) {
    dsv4_hc_post_gated_rms_norm(ctx, dst, scale_node, scale2_node, norm_node, mul_node, post_d, norm_d, q8_1_d);
}

bool ggml_cuda_hc_post_norm_check_enabled() {
    static const bool enabled = getenv("GGML_CUDA_HC_POST_NORM_CHECK") != nullptr && std::atoi(getenv("GGML_CUDA_HC_POST_NORM_CHECK"));
    return enabled;
}

template <typename T>
static int64_t dsv4_hc_count_differing(const T * a, const T * b, const size_t n, cudaStream_t stream, int64_t * first = nullptr) {
    std::vector<T> host_a(n);
    std::vector<T> host_b(n);
    CUDA_CHECK(cudaMemcpyAsync(host_a.data(), a, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_b.data(), b, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (first) {
        const auto mismatch = std::mismatch(host_a.begin(), host_a.end(), host_b.begin());
        *first = mismatch.first == host_a.end() ? -1 : (int64_t) (mismatch.first - host_a.begin());
    }
    return std::inner_product(host_a.begin(), host_a.end(), host_b.begin(), (int64_t) 0, std::plus<int64_t>(), std::not_equal_to<T>());
}

void ggml_cuda_dsv4_hc_post_rms_norm_check(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_node, const ggml_tensor * scale2_node, ggml_tensor * norm_node, ggml_tensor * mul_node) {
    cudaStream_t stream = ctx.stream();

    const int64_t n_tokens = dst->src[0]->ne[1];
    const int64_t ne10     = mul_node->ne[0]*mul_node->ne[1];
    GGML_ASSERT(n_tokens >= 1 && n_tokens <= MMVQ_MAX_ROW_SEGMENT_COLS);

    // the fused kernel writes the q8_1 rows back to back, which is the layout mul_mat_vec_q reads only without row padding
    const bool   check_q8_1  = ne10 % QK8_1 == 0 && GGML_PAD(ne10, MATRIX_ROW_PADDING) == ne10;
    const size_t q8_1_nbytes = check_q8_1 ? n_tokens*ne10/QK8_1*sizeof(block_q8_1) : 0;

    ggml_cuda_pool_alloc<float> post_fused(ctx.pool(), ggml_nelements(dst));
    ggml_cuda_pool_alloc<float> norm_fused(ctx.pool(), ggml_nelements(mul_node));
    ggml_cuda_pool_alloc<char>  q8_1_fused(ctx.pool());
    ggml_cuda_pool_alloc<char>  q8_1_ref(ctx.pool());
    if (check_q8_1) {
        q8_1_fused.alloc(q8_1_nbytes);
        q8_1_ref.alloc(q8_1_nbytes);
    }

    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * logits   = scale_node->src[0];

    ggml_cuda_pool_alloc<char> x_snapshot(ctx.pool());
    ggml_cuda_pool_alloc<char> residual_snapshot(ctx.pool());
    ggml_cuda_pool_alloc<char> logits_snapshot(ctx.pool());
    const auto take_snapshot = [&](ggml_cuda_pool_alloc<char> & snapshot, const ggml_tensor * t) {
        snapshot.alloc(ggml_nbytes(t));
        CUDA_CHECK(cudaMemcpyAsync(snapshot.get(), t->data, ggml_nbytes(t), cudaMemcpyDeviceToDevice, stream));
    };
    take_snapshot(x_snapshot, x);
    take_snapshot(residual_snapshot, residual);
    take_snapshot(logits_snapshot, logits);

    ggml_cuda_op_dsv4_hc_post_gated_rms_norm_to(ctx, dst, scale_node, scale2_node, norm_node, mul_node,
            post_fused.get(), norm_fused.get(), check_q8_1 ? (block_q8_1 *) q8_1_fused.get() : nullptr);

    ggml_cuda_op_dsv4_hc_post_gated(ctx, dst, scale_node, scale2_node);
    // compare post before the norm runs: the norm output may take over post's memory
    int64_t first_diff_post = -1;
    const int64_t n_diff_post = dsv4_hc_count_differing<uint32_t>((const uint32_t *) dst->data, (const uint32_t *) post_fused.get(), ggml_nelements(dst), stream, &first_diff_post);

    static std::atomic<int> diag_budget(20);
    if (n_diff_post != 0 && diag_budget.fetch_sub(1) > 0) {
        const int64_t k        = first_diff_post;
        const int64_t n_embd   = x->ne[0];
        const int64_t hc       = residual->ne[1];
        const int64_t col      = k % n_embd;
        const int64_t hc_strm  = (k / n_embd) % hc;
        const int64_t token    = k / (n_embd*hc);

        const auto read_word = [&](const void * base, const size_t offset) {
            uint32_t word;
            CUDA_CHECK(cudaMemcpyAsync(&word, (const char *) base + offset, sizeof(word), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return word;
        };
        const auto as_float = [](const uint32_t word) {
            float value;
            memcpy(&value, &word, sizeof(value));
            return value;
        };
        const auto count_words = [&](const void * a, const void * b, const size_t n_words) {
            return dsv4_hc_count_differing<uint32_t>((const uint32_t *) a, (const uint32_t *) b, n_words, stream);
        };

        const size_t post_offset   = k*sizeof(float);
        const size_t x_offset      = token*x->nb[1] + col*sizeof(float);
        const size_t res_offset    = hc_strm*residual->nb[1] + token*residual->nb[2] + col*sizeof(float);
        const size_t logits_offset = hc_strm*logits->nb[0] + token*logits->nb[1];

        const uint32_t word_a = read_word(post_fused.get(), post_offset);
        const uint32_t word_b = read_word(dst->data, post_offset);

        ggml_cuda_pool_alloc<float> dst_before(ctx.pool(), ggml_nelements(dst));
        CUDA_CHECK(cudaMemcpyAsync(dst_before.get(), dst->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));

        ggml_cuda_pool_alloc<float> post2(ctx.pool(), ggml_nelements(dst));
        ggml_cuda_pool_alloc<float> norm2(ctx.pool(), ggml_nelements(mul_node));
        ggml_cuda_op_dsv4_hc_post_gated_rms_norm_to(ctx, dst, scale_node, scale2_node, norm_node, mul_node, post2.get(), norm2.get(), nullptr);
        const uint32_t word_a2 = read_word(post2.get(), post_offset);
        const int64_t n_post2_vs_fused = count_words(post2.get(), post_fused.get(), ggml_nelements(dst));
        const int64_t n_post2_vs_dst   = count_words(post2.get(), dst->data, ggml_nelements(dst));

        ggml_cuda_op_dsv4_hc_post_gated(ctx, dst, scale_node, scale2_node);
        const uint32_t word_b2 = read_word(dst->data, post_offset);
        const int64_t n_dst_vs_before = count_words(dst->data, dst_before.get(), ggml_nelements(dst));

        const int64_t n_x_changed      = count_words(x_snapshot.get(), x->data, ggml_nbytes(x)/sizeof(uint32_t));
        const int64_t n_res_changed    = count_words(residual_snapshot.get(), residual->data, ggml_nbytes(residual)/sizeof(uint32_t));
        const int64_t n_logits_changed = count_words(logits_snapshot.get(), logits->data, ggml_nbytes(logits)/sizeof(uint32_t));

        const uint32_t x_now      = read_word(x->data, x_offset);
        const uint32_t x_snap     = read_word(x_snapshot.get(), x_offset);
        const uint32_t res_now    = read_word(residual->data, res_offset);
        const uint32_t res_snap   = read_word(residual_snapshot.get(), res_offset);
        const uint32_t logit_now  = read_word(logits->data, logits_offset);
        const uint32_t logit_snap = read_word(logits_snapshot.get(), logits_offset);

        GGML_LOG_WARN("hc_post_norm_check diag: n_tokens=%" PRId64 " dst=%s x=%s residual=%s logits=%s ptrs dst=%p x=%p residual=%p logits=%p post_fused=%p"
                " k=%" PRId64 " token=%" PRId64 " stream=%" PRId64 " col=%" PRId64 "\n",
                n_tokens, dst->name, x->name, residual->name, logits->name,
                dst->data, x->data, residual->data, logits->data, (void *) post_fused.get(),
                k, token, hc_strm, col);
        GGML_LOG_WARN("hc_post_norm_check diag: A=0x%08x (%.9g) B=0x%08x (%.9g) A2=0x%08x (%.9g) B2=0x%08x (%.9g)"
                " x now=%.9g snap=%.9g residual now=%.9g snap=%.9g logit now=%.9g snap=%.9g"
                " input words changed x=%" PRId64 " residual=%" PRId64 " logits=%" PRId64
                " post2_vs_fused=%" PRId64 " post2_vs_dst=%" PRId64 " dst_vs_before=%" PRId64 "\n",
                word_a, as_float(word_a), word_b, as_float(word_b), word_a2, as_float(word_a2), word_b2, as_float(word_b2),
                as_float(x_now), as_float(x_snap), as_float(res_now), as_float(res_snap), as_float(logit_now), as_float(logit_snap),
                n_x_changed, n_res_changed, n_logits_changed,
                n_post2_vs_fused, n_post2_vs_dst, n_dst_vs_before);
    }
    ggml_cuda_op_rms_norm_fused(ctx, norm_node, mul_node);

    if (check_q8_1) {
        quantize_row_q8_1_cuda((const float *) mul_node->data, nullptr, q8_1_ref.get(), GGML_TYPE_Q8_0, ne10, ne10, n_tokens*ne10, n_tokens*ne10,
                ne10, n_tokens, 1, 1, stream);
    }

    const int64_t n_diff_norm = dsv4_hc_count_differing<uint32_t>((const uint32_t *) mul_node->data, (const uint32_t *) norm_fused.get(), ggml_nelements(mul_node), stream);
    const int64_t n_diff_q8_1 = check_q8_1 ? dsv4_hc_count_differing<char>(q8_1_ref.get(), q8_1_fused.get(), q8_1_nbytes, stream) : 0;

    static std::atomic<int64_t> n_checked_ncols[MMVQ_MAX_ROW_SEGMENT_COLS + 1];
    static std::atomic<int64_t> n_differing_ncols[MMVQ_MAX_ROW_SEGMENT_COLS + 1];
    const int64_t n_seen = n_checked_ncols[n_tokens].fetch_add(1) + 1;
    const bool differs = n_diff_post != 0 || n_diff_norm != 0 || n_diff_q8_1 != 0;
    const int64_t n_differing = differs ? n_differing_ncols[n_tokens].fetch_add(1) + 1 : n_differing_ncols[n_tokens].load();
    if (differs || n_seen <= 3 || n_seen % 1000 == 0) {
        GGML_LOG_WARN("hc_post_norm_check: ncols=%d %" PRId64 " post values differ, %" PRId64 " norm values differ, %" PRId64 " q8_1 bytes differ (%" PRId64 " checks, %" PRId64 " differing)\n",
                (int) n_tokens, n_diff_post, n_diff_norm, n_diff_q8_1, n_seen, n_differing);
    }
}
