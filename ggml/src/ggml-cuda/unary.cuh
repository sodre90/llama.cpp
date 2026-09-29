#pragma once
#include "common.cuh"

#include <type_traits>

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

void ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * unary_node, ggml_tensor * scale2_node);

// SCALE + SILU on plain device buffers, for the hc up+pre fusion check; dst_q8_1 may be null
void ggml_cuda_scale_silu_raw(const float * x, float * dst, block_q8_1 * dst_q8_1, float scale, float bias, int k,
        int64_t ne10, int64_t ne10_padded, cudaStream_t stream);

// add = addend + x * sigmoid(gate), one gate value per row of x
void ggml_cuda_op_sigmoid_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * sigmoid_node, ggml_tensor * mul_node, ggml_tensor * add_node);

__device__ __forceinline__ float ggml_cuda_op_sigmoid_single(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_softplus_single(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// see ggml_cuda_mm_fusion_args_host::row_segments; a kernel parameter of its own, so the other launches' arguments stay small
struct mmvq_row_segments_args {
    int32_t           n                                  = 0;
    const void *      x[MMVQ_MAX_ROW_SEGMENTS]           = {};
    float *           dst[MMVQ_MAX_ROW_SEGMENTS]         = {};
    uint32_t          nrows[MMVQ_MAX_ROW_SEGMENTS]       = {};
    uint32_t          first_block[MMVQ_MAX_ROW_SEGMENTS] = {};
    mmvq_row_epilogue epilogue[MMVQ_MAX_ROW_SEGMENTS]    = {};
    const float *     bias[MMVQ_MAX_ROW_SEGMENTS]        = {};
    const float *     scale[MMVQ_MAX_ROW_SEGMENTS]       = {};
};

struct mmvq_no_row_segments {};

template <bool row_segments>
using mmvq_row_segments_param = std::conditional_t<row_segments, mmvq_row_segments_args, mmvq_no_row_segments>;

// the same expressions as the graph's ADD, SOFTPLUS + MUL and SIGMOID kernels, so fusing them changes no bits
static __device__ __forceinline__ float ggml_cuda_apply_row_epilogue(
        const float x, const mmvq_row_epilogue epilogue, const float * bias, const float * scale, const int row) {
    switch (epilogue) {
        case MMVQ_ROW_EPILOGUE_SIGMOID:
            return ggml_cuda_op_sigmoid_single(x);
        case MMVQ_ROW_EPILOGUE_SOFTPLUS_BIAS_SCALE:
            return ggml_cuda_op_softplus_single(x + bias[row]) * scale[row];
        default:
            return x;
    }
}

__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_clamp_single(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    return ggml_cuda_op_silu_single(gate) * up;
}

// the SCALE + UNARY step of scale_unary_kernel, shared with the hc up+pre matvec so both round the same
template <float (*op)(float)>
static __device__ __forceinline__ float ggml_cuda_scale_unary_single(const float scale, const float x, const float bias) {
    return op(scale * x + bias);
}
