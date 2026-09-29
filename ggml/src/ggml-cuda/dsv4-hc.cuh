#pragma once
#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst's post weights (src[2]) are the output of scale2(sigmoid(scale(logits))): computes them from the logits in the same
// kernel, so that chain is never launched
void ggml_cuda_op_dsv4_hc_post_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_node, const ggml_tensor * scale2_node);

// the gated hc_post above, followed by RMS_NORM of its output and MUL by a per-stream gamma: writes both dst and mul_node
void ggml_cuda_op_dsv4_hc_post_gated_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_node, const ggml_tensor * scale2_node, const ggml_tensor * norm_node, ggml_tensor * mul_node);

// one stream of the gated mean in dsv4_hc_pre_f32, shared with the hc up+pre matvec so both round the same
static __device__ __forceinline__ float ggml_cuda_dsv4_hc_pre_gated_step(const float sum, const float xv, const float gate) {
    const float wv = 1.0f / (1.0f + expf(-gate));
    return sum + xv * wv;
}

// the gated hc_pre on plain device buffers, for the hc up+pre fusion check; dst_q8_1 may be null
void ggml_cuda_dsv4_hc_pre_gated_raw(const float * x, const float * gate, float * dst, block_q8_1 * dst_q8_1,
        int64_t n_embd, int64_t hc, int64_t n_tokens, int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2, int64_t sd0, int64_t sd1, float scale, cudaStream_t stream);
