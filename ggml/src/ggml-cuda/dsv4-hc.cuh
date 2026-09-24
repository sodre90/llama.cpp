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
