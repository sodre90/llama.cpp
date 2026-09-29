#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// weight types whose single-token MUL_MATs can share one launch (ggml_cuda_mm_fusion_args_host::row_segment_nodes)
bool ggml_cuda_mmvq_row_segments_supported(enum ggml_type type);

// GGML_CUDA_MMVQ_MULTI_ROWS_CHECK=1 compares every multi-row launch against the one-row kernel on the host, which a captured graph cannot do
bool ggml_cuda_mmvq_multi_rows_check_enabled();

// GGML_CUDA_MMVQ_TRIM_WARPS_CHECK=1 compares every trimmed-block launch against the default block on the host, which a captured graph cannot do
bool ggml_cuda_mmvq_trim_warps_check_enabled();

// GGML_CUDA_Q8_1_PREQ_CHECK=1 compares every reused q8_1 copy against a fresh quantization on the host, which a captured graph cannot do
bool ggml_cuda_q8_1_preq_check_enabled();

// GGML_CUDA_Q8_1_REUSE_DEBUG=1 logs the launches that quantize src1 although the reuse slots could have held it, and the slots dropped by overwrites
bool ggml_cuda_q8_1_reuse_debug_enabled();

// GGML_CUDA_MMVQ_QUANT_PROLOGUE_CHECK=1 compares every launch that quantizes src1 in the kernel against the quantize + matvec kernels on the host, which a captured graph cannot do
bool ggml_cuda_mmvq_quant_prologue_check_enabled();

// GGML_CUDA_MOE_FILL_CHECK=1 compares every MoE launch that fills missed experts against a launch that reads them in place, on the host, which a captured graph cannot do
bool ggml_cuda_moe_fill_check_enabled();

// GGML_CUDA_HC_UP_PRE_CHECK=1 compares every fused hc up+pre launch against the unfused ops on the host, which a captured graph cannot do
bool ggml_cuda_hc_up_pre_check_enabled();

// GGML_CUDA_HC_INJECT_FUSION=0 keeps the BF16 inject matvec of the hc mix out of the hc up+pre launch
bool ggml_cuda_hc_inject_fusion_enabled();

// whether hc_up would run as the one-warp small-K kernel with a single K pass, so the hc up+pre kernel can match it
bool ggml_cuda_mmvq_hc_up_pre_supported(const ggml_tensor * w_up, int64_t ncols_dst, int cc);

// the SCALE + SILU, hc up MUL_MAT and gated DSV4_HC_PRE nodes can run as one launch on this device
bool ggml_cuda_hc_up_pre_supported(const ggml_tensor * scale_node, const ggml_tensor * mm_node, const ggml_tensor * pre_node, int cc);

// writes pre_node from the three nodes, none of the intermediates
void ggml_cuda_op_mul_mat_vec_q_hc_up_pre(ggml_backend_cuda_context & ctx,
    const ggml_tensor * scale_node, const ggml_tensor * mm_node, ggml_tensor * pre_node);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
