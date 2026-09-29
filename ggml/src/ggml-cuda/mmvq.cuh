#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// weight types whose single-token MUL_MATs can share one launch (ggml_cuda_mm_fusion_args_host::row_segment_nodes)
bool ggml_cuda_mmvq_row_segments_supported(enum ggml_type type);

// GGML_CUDA_MMVQ_MULTI_ROWS_CHECK=1 compares every multi-row launch against the one-row kernel on the host, which a captured graph cannot do
bool ggml_cuda_mmvq_multi_rows_check_enabled();

// GGML_CUDA_Q8_1_PREQ_CHECK=1 compares every reused q8_1 copy against a fresh quantization on the host, which a captured graph cannot do
bool ggml_cuda_q8_1_preq_check_enabled();

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
