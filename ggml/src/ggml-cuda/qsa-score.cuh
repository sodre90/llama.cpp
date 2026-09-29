#include "common.cuh"

struct ggml_cuda_qsa_score_epilogue {
    const ggml_tensor * score  = nullptr; // RELU's src[0]: contiguous F32 [nb, n_heads, n_t, ns]
    ggml_tensor *       fin    = nullptr; // final ADD: contiguous F32 [nb, n_t, ns], written by the fused launch
    // mode A (device rule), all F32, read through their own nb[] strides
    const ggml_tensor * start  = nullptr; // REPEAT's src[0]: [nb, 1, ns]
    const ggml_tensor * q      = nullptr; // SUB's src[1]: [1, n_t, ns]
    const ggml_tensor * m      = nullptr; // tail ADD's src[1]: [1, n_t, ns]
    const ggml_tensor * spare  = nullptr; // forced ADD's src[1]: [nb, 1, ns]
    float future_min = 0, future_max = 0, tail_min = 0, tail_max = 0; // CLAMP op_params
    float forced_scale = 0, forced_bias = 0, future_scale = 0, future_bias = 0; // SCALE op_params
    // mode B: fin's src[1], F32 [nb, n_t, ns], read through its nb[] strides
    const ggml_tensor * bias   = nullptr;
    int fin_idx = -1; // node index of fin
    bool device_rule() const { return start != nullptr; }
};

void ggml_cuda_op_qsa_score_epilogue(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_score_epilogue & e, float * dst);
void ggml_cuda_qsa_score_check(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_score_epilogue & e, const float * fused);
bool ggml_cuda_qsa_score_fusion_enabled();
bool ggml_cuda_qsa_score_check_enabled();
