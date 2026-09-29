#include "mmvq.cuh"
#include "dsv4-hc.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        // Small K leaves most of 8 warps without K blocks, one warp over several rows keeps its lanes busy.
        // Every ncols_dst splits K like ncols_dst=1, so a batched token gets the same bits as a lone one.
        GGML_UNUSED(ncols_dst);
        if (small_k) {
            return 1;
        }
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
                return 8;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        return small_k ? 4 : 1;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// the weight channel an expert id reads: its cache slot, or the id itself in the host tensor (see ggml_cuda_mmid_host_experts)
static __device__ __forceinline__ uint32_t mmvq_expert_channel(
        const ggml_cuda_mm_fusion_args_device & fusion, const uint32_t expert, const void *& x, const void *& gate) {
    if (fusion.expert_slot == nullptr) {
        return expert;
    }
    const int32_t slot = fusion.expert_slot[expert];
    if (slot < fusion.n_expert_slots) {
        return slot;
    }
    x    = fusion.x_host;
    gate = fusion.gate_host;
    return expert;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool row_segments = false, int rows_override = 0, int warps_override = 0>
__launch_bounds__((warps_override > 0 ? warps_override : calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters))*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const mmvq_row_segments_param<row_segments> segments, const bool slot_major) {
    uint32_t nrows_dst = stride_col_dst;
    uint32_t block_x   = blockIdx.x;
    uint32_t channel_dst = blockIdx.y;
    if (slot_major && ids_ptr) {
        const uint32_t bid = blockIdx.x + gridDim.x*blockIdx.y;
        channel_dst = bid % gridDim.y;
        block_x     = bid / gridDim.y;
    }
    const bool shared_expert = has_fusion && fusion.shared_up && channel_dst == gridDim.y - 1;
    if (shared_expert) {
        channel_dst = 0;
        vx_ptr = fusion.shared_up;
        dst_ptr = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }
    uint32_t block_row = block_x;
    [[maybe_unused]] mmvq_row_epilogue epilogue = MMVQ_ROW_EPILOGUE_NONE;
    [[maybe_unused]] const float * epilogue_bias  = nullptr;
    [[maybe_unused]] const float * epilogue_scale = nullptr;
    if constexpr (row_segments) {
        static_assert(ncols_dst <= MMVQ_MAX_ROW_SEGMENT_COLS && !has_fusion, "row segments are plain matvecs of a few tokens");
#pragma unroll
        for (int s = 0; s < MMVQ_MAX_ROW_SEGMENTS; ++s) {
            if (block_x >= segments.first_block[s]) {
                vx_ptr         = segments.x[s];
                dst_ptr        = segments.dst[s];
                nrows_dst      = segments.nrows[s];
                block_row      = block_x - segments.first_block[s];
                epilogue       = segments.epilogue[s];
                epilogue_bias  = segments.bias[s];
                epilogue_scale = segments.scale[s];
            }
        }
    }

    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = warps_override > 0 ? warps_override : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    // rows per block does not change the K split or the reduction, so the bits match the default
    constexpr int rows_per_cuda_block = rows_override > 0 ? rows_override : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*block_row;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const void * x_weights = vx_ptr;
    const uint32_t channel_w = ncols_dst == 1 && ids ? mmvq_expert_channel(fusion, channel_x, x_weights, vgate) : channel_x;
    const void * GGML_CUDA_RESTRICT vx = x_weights;

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_w*stride_channel_x + row0*stride_row_x;

    // small-K rows take only a few K steps per thread, unrolling them is slower
    constexpr int kbx_unroll = small_k ? 1 : 2;
#pragma unroll kbx_unroll
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                if constexpr (row_segments) {
                    result = ggml_cuda_apply_row_epilogue(result, epilogue, epilogue_bias, epilogue_scale, row0 + i);
                    dst[j*nrows_dst + i] = result;
                } else {
                    dst[j*stride_col_dst + i] = result;
                }
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const bool slot_major) {
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    uint32_t block_x     = blockIdx.x;
    uint32_t channel_dst = blockIdx.y;
    if (slot_major) {
        const uint32_t bid = blockIdx.x + gridDim.x*blockIdx.y;
        channel_dst = bid % gridDim.y;
        block_x     = bid / gridDim.y;
    }

    const bool shared_expert = has_fusion && fusion.shared_up && channel_dst == gridDim.y - 1;
    if (shared_expert) {
        channel_dst = 0;
        vx_ptr = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const int      row0        = c_rows_per_block*block_x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const void * x_weights = vx_ptr;
    const uint32_t channel_w = mmvq_expert_channel(fusion, channel_x, x_weights, vgate);
    const void * GGML_CUDA_RESTRICT vx = x_weights;

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_w*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

static constexpr int mmvq_hc_up_pre_warps   = 16;
static constexpr int mmvq_hc_up_pre_outputs = 32;
static constexpr int mmvq_hc_up_pre_max_k   = 512;
static constexpr int mmvq_hc_up_pre_max_hc  = 4;

static_assert(mmvq_hc_up_pre_outputs % mmvq_hc_up_pre_warps == 0, "outputs must divide evenly between warps");
static constexpr int mmvq_hc_up_pre_outputs_per_warp = mmvq_hc_up_pre_outputs / mmvq_hc_up_pre_warps;

// SCALE + SILU of the hc down projection, the hc up matvec and the gated hc_pre in one launch: each warp takes two
// output elements, and its lanes split K like the one-warp small-K kernel of mul_mat_vec_q, so every value matches the unfused ops
template <ggml_type type, int ncols_dst>
__launch_bounds__(mmvq_hc_up_pre_warps*32, 1)
static __global__ void mul_mat_vec_q_hc_up_pre(
        const float * lo, const void * vx_ptr, const float * xn, float * dst, block_q8_1 * dst_q8_1,
        const float lo_scale, const float lo_bias, const float pre_scale,
        const int ncols_x, const int n_embd, const int hc, const int stride_row_x,
        const int64_t sx0, const int64_t sx1, const int64_t sx2, const int64_t sd0, const int64_t sd1) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = vdr*warp_size/qi;

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    __shared__ __align__(16) char y_lds_bytes[ncols_dst*(mmvq_hc_up_pre_max_k/QK8_1)*sizeof(block_q8_1)];
    __shared__ float out_lds[ncols_dst][mmvq_hc_up_pre_outputs];
    block_q8_1 * y_lds = (block_q8_1 *) y_lds_bytes;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int blocks_per_row_x = ncols_x / qk;
    const int y_blocks_per_col = ncols_x / QK8_1;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    constexpr int max_rounds = (ncols_dst*(mmvq_hc_up_pre_max_k/QK8_1) + mmvq_hc_up_pre_warps - 1) / mmvq_hc_up_pre_warps;
    const int n_y_blocks = ncols_dst*y_blocks_per_col;
    float y_round[max_rounds];
#pragma unroll
    for (int r = 0; r < max_rounds; ++r) {
        const int b = warp + r*mmvq_hc_up_pre_warps;
        y_round[r] = b < n_y_blocks ? lo[(int64_t) (b / y_blocks_per_col)*ncols_x + (b % y_blocks_per_col)*QK8_1 + lane] : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < max_rounds; ++r) {
        const int b = warp + r*mmvq_hc_up_pre_warps;
        if (b < n_y_blocks) {
            const float y = ggml_cuda_scale_unary_single<ggml_cuda_op_silu_single>(lo_scale, y_round[r], lo_bias);
            quantize_q8_1_element(y, y_lds + (b / y_blocks_per_col)*y_blocks_per_col, (b % y_blocks_per_col)*QK8_1 + lane);
        }
    }
    __syncthreads();

    const void * GGML_CUDA_RESTRICT vx = vx_ptr;

    // the streams sit inside the K loop, as the rows of a block do in mul_mat_vec_q, so their loads go out together
    float tmp[mmvq_hc_up_pre_outputs_per_warp][mmvq_hc_up_pre_max_hc][ncols_dst] = {{{0.0f}}};
#pragma unroll 1
    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (lane % (qi/vdr));

#pragma unroll
        for (int o = 0; o < mmvq_hc_up_pre_outputs_per_warp; ++o) {
            const int i_o = blockIdx.x*mmvq_hc_up_pre_outputs + o*mmvq_hc_up_pre_warps + warp;
#pragma unroll
            for (int c = 0; c < mmvq_hc_up_pre_max_hc; ++c) {
                if (c < hc) {
                    const int kbx_offset = (c*n_embd + i_o)*stride_row_x;
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        tmp[o][c][j] += vec_dot_q_cuda(vx, &y_lds[j*y_blocks_per_col + kby], kbx_offset + kbx, kqs);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int o = 0; o < mmvq_hc_up_pre_outputs_per_warp; ++o) {
        const int i_o = blockIdx.x*mmvq_hc_up_pre_outputs + o*mmvq_hc_up_pre_warps + warp;

        float gate[mmvq_hc_up_pre_max_hc][ncols_dst];
#pragma unroll
        for (int c = 0; c < mmvq_hc_up_pre_max_hc; ++c) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                gate[c][j] = warp_reduce_sum<warp_size>(tmp[o][c][j]);
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int ih = 0; ih < mmvq_hc_up_pre_max_hc; ++ih) {
                if (ih < hc) {
                    sum = ggml_cuda_dsv4_hc_pre_gated_step(sum, xn[i_o*sx0 + ih*sx1 + j*sx2], gate[ih][j]);
                }
            }
            if (lane == 0) {
                out_lds[j][o*mmvq_hc_up_pre_warps + warp] = pre_scale * sum;
            }
        }
    }
    __syncthreads();

    if (warp < ncols_dst) {
        const float   out = out_lds[warp][lane];
        const int64_t ir  = (int64_t) blockIdx.x*mmvq_hc_up_pre_outputs + lane;
        dst[ir*sd0 + warp*sd1] = out;
        if (dst_q8_1 != nullptr) {
            quantize_q8_1_element(out, dst_q8_1, warp*(int64_t) n_embd + ir);
        }
    }
}

static constexpr bool mmvq_row_segments_supported(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_XS;
}

bool ggml_cuda_mmvq_row_segments_supported(ggml_type type) {
    return mmvq_row_segments_supported(type);
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false,
        const int rows_override = 0, const int warps_override = 0) {
    const int nwarps = warps_override > 0 ? warps_override : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = rows_override > 0 ? rows_override : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

// lays the segments out block after block; returns the row count that calc_launch_params turns into their total
static int mmvq_place_row_segments(mmvq_row_segments_args & segments, const int rows_per_block) {
    uint32_t nblocks = 0;
    for (int s = 0; s < MMVQ_MAX_ROW_SEGMENTS; ++s) {
        if (s < segments.n) {
            segments.first_block[s] = nblocks;
            nblocks += (segments.nrows[s] + rows_per_block - 1) / rows_per_block;
        } else {
            segments.first_block[s] = UINT32_MAX;
        }
    }
    return nblocks * rows_per_block;
}

static constexpr int mmvq_trim_warps = 4;

static constexpr bool mmvq_trim_warps_supported(ggml_type type) {
    return type == GGML_TYPE_IQ4_XS;
}

static bool ggml_cuda_mmvq_trim_warps() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_TRIM_WARPS");
        const bool on = env == nullptr || std::atoi(env) != 0;
        GGML_LOG_WARN("ggml_cuda: mmvq trim warps: %s\n", on ? "on (IQ4_XS, 4 warps when K fits one pass)" : "off");
        return on;
    }();
    return enabled;
}

bool ggml_cuda_mmvq_trim_warps_check_enabled() {
    static const bool enabled = getenv("GGML_CUDA_MMVQ_TRIM_WARPS_CHECK") != nullptr && std::atoi(getenv("GGML_CUDA_MMVQ_TRIM_WARPS_CHECK"));
    return enabled;
}

static constexpr bool mmvq_multi_rows_supported(ggml_type type) {
    return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_Q8_0   || type == GGML_TYPE_Q5_K   || type == GGML_TYPE_Q4_K;
}

static constexpr bool mmvq_multi_rows_default_type(ggml_type type) {
    return type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q4_K;
}

static bool ggml_cuda_mmvq_multi_rows_all_types() {
    static const bool all_types = getenv("GGML_CUDA_MMVQ_MULTI_ROWS_ALL_TYPES") != nullptr && std::atoi(getenv("GGML_CUDA_MMVQ_MULTI_ROWS_ALL_TYPES"));
    return all_types;
}

static int ggml_cuda_mmvq_multi_rows() {
    static const int rows = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_MULTI_ROWS");
        if (env == nullptr) {
            return 2;
        }
        const int value = std::atoi(env);
        return value == 1 || value == 2 || value == 4 ? value : 1;
    }();
    return rows;
}

// few-row matrices would drop into the few-waves regime, where fewer blocks hide less latency
static int64_t ggml_cuda_mmvq_multi_rows_min_rows() {
    static const int64_t min_rows = getenv("GGML_CUDA_MMVQ_MULTI_ROWS_MIN_ROWS") ? std::atoll(getenv("GGML_CUDA_MMVQ_MULTI_ROWS_MIN_ROWS")) : 2048;
    return min_rows;
}

static bool ggml_cuda_moe_slot_major() {
    static const bool slot_major = [] {
        const char * env = getenv("GGML_CUDA_MOE_SLOT_MAJOR");
        const bool enabled = env == nullptr || std::atoi(env) != 0;
        GGML_LOG_WARN("ggml_cuda: moe matvec block order: %s\n", enabled ? "slot-major" : "row-major");
        return enabled;
    }();
    return slot_major;
}

bool ggml_cuda_mmvq_multi_rows_check_enabled() {
    static const bool enabled = getenv("GGML_CUDA_MMVQ_MULTI_ROWS_CHECK") != nullptr && std::atoi(getenv("GGML_CUDA_MMVQ_MULTI_ROWS_CHECK"));
    return enabled;
}

struct mmvq_check_output {
    float * dst;
    size_t  n;
    float * scratch = nullptr;
};

static int64_t mmvq_count_differing_bits(const float * a, const float * b, const size_t n, cudaStream_t stream) {
    std::vector<float> host_a(n);
    std::vector<float> host_b(n);
    CUDA_CHECK(cudaMemcpyAsync(host_a.data(), a, n*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_b.data(), b, n*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t n_diff = 0;
    for (size_t k = 0; k < n; ++k) {
        n_diff += memcmp(&host_a[k], &host_b[k], sizeof(float)) != 0;
    }
    return n_diff;
}

static void mmvq_multi_rows_check_report(const ggml_type type, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int nchannels_dst, const int nsamples_dst, const int rows, const int64_t n_diff) {
    static std::atomic<int64_t> n_checked{0};
    const int64_t n_seen = n_checked.fetch_add(1) + 1;
    if (n_diff != 0 || n_seen % 1000 == 1) {
        GGML_LOG_WARN("%s: %s x [%d, %d] ncols=%d channels=%d samples=%d R=%d %" PRId64 " values differ (%" PRId64 " checked)\n", __func__,
                ggml_type_name(type), ncols_x, nrows_x, ncols_dst, nchannels_dst, nsamples_dst, rows, n_diff, n_seen);
    }
    GGML_ASSERT(n_diff == 0);
}

static void mmvq_trim_warps_check_report(const ggml_type type, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int nchannels_dst, const int nsamples_dst, const int64_t n_diff) {
    static std::atomic<int64_t> n_checked{0};
    const int64_t n_seen = n_checked.fetch_add(1) + 1;
    if (n_diff != 0 || n_seen % 1000 == 1) {
        GGML_LOG_WARN("%s: %s [%d, %d] ncols=%d channels=%d samples=%d %" PRId64 " values differ (%" PRId64 " checked)\n", __func__,
                ggml_type_name(type), ncols_x, nrows_x, ncols_dst, nchannels_dst, nsamples_dst, n_diff, n_seen);
    }
    GGML_ASSERT(n_diff == 0);
}

// copies dst to scratch, lets run_reference rewrite it there (segment dsts swapped for scratch) and counts the values whose bits differ
template <typename run_reference_t>
static int64_t mmvq_count_differing_bits_vs_reference(const run_reference_t & run_reference, float * dst, const mmvq_row_segments_args * segments,
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_dst,
        const int stride_col_dst, const int stride_channel_dst, const int stride_sample_dst, cudaStream_t stream) {
    const int64_t dst_extent = (int64_t) (nsamples_dst - 1)*stride_sample_dst + (int64_t) (nchannels_dst - 1)*stride_channel_dst;
    std::vector<mmvq_check_output> outputs;
    if (segments) {
        for (int s = 0; s < segments->n; ++s) {
            outputs.push_back({segments->dst[s], (size_t) (dst_extent + (int64_t) ncols_dst*segments->nrows[s])});
        }
    } else {
        outputs.push_back({dst, (size_t) (dst_extent + (int64_t) (ncols_dst - 1)*stride_col_dst + nrows_x)});
    }

    mmvq_row_segments_args segments_ref;
    if (segments) {
        segments_ref = *segments;
    }
    for (size_t k = 0; k < outputs.size(); ++k) {
        mmvq_check_output & output = outputs[k];
        CUDA_CHECK(cudaMalloc(&output.scratch, output.n*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(output.scratch, output.dst, output.n*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (segments) {
            segments_ref.dst[k] = output.scratch;
        }
    }

    run_reference(outputs[0].scratch, segments ? &segments_ref : nullptr);

    int64_t n_diff = 0;
    for (const mmvq_check_output & output : outputs) {
        n_diff += mmvq_count_differing_bits(output.dst, output.scratch, output.n, stream);
        CUDA_CHECK(cudaFree(output.scratch));
    }
    return n_diff;
}

bool ggml_cuda_q8_1_preq_check_enabled() {
    static const bool enabled = getenv("GGML_CUDA_Q8_1_PREQ_CHECK") != nullptr && std::atoi(getenv("GGML_CUDA_Q8_1_PREQ_CHECK"));
    return enabled;
}

// only the data blocks of each row are compared, as mul_mat_vec_q never reads the padding blocks
static void mmvq_q8_1_preq_check(const ggml_tensor * src1, const void * cached, const void * fresh, const int64_t ne10,
        const int64_t ne10_padded, const int64_t rows, cudaStream_t stream) {
    const size_t row_stride = ne10_padded/QK8_1*sizeof(block_q8_1);
    const size_t row_bytes  = ne10/QK8_1*sizeof(block_q8_1);
    std::vector<char> host_cached(rows*row_stride);
    std::vector<char> host_fresh(rows*row_stride);
    CUDA_CHECK(cudaMemcpyAsync(host_cached.data(), cached, host_cached.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_fresh.data(),  fresh,  host_fresh.size(),  cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t n_diff = 0;
    for (int64_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < row_bytes; ++b) {
            n_diff += host_cached[r*row_stride + b] != host_fresh[r*row_stride + b];
        }
    }

    static std::atomic<int64_t> n_checked{0};
    static std::atomic<bool>    padded_rows_logged{false};
    const int64_t n_seen = n_checked.fetch_add(1) + 1;
    const bool first_padded_rows = rows > 1 && ne10_padded != ne10 && !padded_rows_logged.exchange(true);
    if (n_diff != 0 || n_seen % 1000 == 1 || first_padded_rows) {
        GGML_LOG_WARN("%s: %s [%" PRId64 ", %" PRId64 "] ne10_padded %" PRId64 " %" PRId64 " bytes differ (%" PRId64 " checked)\n", __func__,
                src1->name, ne10, rows, ne10_padded, n_diff, n_seen);
    }
    GGML_ASSERT(n_diff == 0);
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, int rows_override = 0, int warps_override = 0>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream, const mmvq_row_segments_args * segments = nullptr) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst <= MMVQ_MAX_ROW_SEGMENT_COLS && mmvq_row_segments_supported(type)) {
        if (segments) {
            GGML_ASSERT(!has_fusion && !ids);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, true, rows_override, warps_override>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, *segments, false);
            return;
        }
    }
    GGML_ASSERT(!segments);

    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, false, 0, warps_override>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, mmvq_no_row_segments{}, ids && ggml_cuda_moe_slot_major());
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, false, rows_override, warps_override>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, mmvq_no_row_segments{}, ids && ggml_cuda_moe_slot_major());
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    const bool slot_major = ggml_cuda_moe_slot_major();

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, slot_major);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, slot_major);
    }
}

template <ggml_type type>
static bool mmvq_should_use_small_k(const int cc, const mmvq_parameter_table_id table_id, const int blocks_per_row_x,
        const int blocks_per_iter_1warp, const int c_ncols_dst) {
    // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
    // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
    const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
    bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

    constexpr std::array<ggml_type, 2> iq_slow_turing = {
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
    };
    constexpr std::array<ggml_type, 8> iq_slow_other = {
        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
    };
    constexpr std::array<ggml_type, 3> slow_pascal = {
        GGML_TYPE_IQ3_S,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
    };

    const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
    const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

    if (is_nvidia_turing_plus) {
        if (c_ncols_dst == 1 &&
                std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
            use = false;
        }
    } else if ((c_ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
            (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
            (GGML_CUDA_CC_IS_RDNA(cc) && !GGML_CUDA_CC_IS_RDNA4(cc))) {
        use = false;
    }

    return use;
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const mmvq_row_segments_args * segments = nullptr) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);
    GGML_ASSERT(!segments || ncols_dst <= MMVQ_MAX_ROW_SEGMENT_COLS);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    // K fits one pass of the trimmed block, so the dropped warps only held zero partials
    [[maybe_unused]] const bool trim_warps = mmvq_trim_warps_supported(type) && table_id == MMVQ_PARAMETERS_RDNA4 && !has_ids &&
        calc_nwarps(type, 1, table_id) > mmvq_trim_warps && blocks_per_row_x <= mmvq_trim_warps * blocks_per_iter_1warp &&
        ggml_cuda_mmvq_trim_warps();

    const auto should_use_small_k = [&](int c_ncols_dst) {
        return mmvq_should_use_small_k<type>(cc, table_id, blocks_per_row_x, blocks_per_iter_1warp, c_ncols_dst);
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    // on RDNA4 every column takes the ncols_dst=1 K split, see calc_nwarps
    const bool multi_col_small_k = table_id == MMVQ_PARAMETERS_RDNA4 && should_use_small_k(1);
    const auto launch_multi_col = [&](auto ncols_dst_tag) {
        constexpr int c_ncols_dst = decltype(ncols_dst_tag)::value;
        const auto launch = [&](auto small_k_tag, auto rows_tag, auto warps_tag) {
            constexpr bool c_small_k = decltype(small_k_tag)::value;
            constexpr int  c_rows    = decltype(rows_tag)::value;
            constexpr int  c_warps   = decltype(warps_tag)::value;

            const auto run = [&](auto rows_run_tag, auto warps_run_tag, float * dst_run, const mmvq_row_segments_args * segments_run) {
                constexpr int c_rows_run  = decltype(rows_run_tag)::value;
                constexpr int c_warps_run = decltype(warps_run_tag)::value;
                mmvq_row_segments_args segments_launch;
                if (segments_run) {
                    segments_launch = *segments_run;
                }
                const int rows_per_block = c_rows_run > 0 ? c_rows_run : calc_rows_per_block(c_ncols_dst, table_id, c_small_k,
                    c_warps_run > 0 ? c_warps_run : calc_nwarps(type, c_ncols_dst, table_id, c_small_k, false));
                const int nrows_launch = segments_run ? mmvq_place_row_segments(segments_launch, rows_per_block) : nrows_x;
                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_launch, nchannels_dst, nsamples_dst, warp_size, table_id, c_small_k, false, c_rows_run, c_warps_run);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, false, c_rows_run, c_warps_run>(vx, vy, ids, fusion, dst_run, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                     dims.first, dims.second, 0, ids_stride, stream, segments_run ? &segments_launch : nullptr);
            };

            [[maybe_unused]] const auto count_differing_bits_vs_default = [&] {
                return mmvq_count_differing_bits_vs_reference([&](float * dst_ref, const mmvq_row_segments_args * segments_ref) {
                        run(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}, dst_ref, segments_ref);
                    }, dst, segments, c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, stride_col_dst, stride_channel_dst, stride_sample_dst, stream);
            };

            run(rows_tag, warps_tag, dst, segments);

            if constexpr (c_rows > 0) {
                if (ggml_cuda_mmvq_multi_rows_check_enabled()) {
                    mmvq_multi_rows_check_report(type, ncols_x, nrows_x, c_ncols_dst, nchannels_dst, nsamples_dst, c_rows, count_differing_bits_vs_default());
                }
            }
            if constexpr (c_warps > 0) {
                if (ggml_cuda_mmvq_trim_warps_check_enabled()) {
                    mmvq_trim_warps_check_report(type, ncols_x, nrows_x, c_ncols_dst, nchannels_dst, nsamples_dst, count_differing_bits_vs_default());
                }
            }
        };

        if (multi_col_small_k) {
            launch(std::true_type{}, std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{});
            return;
        }
        if constexpr (mmvq_trim_warps_supported(type) && c_ncols_dst <= 4) {
            if (trim_warps) {
                launch(std::false_type{}, std::integral_constant<int, 0>{}, std::integral_constant<int, mmvq_trim_warps>{});
                return;
            }
        }
        if constexpr (mmvq_multi_rows_supported(type) && c_ncols_dst >= 2 && c_ncols_dst <= 4) {
            const bool all_types = ggml_cuda_mmvq_multi_rows_all_types();
            const bool allowed = all_types || (mmvq_multi_rows_default_type(type) && !segments);
            const int rows = allowed ? ggml_cuda_mmvq_multi_rows() : 1;
            const bool divisible = segments ?
                std::all_of(segments->nrows, segments->nrows + segments->n, [&](uint32_t n) { return n % rows == 0; }) : nrows_x % rows == 0;
            const int64_t total_rows = segments ? std::accumulate(segments->nrows, segments->nrows + segments->n, (int64_t) 0) : (int64_t) nrows_x;
            const bool enough_rows = total_rows >= ggml_cuda_mmvq_multi_rows_min_rows();
            if (table_id == MMVQ_PARAMETERS_RDNA4 && divisible && enough_rows && rows == 4) {
                launch(std::false_type{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 0>{});
                return;
            }
            if (table_id == MMVQ_PARAMETERS_RDNA4 && divisible && enough_rows && rows == 2) {
                launch(std::false_type{}, std::integral_constant<int, 2>{}, std::integral_constant<int, 0>{});
                return;
            }
        }
        launch(std::false_type{}, std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{});
    };

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag, auto warps_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;
                constexpr int  c_warps       = decltype(warps_tag)::value;

                const auto run = [&](auto warps_run_tag, float * dst_run, const mmvq_row_segments_args * segments_run) {
                    constexpr int c_warps_run = decltype(warps_run_tag)::value;
                    mmvq_row_segments_args segments_launch;
                    if (segments_run) {
                        segments_launch = *segments_run;
                    }
                    const int nrows_launch = segments_run ?
                        mmvq_place_row_segments(segments_launch, calc_rows_per_block(c_ncols_dst, table_id, c_small_k,
                            c_warps_run > 0 ? c_warps_run : calc_nwarps(type, c_ncols_dst, table_id, c_small_k, c_halve_iters))) : nrows_x;
                    const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_launch, nchannels_dst + (fusion.shared_up != nullptr),
                                                                                  nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters, 0, c_warps_run);
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters, 0, c_warps_run>(
                        vx, vy, ids, fusion, dst_run, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                        stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                        stream, segments_run ? &segments_launch : nullptr);
                };

                run(warps_tag, dst, segments);

                if constexpr (c_warps > 0) {
                    if (ggml_cuda_mmvq_trim_warps_check_enabled()) {
                        const int64_t n_diff = mmvq_count_differing_bits_vs_reference([&](float * dst_ref, const mmvq_row_segments_args * segments_ref) {
                                run(std::integral_constant<int, 0>{}, dst_ref, segments_ref);
                            }, dst, segments, c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, stride_col_dst, stride_channel_dst, stride_sample_dst, stream);
                        mmvq_trim_warps_check_report(type, ncols_x, nrows_x, c_ncols_dst, nchannels_dst, nsamples_dst, n_diff);
                    }
                }
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{}, std::integral_constant<int, 0>{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{}, std::integral_constant<int, 0>{});
            } else {
                if constexpr (mmvq_trim_warps_supported(type)) {
                    if (trim_warps) {
                        launch(std::false_type{}, std::false_type{}, std::integral_constant<int, mmvq_trim_warps>{});
                        break;
                    }
                }
                launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 0>{});
            }
        } break;
        case 2:
            launch_multi_col(std::integral_constant<int, 2>{});
            break;
        case 3:
            launch_multi_col(std::integral_constant<int, 3>{});
            break;
        case 4:
            launch_multi_col(std::integral_constant<int, 4>{});
            break;
        case 5:
            launch_multi_col(std::integral_constant<int, 5>{});
            break;
        case 6:
            launch_multi_col(std::integral_constant<int, 6>{});
            break;
        case 7:
            launch_multi_col(std::integral_constant<int, 7>{});
            break;
        case 8:
            launch_multi_col(std::integral_constant<int, 8>{});
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mmvq_row_segments_args segments;

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1 || (fusion->n_row_segments > 0 && dst->ne[1] <= MMVQ_MAX_ROW_SEGMENT_COLS));
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;

        if (fusion->n_row_segments > 0) {
            GGML_ASSERT(!ids && !fusion->x_bias && !fusion->gate && mmvq_row_segments_supported(src0->type));
            GGML_ASSERT(fusion->row_segments[0].mm == dst);
            segments.n = fusion->n_row_segments;
            for (int s = 0; s < fusion->n_row_segments; ++s) {
                const ggml_cuda_mmvq_row_segment & segment = fusion->row_segments[s];
                const ggml_tensor * mm = segment.mm;
                GGML_ASSERT(mm->src[1] == src1 && mm->src[0]->type == src0->type && mm->src[0]->ne[0] == ne00 &&
                            mm->src[0]->nb[1] == nb01 && ggml_nrows(mm) == ggml_nrows(dst) && ggml_is_contiguous(mm));
                GGML_ASSERT(ggml_nelements(segment.out) == ggml_nelements(mm) && ggml_is_contiguous(segment.out));
                segments.x[s]        = mm->src[0]->data;
                segments.dst[s]      = (float *) segment.out->data;
                segments.nrows[s]    = mm->ne[0];
                segments.epilogue[s] = segment.epilogue;
                if (segment.epilogue == MMVQ_ROW_EPILOGUE_SOFTPLUS_BIAS_SCALE) {
                    GGML_ASSERT(ggml_nelements(segment.bias)  == mm->ne[0] && ggml_is_contiguous(segment.bias));
                    GGML_ASSERT(ggml_nelements(segment.scale) == mm->ne[0] && ggml_is_contiguous(segment.scale));
                    segments.bias[s]  = (const float *) segment.bias->data;
                    segments.scale[s] = (const float *) segment.scale->data;
                }
            }
        }
    }

    const ggml_tensor * x_node    = fusion ? fusion->x_node    : dst;
    const ggml_tensor * gate_node = fusion ? fusion->gate_node : nullptr;
    GGML_ASSERT(!ids || (x_node && x_node->src[0] == src0));
    if (const void * x_host = ids ? ggml_cuda_mmid_host_experts(x_node) : nullptr) {
        fusion_local.x_host         = x_host;
        fusion_local.expert_slot    = (const int32_t *) x_node->src[3]->data;
        fusion_local.n_expert_slots = ggml_get_op_params_i32(x_node, 0);
        if (fusion_local.gate) {
            GGML_ASSERT(gate_node && gate_node->src[3] == x_node->src[3] && ggml_get_op_params_i32(gate_node, 0) == fusion_local.n_expert_slots);
            fusion_local.gate_host = ggml_cuda_mmid_host_experts(gate_node);
            GGML_ASSERT(fusion_local.gate_host);
        }
    } else {
        GGML_ASSERT(!ids || !ggml_cuda_mmid_host_experts(gate_node));
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_nbytes = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;

    // a src1 that is not the node's own operand may be a stack copy whose address repeats with other data;
    // the fused paths pass the graph operands of the fused nodes
    auto & reuse = ctx.q8_1_reuse;
    const bool reusable = reuse.enabled && ctx.curr_stream_no == 0 && q8_1_nbytes <= reuse.size &&
        (fusion != nullptr || (dst->src[1] == src1 && dst->src[0] == src0 && (!ids || dst->src[2] == ids)));

    ggml_cuda_pool_alloc<char> src1_q8_1_alloc(ctx.pool());
    const auto * cached = reusable ? reuse.find(src1, q8_1_nbytes) : nullptr;
    char * src1_q8_1 = (char *) (cached ? cached->buf : reusable ? reuse.claim(src1, q8_1_nbytes)->buf : src1_q8_1_alloc.alloc(q8_1_nbytes));
    if (!cached) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    } else if (ggml_cuda_q8_1_preq_check_enabled()) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        ggml_cuda_pool_alloc<char> fresh_q8_1(ctx.pool(), q8_1_nbytes);
        quantize_row_q8_1_cuda(src1_d, nullptr, fresh_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        mmvq_q8_1_preq_check(src1, src1_q8_1, fresh_q8_1.get(), ne10, ne10_padded, ne11*ne12*ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    if (segments.n > 0) {
        const auto launch_segments = [&](auto type_tag) {
            mul_mat_vec_q_switch_ncols_dst<decltype(type_tag)::value>(
                src0->data, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
                ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,               ids_stride, stream, &segments);
        };
        static std::atomic<bool> iq4_xs_logged{false};
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                launch_segments(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});
                break;
            case GGML_TYPE_IQ4_XS:
                if (!iq4_xs_logged.exchange(true)) {
                    GGML_LOG_WARN("ggml_cuda: row segments: IQ4_XS matvecs fused\n");
                }
                launch_segments(std::integral_constant<ggml_type, GGML_TYPE_IQ4_XS>{});
                break;
            default:
                GGML_ABORT("unsupported type for row segments: %s", ggml_type_name(src0->type));
        }
        return;
    }

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

static bool ggml_cuda_hc_up_pre_fusion() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_HC_UP_PRE_FUSION");
        const bool on = env == nullptr || std::atoi(env) != 0;
        GGML_LOG_WARN("ggml_cuda: hc up+pre fusion: %s\n", on ? "on" : "off");
        return on;
    }();
    return enabled;
}

bool ggml_cuda_hc_up_pre_check_enabled() {
    static const bool enabled = getenv("GGML_CUDA_HC_UP_PRE_CHECK") != nullptr && std::atoi(getenv("GGML_CUDA_HC_UP_PRE_CHECK"));
    return enabled;
}

bool ggml_cuda_mmvq_hc_up_pre_supported(const ggml_tensor * w_up, const int64_t ncols_dst, const int cc) {
    constexpr ggml_type type = GGML_TYPE_IQ4_NL;
    if (!ggml_cuda_hc_up_pre_fusion() || w_up->type != type || ncols_dst < 1 || ncols_dst > 4) {
        return false;
    }

    const int64_t ncols_x = w_up->ne[0];
    if (ncols_x % QK8_1 != 0 || ncols_x > mmvq_hc_up_pre_max_k || w_up->ne[2] != 1 || w_up->ne[3] != 1 ||
            w_up->nb[0] != ggml_type_size(type) || w_up->nb[1] % ggml_type_size(type) != 0) {
        return false;
    }

    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    const int blocks_per_row_x      = ncols_x / qk;
    const int blocks_per_iter_1warp = vdr * warp_size / qi;

    return table_id == MMVQ_PARAMETERS_RDNA4 && warp_size == 32 && blocks_per_row_x <= blocks_per_iter_1warp &&
        mmvq_should_use_small_k<type>(cc, table_id, blocks_per_row_x, blocks_per_iter_1warp, 1);
}

bool ggml_cuda_hc_up_pre_supported(const ggml_tensor * scale_node, const ggml_tensor * mm_node, const ggml_tensor * pre_node, const int cc) {
    const ggml_tensor * lo = scale_node->src[0];
    const ggml_tensor * xn = pre_node->src[0];

    const int64_t n_embd = xn->ne[0];
    const int64_t hc     = xn->ne[1];
    const int64_t nt     = xn->ne[2];

    return ggml_cuda_mmvq_hc_up_pre_supported(mm_node->src[0], nt, cc) &&
        n_embd % mmvq_hc_up_pre_outputs == 0 && hc >= 1 && hc <= mmvq_hc_up_pre_max_hc && xn->ne[3] == 1 && xn->type == GGML_TYPE_F32 &&
        pre_node->type == GGML_TYPE_F32 && ggml_is_contiguous(pre_node) && pre_node->ne[0] == n_embd && pre_node->ne[1] == nt &&
        lo->type == GGML_TYPE_F32 && ggml_is_contiguous(lo) && lo->ne[0] == mm_node->src[0]->ne[0] && lo->ne[1] == nt && lo->ne[2] == 1 && lo->ne[3] == 1 &&
        mm_node->type == GGML_TYPE_F32 && ggml_is_contiguous(mm_node) && mm_node->ne[0] == n_embd*hc && mm_node->ne[1] == nt;
}

template <int c_ncols_dst>
static void mmvq_hc_up_pre_launch(
        const float * lo, const void * vx, const float * xn, float * dst, block_q8_1 * dst_q8_1,
        const float lo_scale, const float lo_bias, const float pre_scale,
        const int ncols_x, const int n_embd, const int hc, const int stride_row_x,
        const int64_t sx0, const int64_t sx1, const int64_t sx2, const int64_t sd0, const int64_t sd1, cudaStream_t stream) {
    const dim3 block_nums(n_embd / mmvq_hc_up_pre_outputs, 1, 1);
    const dim3 block_dims(32, mmvq_hc_up_pre_warps, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_hc_up_pre<GGML_TYPE_IQ4_NL, c_ncols_dst>, launch_params,
        lo, vx, xn, dst, dst_q8_1, lo_scale, lo_bias, pre_scale, ncols_x, n_embd, hc, stride_row_x, sx0, sx1, sx2, sd0, sd1);
}

// runs the unfused SCALE + SILU, hc up matvec and gated hc_pre into scratch buffers and compares them with the fused outputs
static void mmvq_hc_up_pre_check(ggml_backend_cuda_context & ctx, const ggml_tensor * scale_node, const ggml_tensor * mm_node,
        const ggml_tensor * pre_node, const block_q8_1 * fused_q8_1) {
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * lo    = scale_node->src[0];
    const ggml_tensor * w_up  = mm_node->src[0];
    const ggml_tensor * xn    = pre_node->src[0];
    const ggml_tensor * gate  = pre_node->src[1];

    const int64_t ncols_x = w_up->ne[0];
    const int64_t hc_dim  = mm_node->ne[0];
    const int64_t n_embd  = xn->ne[0];
    const int64_t hc      = xn->ne[1];
    const int64_t nt      = xn->ne[2];

    const int64_t ne10_padded = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    const size_t  silu_q8_1_bytes = nt*ne10_padded/QK8_1*sizeof(block_q8_1);
    const size_t  pre_q8_1_bytes  = nt*n_embd/QK8_1*sizeof(block_q8_1);
    const size_t  ts_src0 = ggml_type_size(w_up->type);

    float * silu_ref = nullptr;
    float * gate_ref = nullptr;
    float * pre_ref  = nullptr;
    char  * silu_q8_1_ref = nullptr;
    char  * pre_q8_1_ref  = nullptr;
    CUDA_CHECK(cudaMalloc(&silu_ref, nt*ncols_x*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gate_ref, nt*hc_dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pre_ref,  nt*n_embd*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&silu_q8_1_ref, silu_q8_1_bytes));
    CUDA_CHECK(cudaMalloc(&pre_q8_1_ref,  pre_q8_1_bytes));

    const float * scale = (const float *) scale_node->op_params;
    ggml_cuda_scale_silu_raw((const float *) lo->data, silu_ref, (block_q8_1 *) silu_q8_1_ref, scale[0], scale[1],
        (int) (nt*ncols_x), ncols_x, ne10_padded, stream);

    const int64_t s01 = w_up->nb[1] / ts_src0;
    const int64_t s02 = w_up->nb[2] / ts_src0;
    const int64_t s03 = w_up->nb[3] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s12 = nt*s11;
    const int64_t s13 = s12;
    const int64_t s1  = mm_node->nb[1] / sizeof(float);
    const int64_t s2  = mm_node->nb[2] / sizeof(float);
    const int64_t s3  = mm_node->nb[3] / sizeof(float);
    mul_mat_vec_q_switch_type(
        w_up->data, w_up->type, silu_q8_1_ref, nullptr, ggml_cuda_mm_fusion_args_device{}, gate_ref, ncols_x,
        hc_dim,        nt,  s01, s11, s1,
        1, 1, 1,       s02, s12, s2,
        1, 1,          s03, s13, s3, 0, stream);

    ggml_cuda_dsv4_hc_pre_gated_raw((const float *) xn->data, gate_ref, pre_ref, (block_q8_1 *) pre_q8_1_ref, n_embd, hc, nt,
        xn->nb[0] / sizeof(float), xn->nb[1] / sizeof(float), xn->nb[2] / sizeof(float),
        gate->nb[0] / sizeof(float), gate->nb[1] / sizeof(float), gate->nb[2] / sizeof(float),
        pre_node->nb[0] / sizeof(float), pre_node->nb[1] / sizeof(float), ggml_get_op_params_f32(pre_node, 0), stream);

    const int64_t n_diff = mmvq_count_differing_bits((const float *) pre_node->data, pre_ref, nt*n_embd, stream);

    int64_t n_diff_q8_1 = 0;
    if (fused_q8_1 != nullptr) {
        std::vector<char> host_fused(pre_q8_1_bytes);
        std::vector<char> host_ref(pre_q8_1_bytes);
        CUDA_CHECK(cudaMemcpyAsync(host_fused.data(), fused_q8_1,   pre_q8_1_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_ref.data(),   pre_q8_1_ref, pre_q8_1_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        n_diff_q8_1 = std::inner_product(host_fused.begin(), host_fused.end(), host_ref.begin(), (int64_t) 0,
            std::plus<int64_t>(), std::not_equal_to<char>());
    }

    CUDA_CHECK(cudaFree(silu_ref));
    CUDA_CHECK(cudaFree(gate_ref));
    CUDA_CHECK(cudaFree(pre_ref));
    CUDA_CHECK(cudaFree(silu_q8_1_ref));
    CUDA_CHECK(cudaFree(pre_q8_1_ref));

    static std::atomic<int64_t> n_checked{0};
    static std::atomic<int64_t> n_checked_ncols[5];
    const int64_t n_seen = n_checked.fetch_add(1) + 1;
    const bool first_of_ncols = n_checked_ncols[nt].fetch_add(1) == 0;
    if (n_diff != 0 || n_diff_q8_1 != 0 || n_seen % 1000 == 1 || first_of_ncols) {
        GGML_LOG_WARN("hc_up_pre_check: ncols=%d K=%d n_embd=%d %" PRId64 " values differ, %" PRId64 " q8_1 bytes differ (%" PRId64 " checked)\n",
                (int) nt, (int) ncols_x, (int) n_embd, n_diff, n_diff_q8_1, n_seen);
    }
    GGML_ASSERT(n_diff == 0 && n_diff_q8_1 == 0);
}

void ggml_cuda_op_mul_mat_vec_q_hc_up_pre(ggml_backend_cuda_context & ctx,
        const ggml_tensor * scale_node, const ggml_tensor * mm_node, ggml_tensor * pre_node) {
    const ggml_tensor * lo   = scale_node->src[0];
    const ggml_tensor * w_up = mm_node->src[0];
    const ggml_tensor * xn   = pre_node->src[0];

    const int64_t ncols_x = w_up->ne[0];
    const int64_t n_embd  = xn->ne[0];
    const int64_t hc      = xn->ne[1];
    const int64_t nt      = xn->ne[2];

    const float * scale     = (const float *) scale_node->op_params;
    const float   pre_scale = ggml_get_op_params_f32(pre_node, 0);

    block_q8_1 * dst_q8_1 = ctx.q8_1_prequantize_dst(pre_node);

    const auto launch = [&](auto ncols_tag) {
        mmvq_hc_up_pre_launch<decltype(ncols_tag)::value>(
            (const float *) lo->data, w_up->data, (const float *) xn->data, (float *) pre_node->data, dst_q8_1,
            scale[0], scale[1], pre_scale, ncols_x, n_embd, hc, w_up->nb[1] / ggml_type_size(w_up->type),
            xn->nb[0] / sizeof(float), xn->nb[1] / sizeof(float), xn->nb[2] / sizeof(float),
            pre_node->nb[0] / sizeof(float), pre_node->nb[1] / sizeof(float), ctx.stream());
    };

    switch (nt) {
        case 1:
            launch(std::integral_constant<int, 1>{});
            break;
        case 2:
            launch(std::integral_constant<int, 2>{});
            break;
        case 3:
            launch(std::integral_constant<int, 3>{});
            break;
        case 4:
            launch(std::integral_constant<int, 4>{});
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    if (ggml_cuda_hc_up_pre_check_enabled()) {
        mmvq_hc_up_pre_check(ctx, scale_node, mm_node, pre_node, dst_q8_1);
    }
}
