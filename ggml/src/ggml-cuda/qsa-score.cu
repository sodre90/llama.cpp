#include "qsa-score.cuh"

#include <atomic>
#include <cinttypes>
#include <functional>
#include <numeric>
#include <vector>

#define CUDA_QSA_SCORE_BLOCK_SIZE 256

struct qsa_score_strided {
    const char * data;
    int64_t      nb0;
    int64_t      nb1;
    int64_t      nb2;

    __device__ __forceinline__ float at(const int64_t i0, const int64_t i1, const int64_t i2) const {
        return *(const float *) (data + i0*nb0 + i1*nb1 + i2*nb2);
    }
};

struct qsa_score_params {
    const float * score;
    float *       dst;
    int64_t       nb;
    int64_t       n_heads;
    int64_t       n_t;
    qsa_score_strided start;
    qsa_score_strided q;
    qsa_score_strided m;
    qsa_score_strided spare;
    qsa_score_strided bias;
    float future_min;
    float future_max;
    float tail_min;
    float tail_max;
    float forced_scale;
    float forced_bias;
    float future_scale;
    float future_bias;
};

// every step rounds on its own, as the RELU, ADD, SUB, CLAMP and SCALE kernels do
template <bool device_rule>
static __global__ void qsa_score_epilogue_kernel(const qsa_score_params p) {
    ggml_cuda_pdl_lc();
    const int64_t b = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t s = blockIdx.z;

    if (b >= p.nb) {
        return;
    }

    ggml_cuda_pdl_sync();
    const float * g = p.score + ((s*p.n_t + t)*p.n_heads)*p.nb + b;

    float sum = __fadd_rn(fmaxf(g[0], 0.0f), fmaxf(g[p.nb], 0.0f));
    for (int64_t h = 2; h < p.n_heads; ++h) {
        sum = __fadd_rn(sum, fmaxf(g[h*p.nb], 0.0f));
    }

    float bias;
    if (device_rule) {
        const float ahead  = __fsub_rn(p.start.at(b, 0, s), p.q.at(0, t, s));
        const float future = fminf(fmaxf(ahead, p.future_min), p.future_max);
        const float tail   = fminf(fmaxf(__fadd_rn(ahead, p.m.at(0, t, s)), p.tail_min), p.tail_max);
        const float forced = __fadd_rn(tail, p.spare.at(b, 0, s));
        const float bias_forced = __fadd_rn(__fmul_rn(p.forced_scale, forced), p.forced_bias);
        const float bias_future = __fadd_rn(__fmul_rn(p.future_scale, future), p.future_bias);
        bias = __fadd_rn(bias_forced, bias_future);
    } else {
        bias = p.bias.at(b, t, s);
    }

    p.dst[(s*p.n_t + t)*p.nb + b] = __fadd_rn(sum, bias);
}

static qsa_score_strided qsa_score_strided_of(const ggml_tensor * t) {
    return { t ? (const char *) t->data : nullptr, t ? (int64_t) t->nb[0] : 0, t ? (int64_t) t->nb[1] : 0, t ? (int64_t) t->nb[2] : 0 };
}

void ggml_cuda_op_qsa_score_epilogue(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_score_epilogue & e, float * dst) {
    const ggml_tensor * score = e.score;

    GGML_ASSERT(score->type == GGML_TYPE_F32 && ggml_is_contiguous(score));
    GGML_ASSERT(e.fin->type == GGML_TYPE_F32 && ggml_is_contiguous(e.fin));
    GGML_ASSERT(score->ne[1] >= 2);

    qsa_score_params p = {};
    p.score   = (const float *) score->data;
    p.dst     = dst;
    p.nb      = score->ne[0];
    p.n_heads = score->ne[1];
    p.n_t     = score->ne[2];
    p.start   = qsa_score_strided_of(e.start);
    p.q       = qsa_score_strided_of(e.q);
    p.m       = qsa_score_strided_of(e.m);
    p.spare   = qsa_score_strided_of(e.spare);
    p.bias    = qsa_score_strided_of(e.bias);
    p.future_min   = e.future_min;
    p.future_max   = e.future_max;
    p.tail_min     = e.tail_min;
    p.tail_max     = e.tail_max;
    p.forced_scale = e.forced_scale;
    p.forced_bias  = e.forced_bias;
    p.future_scale = e.future_scale;
    p.future_bias  = e.future_bias;

    const int64_t ns = score->ne[3];
    const dim3 grid((unsigned) ((p.nb + CUDA_QSA_SCORE_BLOCK_SIZE - 1) / CUDA_QSA_SCORE_BLOCK_SIZE), (unsigned) p.n_t, (unsigned) ns);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, CUDA_QSA_SCORE_BLOCK_SIZE, 0, ctx.stream());
    if (e.device_rule()) {
        ggml_cuda_kernel_launch(qsa_score_epilogue_kernel<true>, launch_params, p);
    } else {
        ggml_cuda_kernel_launch(qsa_score_epilogue_kernel<false>, launch_params, p);
    }
}

bool ggml_cuda_qsa_score_fusion_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_QSA_SCORE_FUSION");
        const bool on = env == nullptr || std::atoi(env) != 0;
        GGML_LOG_WARN("ggml_cuda: qsa score fusion: %s\n", on ? "on" : "off");
        return on;
    }();
    return enabled;
}

bool ggml_cuda_qsa_score_check_enabled() {
    static const bool enabled = getenv("GGML_CUDA_QSA_SCORE_CHECK") != nullptr && std::atoi(getenv("GGML_CUDA_QSA_SCORE_CHECK"));
    return enabled;
}

static int64_t qsa_score_count_differing_bits(const float * a, const float * b, const size_t n, cudaStream_t stream) {
    std::vector<uint32_t> host_a(n);
    std::vector<uint32_t> host_b(n);
    CUDA_CHECK(cudaMemcpyAsync(host_a.data(), a, n*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_b.data(), b, n*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return std::inner_product(host_a.begin(), host_a.end(), host_b.begin(), (int64_t) 0, std::plus<int64_t>(), std::not_equal_to<uint32_t>());
}

void ggml_cuda_qsa_score_check(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_score_epilogue & e, const float * fused) {
    const int64_t n_diff = qsa_score_count_differing_bits((const float *) e.fin->data, fused, ggml_nelements(e.fin), ctx.stream());

    static std::atomic<int64_t> n_checked_device_rule{0};
    static std::atomic<int64_t> n_checked_bias{0};
    const int64_t n_seen = (e.device_rule() ? n_checked_device_rule : n_checked_bias).fetch_add(1) + 1;
    if (n_diff != 0 || n_seen % 1000 == 1) {
        GGML_LOG_WARN("qsa_score_check: mode %c nb=%d n_t=%d ns=%d %" PRId64 " values differ (%" PRId64 " checked)\n",
                e.device_rule() ? 'A' : 'B', (int) e.fin->ne[0], (int) e.fin->ne[1], (int) e.fin->ne[2], n_diff, n_seen);
    }
    GGML_ASSERT(n_diff == 0);
}
