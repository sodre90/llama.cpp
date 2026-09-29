#include "argsort.cuh"
#include "top-k.cuh"

#include <atomic>
#include <cstdlib>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(CUB_TOP_K_AVAILABLE) && (defined(GGML_CUDA_USE_CUB) || defined(GGML_USE_HIP))

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

// Derives the state after pass q (1..4) from the state after pass q-1 and the pass q block histograms.
// Same result as the serial walk down from the top bin: the chosen bin is the highest one whose suffix sum reaches the rank.
template<int BLOCK_SIZE, int RADIX_BITS>
static __device__ __forceinline__ top_k_radix_state top_k_radix_derive(
        top_k_radix_state * __restrict__ states,
        const int * __restrict__ block_histograms,
        int nrows,
        int k,
        int blocks_per_row,
        int row,
        int row_block,
        int q) {
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(BLOCK_SIZE == NBINS, "one thread per bin");

    const int tid = threadIdx.x;
    const int shift = 32 - RADIX_BITS * q;
    top_k_radix_state state = q == 1 ? top_k_radix_state{0, 0, k} : states[(size_t) (q - 1) * nrows + row];

    __shared__ int suffix[NBINS];
    __shared__ int chosen_bin;

    int count = 0;
#pragma unroll 8
    for (int b = 0; b < blocks_per_row; ++b) {
        count += block_histograms[((size_t) row * blocks_per_row + b) * NBINS + tid];
    }
    suffix[tid] = count;
    if (tid == 0) {
        chosen_bin = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < NBINS; offset *= 2) {
        const int above = tid + offset < NBINS ? suffix[tid + offset] : 0;
        __syncthreads();
        suffix[tid] += above;
        __syncthreads();
    }

    if (suffix[tid] >= state.rank && (tid == NBINS - 1 || suffix[tid + 1] < state.rank)) {
        chosen_bin = tid;
    }
    __syncthreads();

    const int bin = chosen_bin;
    state.rank -= bin == NBINS - 1 ? 0 : suffix[bin + 1];
    state.prefix |= (uint32_t) bin << shift;
    state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;

    if (row_block == 0 && tid == 0) {
        states[(size_t) q * nrows + row] = state;
    }
    return state;
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram_fused(
        const float * __restrict__ src,
        top_k_radix_state * __restrict__ states,
        const int * __restrict__ prev_histograms,
        int * __restrict__ block_histograms,
        int ncols,
        int nrows,
        int k,
        int blocks_per_row,
        int pass) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int shift = 32 - RADIX_BITS * pass;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    const top_k_radix_state state = pass == 1 ?
        top_k_radix_state{0, 0, k} :
        top_k_radix_derive<BLOCK_SIZE, RADIX_BITS>(states, prev_histograms, nrows, k, blocks_per_row, row, row_block, pass - 1);

    histogram[tid] = 0;
    __syncthreads();

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

// greater counts in the low half of the int, ties in the high half: one scan serves both
static __device__ __forceinline__ int top_k_radix_classify(
        const float * __restrict__ row_src, const top_k_radix_state & state, int col, int col_end) {
    if (col >= col_end) {
        return 0;
    }
    const uint32_t key = top_k_float_to_ordered(row_src[col]);
    return key > state.prefix ? 1 : key == state.prefix ? 1 << 16 : 0;
}

template<int BLOCK_SIZE, bool FUSED>
static __global__ void top_k_radix_count(
        const float * __restrict__ src,
        top_k_radix_state * __restrict__ states,
        const int * __restrict__ last_histograms,
        int * __restrict__ block_counts,
        int ncols,
        int nrows,
        int k,
        int cols_per_block,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const float * row_src = src + (size_t) row * ncols;
    top_k_radix_state state;
    if (FUSED) {
        state = top_k_radix_derive<BLOCK_SIZE, 8>(states, last_histograms, nrows, k, blocks_per_row, row, row_block, 4);
    } else {
        state = states[row];
    }
    const int col_begin = row_block * cols_per_block;
    const int col_end = min(col_begin + cols_per_block, ncols);

    __shared__ int greater;
    __shared__ int equal;
    if (threadIdx.x == 0) {
        greater = 0;
        equal = 0;
    }
    __syncthreads();

    int local_greater = 0;
    int local_equal = 0;
    for (int col = col_begin + threadIdx.x; col < col_end; col += BLOCK_SIZE) {
        const int counts = top_k_radix_classify(row_src, state, col, col_end);
        local_greater += counts & 0xFFFF;
        local_equal += counts >> 16;
    }
    atomicAdd(&greater, local_greater);
    atomicAdd(&equal, local_equal);
    __syncthreads();

    if (threadIdx.x == 0) {
        block_counts[2 * blockIdx.x + 0] = greater;
        block_counts[2 * blockIdx.x + 1] = equal;
    }
}

// The winners are written in column order, the keys above the threshold first and the ties with it
// last, so the output is the same on every run and its tail holds the lowest-scoring winners.
// Callers rely on both: summation order follows the list, and a list cut short drops its tail.
template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int * __restrict__ block_counts,
        int ncols,
        int k,
        int cols_per_block,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const top_k_radix_state state = states[row];
    const int col_begin = row_block * cols_per_block;
    const int col_end = min(col_begin + cols_per_block, ncols);

    int n_greater = 0;
    int n_equal = 0;
    for (int b = 0; b < row_block; ++b) {
        n_greater += block_counts[2 * (row * blocks_per_row + b) + 0];
        n_equal += block_counts[2 * (row * blocks_per_row + b) + 1];
    }

    __shared__ int scan[BLOCK_SIZE];

    for (int c0 = col_begin; c0 < col_end && (n_greater < k - state.rank || n_equal < state.rank); c0 += BLOCK_SIZE) {
        const int col = c0 + tid;
        const int counts = top_k_radix_classify(row_src, state, col, col_end);
        scan[tid] = counts;
        __syncthreads();
        for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
            const int preceding = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += preceding;
            __syncthreads();
        }
        if (counts & 0xFFFF) {
            row_dst[n_greater + (scan[tid] & 0xFFFF) - 1] = col;
        } else if (counts) {
            const int pos = n_equal + (scan[tid] >> 16) - 1;
            if (pos < state.rank) {
                row_dst[k - state.rank + pos] = col;
            }
        }
        const int total = scan[BLOCK_SIZE - 1];
        n_greater += total & 0xFFFF;
        n_equal += total >> 16;
        __syncthreads();
    }
}

static bool top_k_fused_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_TOPK_FUSED");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    const int cols_per_block = (ncols + blocks_per_row - 1) / blocks_per_row;
    const dim3 row_grid(blocks_per_row * nrows);
    ggml_cuda_pool_alloc<int> block_counts_alloc(pool, (size_t) nrows * blocks_per_row * 2);
    int * block_counts = block_counts_alloc.get();

    if (top_k_fused_enabled()) {
        static std::atomic<bool> fused_logged{false};
        if (!fused_logged.exchange(true)) {
            GGML_LOG_WARN("ggml_cuda: top-k: fused radix passes\n");
        }

        const size_t histogram_size = (size_t) nrows * blocks_per_row * NBINS;
        ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, 5 * (size_t) nrows);
        ggml_cuda_pool_alloc<int> histograms_alloc(pool, 2 * histogram_size);
        top_k_radix_state * states = states_alloc.get();
        int * histograms = histograms_alloc.get();

        for (int pass = 1; pass <= 4; ++pass) {
            top_k_radix_histogram_fused<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms + ((pass - 1) & 1) * histogram_size, histograms + (pass & 1) * histogram_size,
                    ncols, nrows, k, blocks_per_row, pass);
        }
        top_k_radix_count<BLOCK_SIZE, true>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, block_counts, ncols, nrows, k, cols_per_block, blocks_per_row);
        top_k_radix_gather<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states + 4 * (size_t) nrows, block_counts, ncols, k, cols_per_block, blocks_per_row);
        return;
    }

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_count<BLOCK_SIZE, false>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, states, nullptr, block_counts, ncols, nrows, k, cols_per_block, blocks_per_row);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, block_counts, ncols, k, cols_per_block, blocks_per_row);
}

#endif // !defined(CUB_TOP_K_AVAILABLE) && (defined(GGML_CUDA_USE_CUB) || defined(GGML_USE_HIP))

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // a full sort to keep k of ncols is wasteful, and under stream capture argsort_f32_i32_cuda_cub
    // has to use DeviceSegmentedRadixSort, which gives each row a single thread block
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    if (shared_mem <= max_shared_mem && ncols <= 1024) {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();

        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
