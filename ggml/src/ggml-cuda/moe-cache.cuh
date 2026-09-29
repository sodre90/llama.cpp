#include "common.cuh"

// picks the victim cache slots of the experts a layer's routed ids miss, see the state layout in moe-cache.cu
void ggml_cuda_moe_cache_assign(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
