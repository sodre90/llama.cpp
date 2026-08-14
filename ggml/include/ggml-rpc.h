#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Bumped for the recursive-doubling mesh allreduce. The wire format is unchanged, but the exchange
// SCHEDULE is not: a tree shard expects log2(N) paired exchanges where an all-gather shard sends
// N-1 unsolicited partials, so a half-updated fleet does not disagree, it hangs mid-token with no
// error. The minor check only rejects a server NEWER than the client, which would let exactly the
// dangerous pairing through; major is compared for inequality and catches both directions.
#define RPC_PROTO_MAJOR_VERSION    9
// Bumped for RPC_SET_TENSOR_HASH_NO_CACHE, which a client older than this reads as a cache hit and
// answers by not sending the tensor at all. That is silent: the weights above HASH_THRESHOLD simply
// never arrive and the model generates fluent nonsense. negotiate_hello rejects a server whose minor
// exceeds the client's, so the bump turns a half-updated fleet into a failed handshake. The reverse
// pairing stays fine, since a newer client reads an older server's MISS and sends the tensor.
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, uint32_t poll,
                                                    size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
