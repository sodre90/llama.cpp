#pragma once

// Host row cache for the per-layer token embedding tables that a model reads lazily
// (qwen4exp PLE, gemma3n, gemma4). The n-gram hash spreads its rows evenly over the
// whole table, so one 120-byte row costs a whole 4 KiB page from the model file. The
// cache keeps a part of the rows in plain host memory: a hit is served from RAM and
// never faults the mapping.

#include <cstdint>

struct ggml_tensor;
struct llama_model;

// table is model.per_layer_tok_embd; n_rows <= 0, or a model without such a table,
// leaves the cache off
void llama_ple_rows_init(const llama_model & model, int32_t n_rows);

// true when the cache is on, so the graph takes the host gather instead of ggml_get_rows
bool llama_ple_rows_enabled();

// gather n_rows rows and dequantize them to f32, as ggml_get_rows does; a hit comes from the cache
void llama_ple_rows_gather(const ggml_tensor * table, const int32_t * idx, int64_t n_rows, float * dst);
