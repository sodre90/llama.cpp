#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

// [TAG_QSA_POOLED_CACHE] the pooled rows are written whole by set_rows and read back by get_rows
// (or as a mul_mat operand), never accumulated, so their storage type is free to choose: the
// raw keys they summarise are already q8_0. LLAMA_QSA_POOLED_TYPE=f32|f16|q8_0 (default q8_0).
static ggml_type qsa_pooled_store_type(uint32_t idx_dim) {
    const char * env = getenv("LLAMA_QSA_POOLED_TYPE");
    const std::string want = env ? env : "q8_0";
    ggml_type type = GGML_TYPE_Q8_0;
    if (want == "f32") {
        type = GGML_TYPE_F32;
    } else if (want == "f16") {
        type = GGML_TYPE_F16;
    } else if (want != "q8_0") {
        LLAMA_LOG_WARN("%s: unknown LLAMA_QSA_POOLED_TYPE '%s', using q8_0\n", __func__, want.c_str());
    }
    if (idx_dim % ggml_blck_size(type) != 0) {
        LLAMA_LOG_WARN("%s: indexer head size %u is not a multiple of the %s block, storing pooled keys as f16\n",
                __func__, idx_dim, ggml_type_name(type));
        type = GGML_TYPE_F16;
    }
    return type;
}

// [TAG_QSA_POOLED_CACHE] with the pooled cache on, a raw indexer key is written once and read back
// only while its block is dirty, so the raw cache can live in host memory the device maps
// (LLAMA_QSA_IDX_HOST=1). The full recompute reads every cell every ubatch and would stream the
// whole cache over the bus, so LLAMA_QSA_NO_POOLED_CACHE wins.
static bool qsa_idx_cache_on_host() {
    if (getenv("LLAMA_QSA_IDX_HOST") == nullptr) {
        return false;
    }
    if (getenv("LLAMA_QSA_NO_POOLED_CACHE") != nullptr) {
        LLAMA_LOG_WARN("%s: LLAMA_QSA_IDX_HOST ignored: LLAMA_QSA_NO_POOLED_CACHE reads the whole indexer cache every ubatch\n", __func__);
        return false;
    }
    return true;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx,
    uint32_t kv_unified_per_slot) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // The glm5 next indexer caches key, gate and pooled values per token
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? 3 : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        const bool on_host = offload && qsa_idx_cache_on_host();

        LLAMA_LOG_WARN("%s: creating indexer KV cache, size = %u cells, %s\n", __func__, kv_size,
                on_host ? "in mapped host memory" : "in device memory");

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_", on_host);
    }()) {
    // [TAG_QSA_POOLED_CACHE] one f32 row per position block per layer; multi-stream aware
    if (mem_idx) {
        uint32_t ratio = 0;
        for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
            if (model.hparams.dsv4_compress_ratios[il] > 0) {
                ratio = model.hparams.dsv4_compress_ratios[il];
                break;
            }
        }

        const uint32_t idx_dim        = model.hparams.indexer_head_size;
        const uint32_t n_stream_total = mem_idx->get_n_stream();
        const ggml_type pooled_type   = qsa_pooled_store_type(idx_dim);

        if (ratio > 0 && idx_dim > 0 && n_stream_total > 0) {
            const uint32_t per_seq_limit = (kv_unified_per_slot > 0 && kv_unified_per_slot < kv_size)
                                         ? kv_unified_per_slot
                                         : kv_size;
            // + 1 so a partial trailing block has a slot, + 1 dustbin row for padded writes
            pooled_rows  = per_seq_limit/ratio + 2;
            pooled_ratio = ratio;

            // a unified cache puts every sequence in one stream, so give each its own row range:
            // the position block alone would alias two agents sitting at the same positions
            if (n_stream_total == 1 && n_seq_max > 1) {
                pooled_seq_stride = pooled_rows;
                pooled_rows       = pooled_seq_stride*n_seq_max + 1; // + 1 shared dustbin row
            }

            // one context+buffer per device: the indexer caches of the QSA layers are spread
            // across the layer-split devices, and a row written by a device that does not own
            // it would travel the inter-GPU link every decode step
            std::vector<ggml_backend_buffer_type_t>   bufts;
            std::vector<std::vector<ggml_tensor *>>   per_buf_tensors;

            for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
                // the idx cache is filtered to the QSA layers; get_k_storage on any other layer
                // is out of range
                if (model.hparams.dsv4_compress_ratios[il] == 0) {
                    continue;
                }
                ggml_tensor * k = mem_idx->get_k_storage((int32_t) il);
                if (k == nullptr) {
                    continue;
                }

                // the layer's device memory, not the raw keys' buffer: those may be mapped host memory,
                // and every score reads the whole pooled store
                const ggml_backend_buffer_type_t buft = offload
                    ? ggml_backend_dev_buffer_type(model.dev_layer(il))
                    : ggml_backend_buffer_get_type(k->buffer);

                size_t ci = SIZE_MAX;
                for (size_t j = 0; j < bufts.size(); ++j) {
                    if (bufts[j] == buft) {
                        ci = j;
                        break;
                    }
                }
                if (ci == SIZE_MAX) {
                    ci = bufts.size();
                    bufts.push_back(buft);
                    per_buf_tensors.emplace_back();

                    ggml_init_params ip = {
                        /*.mem_size   =*/ 2*model.hparams.n_layer()*ggml_tensor_overhead(),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    pooled_ctxs.emplace_back(ggml_init(ip));
                }

                ggml_tensor * t = ggml_new_tensor_3d(pooled_ctxs[ci].get(), pooled_type, idx_dim, pooled_rows, n_stream_total);
                ggml_format_name(t, "idx_pooled_l%u", il);
                pooled_k[(int32_t) il] = t;
                per_buf_tensors[ci].push_back(t);
            }

            size_t total_bytes = 0;
            for (size_t ci = 0; ci < bufts.size(); ++ci) {
                pooled_bufs.emplace_back(ggml_backend_alloc_ctx_tensors_from_buft(pooled_ctxs[ci].get(), bufts[ci]));
                GGML_ASSERT(pooled_bufs.back() && "failed to allocate the pooled indexer key cache");
                // stale rows are read (and masked); they must be finite, never uninitialized
                ggml_backend_buffer_clear(pooled_bufs.back().get(), 0);
                total_bytes += ggml_backend_buffer_get_size(pooled_bufs.back().get());
            }

            if (!pooled_k.empty()) {
                LLAMA_LOG_WARN("%s: pooled indexer key cache, %zu layers x %u rows x %u streams on %zu buffers, %s, %.2f MiB\n",
                        __func__, pooled_k.size(), pooled_rows, n_stream_total, pooled_bufs.size(),
                        ggml_type_name(pooled_type), total_bytes/1024.0/1024.0);
            }
        }
    }

    // [TAG_SCHED_ALLOC_DUMP] the memory breakdown reports one "context" total per buffer type;
    // this names its parts
    if (getenv("LLAMA_SCHED_ALLOC_DUMP") != nullptr) {
        auto total_of = [](const std::map<ggml_backend_buffer_type_t, size_t> & mb) {
            size_t total = 0;
            for (const auto & buft_size : mb) {
                total += buft_size.second;
            }
            return total;
        };
        size_t pooled_bytes = 0;
        for (const auto & buf : pooled_bufs) {
            pooled_bytes += ggml_backend_buffer_get_size(buf.get());
        }
        LLAMA_LOG_WARN("%s: context parts: attention kv %zu MiB, recurrent %zu MiB, indexer %zu MiB, pooled keys %zu MiB\n", __func__,
                total_of(get_mem_attn()->memory_breakdown()) >> 20, total_of(get_mem_recr()->memory_breakdown()) >> 20,
                mem_idx ? total_of(mem_idx->memory_breakdown()) >> 20 : (size_t) 0, pooled_bytes >> 20);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // [TAG_QSA_PAD_CELL] set_input_qsa pads the last spare block with an empty cell of the pool, so a
    // batch that fills the pool completely can be placed but aborts the process later, in the graph
    // input. Failing it here instead lets the server halve the batch and, at n_batch 1, spill a slot
    // to host RAM and defer its task until the pool has room (this killed the server 2026-09-19 at
    // 3 x 115-122k prompts against a 280k pool).
    if (get_mem_attn()->get_n_stream() == 1) {
        const uint32_t used = get_mem_attn()->get_used();
        const uint32_t size = get_mem_attn()->get_size();

        if (used + balloc.get_n_tokens() >= size) {
            LLAMA_LOG_WARN("%s: refusing %u tokens into a pool with %u of %u cells used - qsa keeps one empty cell\n",
                    __func__, balloc.get_n_tokens(), used, size);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
    }

    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_reset(-1);
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_rm(seq_id, p0, p1);

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool grouping
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
        // pooled keys, the destination rebuilds its layout and re-pools into the same rows
        mem_idx_stale_set(seq_id_dst, 0);
    }

    // [TAG_QSA_POOLED_CACHE] rows are shared in the single-stream cache; the copy's blocks
    // are refilled from its own cells on its first ubatch
    pooled_shared = pooled_shared || pooled_seq_stride > 0;
    pooled_reset(seq_id_dst);
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }

    // [TAG_QSA_POOLED_CACHE] only seq_id's rows survive as trusted, and not even those if it
    // may have been reading a row pooled into a sequence that just went away
    const int64_t keep = pooled_shared || !pooled_w.count(seq_id) ? 0 : pooled_w[seq_id];
    pooled_w.clear();
    pooled_w[seq_id] = keep;
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }

    // [TAG_QSA_POOLED_CACHE] shifting positions remaps every block
    pooled_reset(seq_id);
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_reset(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & buf : pooled_bufs) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                // the restore rewrites the cells behind the pool layout's back
                mem_idx_stale_set(seq_id, 0);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    // [TAG_QSA_POOLED_CACHE] a full restore rewrites the indexer cells with arbitrary
    // content, so no pooled row can be trusted; the next ubatch refills the whole range.
    // A PARTIAL_ONLY restore (speculative checkpoint replay) leaves the cells untouched
    // and its rollback arrives through seq_rm, which already clamped the watermark.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        pooled_reset(seq_id);
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }

    // [TAG_QSA_POOLED_CACHE]
    pooled_reset(seq_id);
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

ggml_tensor * llama_memory_hybrid_idx::get_pooled_k(int32_t il) const {
    const auto it = pooled_k.find(il);
    return it == pooled_k.end() ? nullptr : it->second;
}

int64_t & llama_memory_hybrid_idx::pooled_valid(llama_seq_id seq_id) const {
    return pooled_w[seq_id];
}

void llama_memory_hybrid_idx::pooled_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (pooled_k.empty()) {
        return;
    }

    if (seq_id < 0) {
        if (p0 <= 0 && p1 < 0) {
            pooled_reset(-1);
            return;
        }
        for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
            pooled_rm(sq, p0, p1);
        }
        return;
    }

    if (p0 <= 0 && p1 < 0) {
        pooled_reset(seq_id);
        return;
    }

    // a block whose cells belong to several sequences is pooled once, into the lowest sharer's
    // row, and every sharer reads that row. When sharing ends the block changes owner, so the
    // new owner's own row is stale - repool everything rather than track which blocks moved.
    if (pooled_shared) {
        pooled_w.clear();
        return;
    }

    // blocks at or beyond the first removed position lose members; earlier rows keep their
    // content (removals only ever drop the tail or a middle range, never rewrite the prefix)
    const int64_t blk = pooled_ratio > 0 ? std::max<llama_pos>(p0, 0)/pooled_ratio : 0;

    auto & w = pooled_w[seq_id];
    w = std::min(w, blk);
}

void llama_memory_hybrid_idx::pooled_reset(llama_seq_id seq_id) {
    if (seq_id < 0) {
        pooled_w.clear();
        pooled_shared = false;
    } else {
        pooled_w[seq_id] = 0;
    }
}

// [TAG_QSA_SEQ_SCOPE] the sequences of a ubatch made of one equal run of tokens per sequence, in
// token order; empty for any other ubatch, and for the reserve pass's ubatch, which has no sequences
static std::vector<llama_seq_id> qsa_scope_seqs(const llama_ubatch & ubatch) {
    if (ubatch.n_tokens == 0 || ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr || ubatch.n_seq_id == nullptr) {
        return {};
    }

    std::vector<llama_seq_id> seqs;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1) {
            return {};
        }

        const llama_seq_id sq = ubatch.seq_id[i][0];

        if (i > 0 && sq == ubatch.seq_id[i - 1][0]) {
            continue;
        }
        if (std::find(seqs.begin(), seqs.end(), sq) != seqs.end()) {
            return {};
        }

        seqs.push_back(sq);
    }

    if (ubatch.n_tokens % seqs.size() != 0) {
        return {};
    }

    const uint32_t n_t = ubatch.n_tokens / (uint32_t) seqs.size();

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.seq_id[i][0] != seqs[i/n_t]) {
            return {};
        }
    }

    return seqs;
}

void llama_memory_hybrid_idx::qsa_sync(
        qsa_input_state & st,
        const llama_kv_cells & cells,
        const llama_ubatch * ubatch,
        int64_t n_kv,
        int64_t r,
        int64_t n_seq_vis,
        bool keep,
        bool need_blk_of,
        bool blk_bias) const {
    const int64_t n_blocks = (n_kv + r - 1)/r;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // r is a power of two in practice, and a shift avoids a 64-bit division per cell
    int64_t r_shift = -1;
    for (int64_t b = 0; b < 63; ++b) {
        if ((int64_t(1) << b) == r) {
            r_shift = b;
        }
    }
    auto bucket_of = [&](int64_t idx) { return r_shift >= 0 ? idx >> r_shift : idx/r; };
    auto slot_of   = [&](int64_t idx) { return r_shift >= 0 ? idx & (r - 1) : idx%r; };

    std::vector<llama_seq_id> seq_present;

    for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
        if (cells.seq_pos_min(sq) >= 0) {
            seq_present.push_back(sq);
        }
    }

    const int32_t n_present = (int32_t) seq_present.size();

    // [TAG_QSA_SEQ_SCOPE] the scope query syncs before the graph is built, and set_input_qsa then
    // finds nothing new; a state that cannot be updated in place would otherwise be rebuilt twice
    if (st.synced && st.cells == &cells && st.mark == cells.journal_end() && st.n_kv == n_kv && st.r == r &&
            st.n_seq_vis == n_seq_vis && st.seq_present == seq_present && st.need_blk_of == need_blk_of &&
            st.pos_2d == ubatch->is_pos_2d()) {
        return;
    }

    // [TAG_QSA_CELL_KEY] a cell's sequence set as a small int, so the per-cell passes never
    // compare or scan the 256-bit sets: key k < n_present is the one sequence seq_present[k], a
    // larger key names a set of several sequences in key_set. Equal keys mean equal sets.
    // -2: a set not interned yet
    auto key_of = [&](int64_t j) -> int32_t {
        if (cells.is_empty(j)) {
            return -1;
        }

        // with one sequence in the stream every non-empty cell holds exactly that one
        if (n_present == 1) {
            return 0;
        }

        // a cell holds only present sequences, so testing those decides whether it holds one
        int32_t key  = -1;
        int32_t n_in = 0;

        for (int32_t k = 0; k < n_present && n_in < 2; ++k) {
            if (cells.seq_has((uint32_t) j, seq_present[k])) {
                key = k;
                n_in++;
            }
        }

        if (n_in < 2) {
            return key;
        }

        const auto & set = cells.seq_get_all((uint32_t) j);

        for (size_t k = 0; k < st.key_set.size(); ++k) {
            if (st.key_set[k] == set) {
                return n_present + (int32_t) k;
            }
        }

        return -2;
    };

    // buckets a sequence's cells can reach: a position is at most the largest one, and a rank
    // is below the sequence's cell count. The slack lets a decoding sequence grow in place
    const int64_t ext_slack = 64;

    auto seq_ext = [&](llama_seq_id sq) {
        int64_t top = cells.seq_pos_max(sq);
        if (st.ranked) {
            top = std::max(top, (int64_t) cells.seq_pos_cells(sq).size() - 1);
        }
        return std::min(n_blocks, bucket_of(top) + 1 + ext_slack);
    };

    auto add_key = [&](int64_t ext) {
        st.grp_ext .push_back(ext);
        st.grp_base.push_back(st.grp_base.back() + ext);

        const int64_t n_grp = st.grp_base.back();

        st.grp_first.resize(n_grp, -1);
        st.grp_slot0.resize(n_grp, -1);
        st.grp_slots.resize(n_grp,  0);
    };

    auto new_multi_key = [&](const llama_kv_cells::seq_set_t & set) {
        int64_t ext = 0;
        for (const llama_seq_id sq : seq_present) {
            if (set.test(sq)) {
                ext = std::max(ext, seq_ext(sq));
            }
        }
        st.key_set.push_back(set);
        add_key(ext);
    };

    // the causal index of a cell: its rank under an image, else its position
    auto idx_of = [&](int64_t j) {
        return st.ranked ? (int64_t) st.rank[j] : (int64_t) cells.pos_get(j);
    };

    // group every cell by (key, bucket); with keys == false the mrope path groups again by rank
    auto group_cells = [&](bool keys) {
        st.oor = false;
        st.dup = false;

        st.grp_ext  .clear();
        st.grp_base .assign(1, 0);
        st.grp_first.clear();
        st.grp_slot0.clear();
        st.grp_slots.clear();

        for (int32_t k = 0; k < n_present; ++k) {
            add_key(seq_ext(seq_present[k]));
        }
        if (!keys) {
            const auto sets = st.key_set;
            st.key_set.clear();
            for (const auto & set : sets) {
                new_multi_key(set);
            }
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            int32_t key = keys ? key_of(j) : st.cell_key[j];

            if (key == -2) {
                key = n_present + (int32_t) st.key_set.size();
                new_multi_key(cells.seq_get_all((uint32_t) j));
            }

            st.cell_key[j] = key;

            if (key < 0) {
                continue;
            }

            const int64_t idx = idx_of(j);
            const int64_t pb  = bucket_of(idx);

            st.cell_idx[j] = (int32_t) idx;

            if (pb >= n_blocks) {
                st.oor = true;
                continue;
            }

            GGML_ASSERT(pb < st.grp_ext[key]);

            const int64_t  g    = st.grp_base[key] + pb;
            const int64_t  slot = slot_of(idx);
            const uint64_t bit  = uint64_t(1) << slot;

            if (st.grp_first[g] < 0) {
                st.grp_first[g] = (int32_t) j;
            }

            if ((st.grp_slots[g] & bit) != 0) {
                st.dup = true;
            }

            st.grp_slots[g] |= bit;

            if (slot == 0) {
                st.grp_slot0[g] = (int32_t) j;
            }
        }
    };

    // key_seq[k*LLAMA_MAX_SEQ + sq]: whether the cells of key k belong to sequence sq
    auto & key_seq = st.key_seq;
    auto & key_low = st.key_low;

    auto build_key_tables = [&]() {
        const int32_t n_key = n_present + (int32_t) st.key_set.size();

        key_seq.assign((size_t) n_key*LLAMA_MAX_SEQ, 0);
        key_low.assign(n_key, LLAMA_MAX_SEQ - 1);

        for (int32_t k = 0; k < n_key; ++k) {
            for (int64_t sq = LLAMA_MAX_SEQ - 1; sq >= 0; --sq) {
                if (k < n_present ? seq_present[k] == sq : st.key_set[k - n_present].test(sq)) {
                    key_seq[k*LLAMA_MAX_SEQ + sq] = 1;
                    key_low[k] = (llama_seq_id) sq;
                }
            }
        }
    };

    auto bid_pos_of = [&](int32_t pb_idx, int32_t slot0) {
        std::array<int32_t, 4> sec_pos = { pb_idx, pb_idx, pb_idx, pb_idx };

        if (st.ranked) {
            const llama_pos p = cells.pos_get(slot0);
            const auto &    e = cells.ext_get(slot0);

            sec_pos = { p, e.y, e.x, p };
        }

        return sec_pos;
    };

    auto build = [&]() {
        st.cells       = &cells;
        st.n_kv        = n_kv;
        st.r           = r;
        st.n_seq_vis   = n_seq_vis;
        st.seq_present = seq_present;
        st.key_set.clear();
        st.cell_key.resize(n_kv);
        st.cell_idx.resize(n_kv);
        st.ranked = false;
        st.ranked_seqs.reset();

        group_cells(true);

        // [TAG_QSA_MROPE_RANK] mrope repeats one position across an image, so rank cells instead of
        // using the position. Ranks are per sequence: a unified cache holds every slot's cells at
        // once, and a global order would count a neighbour's cells into this sequence's blocks.
        // Before this the ranking required the pool to hold a single sequence, so an image under
        // --parallel fell through to the position bucket: ~2000 cells of one image landed in one
        // block, never completed, and every query paid ~500 forced spare blocks per image - which
        // exhausted the top-k budget on image patches and left the text unread.
        // A sequence without an image keeps rank == position (its positions run from 0 without
        // gaps), so only the sequences carrying an image are ordered.
        if (st.dup && ubatch->is_pos_2d()) {
            // the sequences whose cells repeat an index within a block: an mrope image
            llama_kv_cells::seq_set_t dup_seqs;

            std::fill(st.grp_slots.begin(), st.grp_slots.end(), 0);

            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t key = st.cell_key[j];
                const int64_t idx = st.cell_idx[j];

                if (key < 0 || bucket_of(idx) >= n_blocks) {
                    continue;
                }

                const int64_t  g   = st.grp_base[key] + bucket_of(idx);
                const uint64_t bit = uint64_t(1) << slot_of(idx);

                if ((st.grp_slots[g] & bit) != 0) {
                    dup_seqs |= cells.seq_get_all((uint32_t) j);
                }

                st.grp_slots[g] |= bit;
            }

            st.rank.assign(n_kv, -1);
            st.seq_order.resize(LLAMA_MAX_SEQ);

            st.ranked_seqs = dup_seqs;

            for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
                if (!st.ranked_seqs.test(sq)) {
                    continue;
                }

                auto & order = st.seq_order[sq];
                order.clear();

                for (const auto & [p, c] : cells.seq_pos_cells(sq)) {
                    order.push_back((int32_t) c);
                }

                // same total order the mrope causal mask uses: pos, then ext.y, then ext.x. The
                // cells arrive ordered by position, so only the image patches sharing one
                // position are left to sort
                for (size_t b = 0; b < order.size();) {
                    const llama_pos p = cells.pos_get(order[b]);

                    size_t e = b + 1;
                    while (e < order.size() && cells.pos_get(order[e]) == p) {
                        ++e;
                    }

                    if (e - b > 1) {
                        std::sort(order.begin() + b, order.begin() + e, [&cells](int32_t ca, int32_t cb) {
                            const auto & ea = cells.ext_get(ca);

                            return cells.ext_get(cb).is_2d_gt(ea.x, ea.y);
                        });
                    }

                    b = e;
                }

                // a cell shared between sequences (seq_cp) is a common prefix of both, so its rank
                // agrees under either; the lowest sequence assigns it
                for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                    if (st.rank[order[k]] < 0) {
                        st.rank[order[k]] = (int32_t) k;
                    }
                }
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                if (st.rank[j] < 0 && !cells.is_empty(j)) {
                    st.rank[j] = cells.pos_get(j);
                }
            }

            st.ranked = true;

            group_cells(false);
        }

        GGML_ASSERT((!blk_bias || !st.oor) && "qsa: cell position runs past the cell window");

        const int32_t n_key = n_present + (int32_t) st.key_set.size();

        st.bid_idx  .clear();
        st.bid_cell .clear();
        st.bid_slot0.clear();
        st.bid_key  .clear();
        st.bid_grp  .clear();
        st.bid_pos  .clear();
        st.grp_bid.assign(st.grp_base.back(), -1);

        int64_t n_pb = 0;
        for (int32_t k = 0; k < n_key; ++k) {
            n_pb = std::max(n_pb, st.grp_ext[k]);
        }

        // within a bucket, bids follow the groups' first cells in descending order
        std::vector<int64_t> pb_grp;

        for (int64_t pb = 0; pb < n_pb; ++pb) {
            pb_grp.clear();

            for (int32_t k = 0; k < n_key; ++k) {
                if (pb < st.grp_ext[k] && st.grp_slots[st.grp_base[k] + pb] == slots_full) {
                    pb_grp.push_back(st.grp_base[k] + pb);
                }
            }

            for (size_t a = 1; a < pb_grp.size(); ++a) {
                for (size_t b = a; b > 0 && st.grp_first[pb_grp[b - 1]] < st.grp_first[pb_grp[b]]; --b) {
                    std::swap(pb_grp[b - 1], pb_grp[b]);
                }
            }

            for (const int64_t g : pb_grp) {
                st.grp_bid[g] = (int32_t) st.bid_idx.size();

                st.bid_idx  .push_back((int32_t) (pb*r));
                st.bid_cell .push_back(st.grp_first[g]);
                st.bid_slot0.push_back(st.grp_slot0[g]);
                st.bid_key  .push_back(st.cell_key[st.grp_first[g]]);
                st.bid_grp  .push_back(g);

                const auto sec_pos = bid_pos_of((int32_t) (pb*r), st.grp_slot0[g]);
                st.bid_pos.insert(st.bid_pos.end(), sec_pos.begin(), sec_pos.end());
            }
        }

        const int32_t n_bid = (int32_t) st.bid_idx.size();

        GGML_ASSERT(n_bid <= n_blocks);

        // blk_cells also feeds a ggml_get_rows over the whole key cache, so every slot has to name
        // a real cell even when nothing selects it
        st.bid_cells.assign((size_t) n_bid*r, 0);
        st.unpooled.clear();
        st.pad = -1;

        if (need_blk_of) {
            st.blk_of.resize(n_kv);
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t key = st.cell_key[j];

            if (key < 0) {
                if (st.pad < 0) {
                    st.pad = (int32_t) j;
                }
                if (need_blk_of) {
                    st.blk_of[j] = -1;
                }
                continue;
            }

            const int64_t idx = st.cell_idx[j];
            const int64_t pb  = bucket_of(idx);
            const int32_t bid = pb < n_blocks ? st.grp_bid[st.grp_base[key] + pb] : -1;

            if (bid >= 0) {
                st.bid_cells[bid*r + slot_of(idx)] = (int32_t) j;
            } else {
                st.unpooled.push_back((int32_t) j);
            }

            if (need_blk_of) {
                st.blk_of[j] = bid;
            }
        }

        // [TAG_QSA_DEVICE_VIS] the per-cell half of the attention mask's rule, per sequence: a
        // cell is visible to a query of sequence sq when cell_idx[sq][cell] <= q. Under an image
        // the rank orders (pos, y, x) exactly as the mrope mask compares them; otherwise positions
        // are unique within a sequence and the rule is the causal one
        build_key_tables();

        st.vis.resize((size_t) n_kv*n_seq_vis);

        for (int64_t sq = 0; sq < n_seq_vis; ++sq) {
            int32_t key_sq = -3;
            for (int32_t k = 0; k < n_present; ++k) {
                if (seq_present[k] == sq) {
                    key_sq = k;
                }
            }

            int32_t * row = st.vis.data() + sq*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                row[j] = st.cell_key[j] == key_sq ? st.cell_idx[j] : INT32_MAX;
            }

            if (st.key_set.empty()) {
                continue;
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t key = st.cell_key[j];
                if (key >= n_present && key_seq[key*LLAMA_MAX_SEQ + sq]) {
                    row[j] = st.cell_idx[j];
                }
            }
        }

        // LLAMA_QSA_NO_RANK_APPEND=1: a ranked state always rebuilds, as before
        static const bool rank_append = getenv("LLAMA_QSA_NO_RANK_APPEND") == nullptr;

        st.valid = keep && !need_blk_of && !st.dup && !st.oor && (rank_append || !st.ranked);
        st.mark  = cells.journal_end();
    };

    // [TAG_QSA_RANK_APPEND] with an image anywhere in the pool the state is ranked, and a decoded
    // cell of a ranked sequence takes the next rank when it sorts after all its old cells. Else -1
    auto rank_of_new = [&](int64_t j, int32_t key) -> int64_t {
        const llama_seq_id sq = seq_present[key];
        const llama_pos    p  = cells.pos_get(j);

        if (!st.ranked_seqs.test(sq)) {
            st.rank[j] = p;
            return p;
        }

        const auto & sp = cells.seq_pos_cells(sq);
        const auto   it = sp.find({p, (uint32_t) j});

        if (it == sp.end() || (it != sp.begin() && std::prev(it)->first == p)) {
            return -1;
        }

        // the cells after j have to be new as well, so no old cell changes its rank
        int64_t n_after = 0;
        for (auto nx = std::next(it); nx != sp.end(); ++nx, ++n_after) {
            if (nx->first == p || nx->second >= n_kv || st.cell_key[nx->second] >= 0 || n_after >= 64) {
                return -1;
            }
        }

        auto & order = st.seq_order[sq];

        const int64_t rank = (int64_t) sp.size() - 1 - n_after;

        if (rank != (int64_t) order.size()) {
            return -1;
        }

        order.push_back((int32_t) j);
        st.rank[j] = (int32_t) rank;

        return rank;
    };

    // a decode step adds a few cells to empty slots; anything else (a removal, a new shared
    // cell, a repeated index, a grown window) rebuilds. Returns false when it has to
    auto update = [&]() {
        if (!st.valid || st.cells != &cells || st.n_kv != n_kv || st.r != r || st.n_seq_vis != n_seq_vis || st.seq_present != seq_present || need_blk_of) {
            return false;
        }

        size_t n_changed = 0;
        const uint32_t * changed = cells.journal_since(st.mark, n_changed);

        if (changed == nullptr || n_changed > 4096) {
            return false;
        }

        // the journal can name a cell more than once; its first entry already saw the final state
        std::vector<int32_t> added;

        for (size_t c = 0; c < n_changed; ++c) {
            const int64_t j = changed[c];

            if (j >= n_kv || std::find(added.begin(), added.end(), (int32_t) j) != added.end()) {
                continue;
            }

            const int32_t key = key_of(j);

            if (key == st.cell_key[j] && (key < 0 || st.cell_idx[j] == cells.pos_get(j))) {
                continue;
            }

            if (key < 0 || key >= n_present || st.cell_key[j] >= 0) {
                return false;
            }

            const int64_t idx = st.ranked ? rank_of_new(j, key) : cells.pos_get(j);

            if (idx < 0) {
                return false;
            }

            added.push_back((int32_t) j);

            const int64_t pb  = bucket_of(idx);

            if (pb >= n_blocks || pb >= st.grp_ext[key]) {
                return false;
            }

            const int64_t  g    = st.grp_base[key] + pb;
            const int64_t  slot = slot_of(idx);
            const uint64_t bit  = uint64_t(1) << slot;

            if ((st.grp_slots[g] & bit) != 0) {
                return false;
            }

            st.cell_key[j] = key;
            st.cell_idx[j] = (int32_t) idx;

            st.grp_first[g]  = st.grp_first[g] < 0 ? (int32_t) j : std::min(st.grp_first[g], (int32_t) j);
            st.grp_slots[g] |= bit;
            if (slot == 0) {
                st.grp_slot0[g] = (int32_t) j;
            }

            if (seq_present[key] < n_seq_vis) {
                st.vis[seq_present[key]*n_kv + j] = (int32_t) idx;
            }

            if (st.pad == j) {
                do {
                    st.pad++;
                } while (st.pad < n_kv && !cells.is_empty(st.pad));
                if (st.pad >= n_kv) {
                    st.pad = -1;
                }
            }

            if (st.grp_slots[g] != slots_full) {
                st.unpooled.insert(std::upper_bound(st.unpooled.begin(), st.unpooled.end(), (int32_t) j), (int32_t) j);
                continue;
            }

            // the group is complete: it takes a bid in (bucket, descending first cell) order,
            // and its other cells leave the unpooled list
            const int32_t first = st.grp_first[g];
            const int32_t pb_idx = (int32_t) (pb*r);

            int32_t t = 0;
            {
                int32_t lo = 0, hi = (int32_t) st.bid_idx.size();
                while (lo < hi) {
                    const int32_t mid = (lo + hi)/2;
                    if (st.bid_idx[mid] < pb_idx || (st.bid_idx[mid] == pb_idx && st.bid_cell[mid] > first)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                t = lo;
            }

            if ((int64_t) st.bid_idx.size() + 1 > n_blocks) {
                return false;
            }

            std::array<int32_t, 64> members;
            members.fill(0);
            members[slot] = (int32_t) j;

            for (size_t u = 0; u < st.unpooled.size();) {
                const int32_t cu = st.unpooled[u];
                if (st.cell_key[cu] == key && bucket_of(st.cell_idx[cu]) == pb) {
                    members[slot_of(st.cell_idx[cu])] = cu;
                    st.unpooled.erase(st.unpooled.begin() + u);
                } else {
                    ++u;
                }
            }

            const auto sec_pos = bid_pos_of(pb_idx, st.grp_slot0[g]);

            st.bid_idx  .insert(st.bid_idx  .begin() + t, pb_idx);
            st.bid_cell .insert(st.bid_cell .begin() + t, first);
            st.bid_slot0.insert(st.bid_slot0.begin() + t, st.grp_slot0[g]);
            st.bid_key  .insert(st.bid_key  .begin() + t, key);
            st.bid_grp  .insert(st.bid_grp  .begin() + t, g);
            st.bid_pos  .insert(st.bid_pos  .begin() + (size_t) t*4, sec_pos.begin(), sec_pos.end());
            st.bid_cells.insert(st.bid_cells.begin() + (size_t) t*r, members.begin(), members.begin() + r);

            for (size_t u = t; u < st.bid_idx.size(); ++u) {
                st.grp_bid[st.bid_grp[u]] = (int32_t) u;
            }
        }

        st.mark = cells.journal_end();

        build_key_tables();

        return true;
    };

    // LLAMA_QSA_INPUT_CHECK=1 rebuilds after every in-place update and aborts on any difference
    static const bool check = getenv("LLAMA_QSA_INPUT_CHECK") != nullptr;

    const bool in_place = update();

    if (!in_place) {
        build();
    } else if (check) {
        const qsa_input_state upd = st;

        build();

        bool same = upd.cell_key == st.cell_key && upd.cell_idx == st.cell_idx && upd.bid_idx == st.bid_idx &&
            upd.bid_cell == st.bid_cell && upd.bid_key == st.bid_key && upd.bid_pos == st.bid_pos &&
            upd.bid_cells == st.bid_cells && upd.unpooled == st.unpooled && upd.pad == st.pad && upd.vis == st.vis &&
            upd.ranked == st.ranked && upd.ranked_seqs == st.ranked_seqs && (!st.ranked || upd.rank == st.rank);

        for (int sq = 0; same && st.ranked && sq < LLAMA_MAX_SEQ; ++sq) {
            same = !st.ranked_seqs.test(sq) || upd.seq_order[sq] == st.seq_order[sq];
        }

        GGML_ASSERT(same && "qsa: in-place input update differs from a rebuild");
    }

    // a rebuild on every decode step costs host time per token that grows with the whole pool
    static uint64_t n_sync    = 0;
    static uint64_t n_rebuilt = 0;

    n_sync++;
    n_rebuilt += in_place ? 0 : 1;

    const uint64_t n_log = check ? 256 : 4096;

    if (n_sync % n_log == 0) {
        LLAMA_LOG_WARN("qsa: input sync rebuilt %" PRIu64 " of the last %" PRIu64 " (ranked %d, n_kv %" PRId64 ")\n", n_rebuilt, n_log, st.ranked ? 1 : 0, n_kv);
        n_rebuilt = 0;
    }

    st.synced      = true;
    st.need_blk_of = need_blk_of;
    st.pos_2d      = ubatch->is_pos_2d();
}

static double qsa_scope_cap() {
    static const double cap = []() {
        const char * requested = getenv("LLAMA_QSA_SCOPE_CAP");
        return requested == nullptr ? -1.0 : std::max(atof(requested), 0.0);
    }();

    return cap;
}

uint32_t llama_memory_hybrid_idx::qsa_scope_n_blocks(
        const llama_ubatch & ubatch,
        uint32_t n_kv,
        uint32_t ratio,
        uint32_t n_seq_vis,
        uint32_t & n_seq) const {
    n_seq = 0;

    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;

    // up to 1024 columns the device top-k sorts, and a sort returns the winners by score, not in
    // list order, so every list stays longer than that. The padding lets graph reuse hold while a
    // sequence grows
    constexpr int64_t n_blk_min = 1025;
    constexpr int64_t n_blk_pad = 256;

    const auto seqs = qsa_scope_seqs(ubatch);

    if (seqs.empty() || !pooled_is_keyed_by_seq() || get_mem_idx() == nullptr || n_blocks < n_blk_min) {
        return 0;
    }

    qsa_input_state & st = qsa_st;

    const auto & cells = get_mem_idx()->get_cells(seqs[0]);

    qsa_sync(st, cells, &ubatch, n_kv, r, n_seq_vis, true, false, true);

    const int64_t n_key = (int64_t) st.key_low.size();

    std::vector<int64_t> key_n(n_key, 0);
    for (const int32_t k : st.bid_key) {
        key_n[k]++;
    }

    // spare blocks take the unpooled cells r at a time, in cell order, as set_input_qsa packs them
    const int64_t n_up = (int64_t) st.unpooled.size();

    int64_t n_max = 0;

    for (const llama_seq_id sq : seqs) {
        int64_t n = 0;

        for (int64_t k = 0; k < n_key; ++k) {
            if (st.key_seq[k*LLAMA_MAX_SEQ + sq]) {
                n += key_n[k];
            }
        }

        for (int64_t d = 0; d*r < n_up; ++d) {
            for (int64_t u = d*r; u < std::min(n_up, d*r + r); ++u) {
                if (cells.seq_has(st.unpooled[u], sq)) {
                    n++;
                    break;
                }
            }
        }

        n_max = std::max(n_max, n);
    }

    int64_t n_blk = std::max(n_max, n_blk_min);
    n_blk = std::min((n_blk + n_blk_pad - 1)/n_blk_pad*n_blk_pad, n_blocks);

    // every list is padded to the longest, so sequences of very different length can cost more than the pool (LLAMA_QSA_SCOPE_CAP, <= 0: no cap)
    const double scope_cap = qsa_scope_cap();
    const bool over_cap = scope_cap < 0.0 ?
        (int64_t) seqs.size()*n_blk > n_blocks + n_blocks/4 :
        scope_cap > 0.0 && (double) seqs.size()*n_blk > scope_cap * (double) n_blocks;
    if (over_cap) {
        return 0;
    }

    n_seq = (uint32_t) seqs.size();

    return (uint32_t) n_blk;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t n_kv_graph,
        uint32_t n_stream,
        uint32_t ratio,
        bool blk_bias,
        bool causal_attn,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows,
        ggml_tensor * blk_rows,
        const llama_qsa_device_inputs * dev,
        ggml_tensor * scope_rows,
        ggml_tensor * scope_cells,
        ggml_tensor * scope_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    if (cell_blk != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));
    }
    if (blk_cells != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(blk_cells->buffer));
    }

    const int64_t n_kv     = n_kv_graph;
    const int64_t n_ns     = n_stream;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    // same formula as the graph; blk_pos may be null on the pooled path
    const int64_t n_blocks = (n_kv + r - 1)/r;

    // every tensor below was sized from the graph's n_kv, so a mismatch here silently writes
    // past the host input buffer and only surfaces as a fault on the GPU
    GGML_ASSERT(cell_blk  == nullptr || (cell_blk->ne[0]  == n_kv        && cell_blk->ne[1] == n_ns));
    GGML_ASSERT(blk_cells == nullptr || (blk_cells->ne[0] == r*n_blocks  && blk_cells->ne[1] == n_ns));
    GGML_ASSERT(blk_pos   == nullptr ||  blk_pos->ne[0]   == 4*n_blocks*n_ns);
    GGML_ASSERT(bias == nullptr || bias->ne[0] == (blk_bias ? n_blocks : n_kv));
    GGML_ASSERT(dirty_cells == nullptr || (dirty_cells->ne[0] == r*dirty_rows->ne[0] && dirty_cells->ne[1] == n_ns));
    GGML_ASSERT(dirty_pos   == nullptr ||  dirty_pos->ne[0]   == 4*dirty_rows->ne[0]*n_ns);
    GGML_ASSERT(dirty_rows  == nullptr ||  dirty_rows->ne[1]  == n_ns);
    GGML_ASSERT(blk_rows    == nullptr || (blk_rows->ne[0]    == n_blocks    && blk_rows->ne[1] == n_ns));
    GGML_ASSERT(scope_cells == nullptr || (blk_bias && n_ns == 1 && pooled_is_keyed_by_seq() &&
            scope_cells->ne[0] == r && scope_rows->ne[0] == scope_cells->ne[1]*scope_cells->ne[2] &&
            (scope_bias == nullptr || (scope_bias->ne[0] == scope_cells->ne[1] && scope_bias->ne[2] == scope_cells->ne[2] &&
            scope_bias->ne[1]*scope_bias->ne[2] == n_tokens))));

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = cell_blk != nullptr ? (int32_t *) cell_blk->data : nullptr;
    float   * dst_bias      = bias != nullptr ? (float *) bias->data : nullptr;

    // [TAG_QSA_DEVICE_INPUTS] an unreferenced input is never allocated, so only a table with a
    // buffer is filled; the bias tables and the visibility tables are independent sets
    static const llama_qsa_device_inputs no_dev;
    const llama_qsa_device_inputs & tbl = dev != nullptr ? *dev : no_dev;

    const bool dev_bias  = tbl.blk_start   != nullptr && tbl.blk_start->buffer   != nullptr;
    const bool dev_scope = tbl.scope_start != nullptr && tbl.scope_start->buffer != nullptr;
    const bool dev_vis   = tbl.cell_idx    != nullptr && tbl.cell_idx->buffer    != nullptr;

    GGML_ASSERT((dev_bias || dev_scope || bias != nullptr || scope_bias != nullptr) && "qsa: neither a bias input nor its tables");
    GGML_ASSERT(!dev_scope || (scope_cells != nullptr && scope_bias == nullptr &&
            tbl.scope_start->ne[0] == scope_cells->ne[1] && tbl.scope_start->ne[2] == scope_cells->ne[2] &&
            tbl.scope_spare->ne[0] == scope_cells->ne[1] && tbl.scope_spare->ne[2] == scope_cells->ne[2] &&
            tbl.tok_q->ne[1] == n_tps && tbl.tok_m->ne[1] == n_tps));
    GGML_ASSERT(!dev_bias || (blk_bias && tbl.blk_start->ne[0] == n_blocks && tbl.blk_start->ne[2] == n_ns &&
            tbl.blk_spare->ne[0] == n_blocks && tbl.blk_spare->ne[2] == n_ns &&
            tbl.tok_seq->ne[0] == n_tps && tbl.tok_seq->ne[1] == n_ns &&
            tbl.tok_q->ne[1] == n_tps && tbl.tok_q->ne[2] == n_ns &&
            tbl.tok_m->ne[1] == n_tps && tbl.tok_m->ne[2] == n_ns));
    GGML_ASSERT(!dev_vis || (tbl.cell_idx->ne[0] == n_kv && tbl.cell_idx->ne[3] == n_ns &&
            tbl.q_meta->ne[0] == 2 && tbl.q_meta->ne[1] == n_tps && tbl.q_meta->ne[3] == n_ns &&
            tbl.zero_mask->ne[0] == n_kv && ggml_nelements(tbl.zero_mask) == n_kv));

    const int64_t n_seq_bias = dev_bias ? tbl.blk_start->ne[1] : 0;
    const int64_t n_seq_vis  = dev_vis  ? tbl.cell_idx->ne[1]  : 0;

    // the gathered path derives its mask from the tables and never reads the one-row mask
    if (dev_vis && tbl.zero_mask->buffer != nullptr) {
        memset(tbl.zero_mask->data, 0, ggml_nbytes(tbl.zero_mask));
    }

    // [TAG_QSA_POOLED_CACHE] the pooled path drops blk_cells/blk_pos from the graph (the dirty
    // tables replace them), so they may be null here; the block map is still needed for the
    // dirty fill, so it is built either way
    int32_t * dst_blk_cells = blk_cells != nullptr ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos   != nullptr ? (int32_t *) blk_pos->data   : nullptr;

    // a stream's state is kept only when it is the only stream
    qsa_input_state tmp_st;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk != nullptr ? dst_cell_blk + s*n_kv : nullptr;

        const bool need_blk_of = cur_cell_blk != nullptr || (dst_bias != nullptr && !blk_bias);

        qsa_input_state & st = n_ns == 1 ? qsa_st : tmp_st;

        qsa_sync(st, cells, ubatch, n_kv, r, n_seq_vis, n_ns == 1, need_blk_of, blk_bias);

        const auto & key_seq = st.key_seq;
        const auto & key_low = st.key_low;

        const int32_t n_bid = (int32_t) st.bid_idx.size();
        const auto &  unpooled_cells = st.unpooled;
        const int32_t pad = st.pad;
        const bool    ranked = st.ranked;
        const auto &  bid_idx  = st.bid_idx;
        const auto &  bid_cell = st.bid_cell;

        const int32_t dead_bid = n_bid < n_blocks ? n_bid : n_blocks - 1;

        if (cur_cell_blk != nullptr) {
            for (int64_t j = 0; j < n_kv; ++j) {
                cur_cell_blk[j] = st.blk_of[j] < 0 ? dead_bid : st.blk_of[j];
            }
        }

        // unpooled cells sit in spare blocks: each complete block pools r cells, so the
        // unpooled tail takes ceil(n_up / r) spare rows out of the (n_blocks - n_bid) left over.
        const int64_t n_up   = (int64_t) unpooled_cells.size();
        const int64_t n_dead = (n_up + r - 1)/r;

        // [TAG_QSA_BLOCK_TOPK] block-level top-k gathers whole rows of blk_cells, so the spare
        // blocks need the real unpooled cells in their rows. Leaving fill-zero rows would make
        // the always-visible tail - the incomplete block that holds the query's own token -
        // read cell 0 r times over, which the per-cell cell_blk expansion never did.
        std::vector<int32_t> dead_cells;

        if (blk_bias && n_dead > 0) {
            GGML_ASSERT(n_bid + n_dead <= n_blocks && "qsa: not enough block slots for unpooled cells");
            GGML_ASSERT((n_up == n_dead*r || pad >= 0) &&
                    "qsa: no empty cell to pad the spare block with");

            // an empty cell is -inf in the attention mask, so it pads the tail of a real spare
            // block harmlessly - and a cell list built from these rows drops it for the same reason
            dead_cells.resize(n_dead*r);

            for (int64_t idx = 0; idx < n_dead*r; ++idx) {
                dead_cells[idx] = idx < n_up ? unpooled_cells[idx] : pad;
            }
        }

        // whether spare block d holds a cell of seq_id whose causal index is at most q
        auto spare_has = [&](int64_t d, llama_seq_id seq_id, int64_t q) {
            const int32_t * dead_row = &dead_cells[d*r];

            for (int64_t k = 0; k < r; ++k) {
                const int32_t c = dead_row[k];

                if (c >= 0 && !cells.is_empty(c) && cells.seq_has(c, seq_id) && st.cell_idx[c] <= q) {
                    return true;
                }
            }

            return false;
        };

        if (dst_blk_cells != nullptr) {
            int32_t * dst = dst_blk_cells + s*(r*n_blocks);

            std::copy(st.bid_cells.begin(), st.bid_cells.end(), dst);
            std::copy(dead_cells.begin(), dead_cells.end(), dst + (int64_t) n_bid*r);
            std::fill(dst + (int64_t) n_bid*r + (int64_t) dead_cells.size(), dst + r*n_blocks, 0);
        }

        // [TAG_QSA_DEVICE_INPUTS] the per-block half of the bias rule, per sequence: what the
        // per-token loop below compares q against. A spare block's start is the lowest index of
        // the sequence's cells in it, so "start <= q" is exactly its "holds a visible cell"
        if (dev_bias) {
            constexpr float never = 1e30f;

            float * dst_start = (float *) tbl.blk_start->data + s*(n_blocks*n_seq_bias);
            float * dst_spare = (float *) tbl.blk_spare->data + s*n_blocks;

            std::fill(dst_start, dst_start + n_blocks*n_seq_bias, never);
            std::fill(dst_spare, dst_spare + n_blocks, 0.0f);

            for (int32_t b = 0; b < n_bid; ++b) {
                for (int64_t sq = 0; sq < n_seq_bias; ++sq) {
                    if (cells.seq_has((uint32_t) bid_cell[b], (llama_seq_id) sq)) {
                        dst_start[sq*n_blocks + b] = (float) bid_idx[b];
                    }
                }
            }

            for (int64_t d = 0; d < n_dead; ++d) {
                const int32_t   bid      = n_bid + (int32_t) d;
                const int32_t * dead_row = &dead_cells[d*r];

                dst_spare[bid] = 1.0f;

                for (int64_t k = 0; k < r; ++k) {
                    const int32_t c = dead_row[k];

                    if (c < 0 || cells.is_empty(c)) {
                        continue;
                    }

                    for (int64_t sq = 0; sq < n_seq_bias; ++sq) {
                        if (cells.seq_has(c, (llama_seq_id) sq)) {
                            float & start = dst_start[sq*n_blocks + bid];
                            start = std::min(start, (float) st.cell_idx[c]);
                        }
                    }
                }
            }
        }

        if (dev_vis) {
            std::copy(st.vis.begin(), st.vis.end(), (int32_t *) tbl.cell_idx->data + s*(n_kv*n_seq_vis));
        }

        if (dst_blk_pos != nullptr) {
            for (int64_t sec = 0; sec < 4; ++sec) {
                int32_t * dst = dst_blk_pos + sec*(n_blocks*n_ns) + s*n_blocks;

                for (int32_t b = 0; b < n_bid; ++b) {
                    dst[b] = st.bid_pos[b*4 + sec];
                }
                std::fill(dst + n_bid, dst + n_blocks, 0);
            }
        }

        // [TAG_QSA_POOLED_CACHE] which sequence owns each complete block. Groups are keyed on
        // (sequence set, position bucket), so a block never mixes sequences, but a block shared
        // after seq_cp carries several bits: take the lowest. That is not a shortcut - the pooled
        // key is a function of the member cells alone, so the sharers read identical data.
        const int64_t dustbin = (int64_t) get_pooled_rows() - 1;

        std::vector<int32_t> blk_seq;
        if (pooled_is_keyed_by_seq()) {
            blk_seq.resize(n_bid);

            for (int32_t t = 0; t < n_bid; ++t) {
                blk_seq[t] = key_low[st.bid_key[t]];
            }
        }

        // the reader gathers one pooled row per block; blocks past n_bid are spare or dead and
        // must still name a finite row
        if (blk_rows != nullptr) {
            int32_t * dst_b_rows = (int32_t *) blk_rows->data + s*n_blocks;

            for (int64_t b = 0; b < n_blocks; ++b) {
                if (b < n_bid) {
                    const int64_t pb = bid_idx[b]/r;
                    dst_b_rows[b] = (int32_t) pooled_row_of(blk_seq[b], pb);
                } else {
                    dst_b_rows[b] = (int32_t) dustbin;
                }
            }

            // [TAG_QSA_BLK_ROWS_CHECK] the device gather has no bounds check: a row outside the
            // store is an asynchronous illegal memory access that names the node but not the
            // value. Log and send the block to the dustbin instead.
            const int64_t n_rows_valid = (int64_t) get_pooled_rows();
            int64_t n_bad = 0;
            for (int64_t b = 0; b < n_blocks; ++b) {
                if (dst_b_rows[b] >= 0 && dst_b_rows[b] < n_rows_valid) {
                    continue;
                }
                if (n_bad < 8) {
                    LLAMA_LOG_WARN("%s: blk_rows[%" PRId64 "] = %d outside [0, %" PRId64 ") (stream %" PRId64 ", n_bid %d, n_blocks %" PRId64 ", blk_seq %d, cell %d, blk %" PRId64 ", seq set %s)\n",
                            __func__, b, dst_b_rows[b], n_rows_valid, s, n_bid, n_blocks,
                            b < n_bid ? blk_seq[b] : -1,
                            b < n_bid ? bid_cell[b] : -1,
                            b < n_bid ? bid_idx[b]/r : (int64_t) -1,
                            b < n_bid ? cells.seq_get_all((uint32_t) bid_cell[b]).to_string().c_str() : "-");
                }
                dst_b_rows[b] = (int32_t) dustbin;
                n_bad++;
            }
            if (n_bad > 0) {
                LLAMA_LOG_WARN("%s: %" PRId64 " of %" PRId64 " blk_rows out of range, sent to the dustbin\n", __func__, n_bad, n_blocks);
            }
        }

        // [TAG_QSA_SEQ_SCOPE] each sequence's candidates: the complete blocks it holds and the spare
        // blocks with a cell of it, in whole-pool order. Past its list a sequence names the dustbin
        // row and a cell it cannot see, and its bias drops them
        std::vector<std::vector<int32_t>> scope_blk;

        if (scope_cells != nullptr) {
            const int64_t n_blk = scope_cells->ne[1];
            const int64_t n_seq = scope_cells->ne[2];

            const auto seqs = qsa_scope_seqs(*ubatch);

            GGML_ASSERT((int64_t) seqs.size() == n_seq && "qsa: scoped tables sized for another ubatch");

            std::vector<int32_t> col_of(LLAMA_MAX_SEQ, -1);
            for (int64_t v = 0; v < n_seq; ++v) {
                col_of[seqs[v]] = (int32_t) v;
            }

            scope_blk.resize(n_seq);
            for (auto & list : scope_blk) {
                list.reserve(n_blk);
            }

            // keys below n_one name one sequence each
            const int32_t n_one = (int32_t) st.seq_present.size();

            for (int32_t b = 0; b < n_bid; ++b) {
                const int32_t k = st.bid_key[b];

                if (k < n_one) {
                    const int32_t v = col_of[st.seq_present[k]];
                    if (v >= 0) {
                        scope_blk[v].push_back(b);
                    }
                    continue;
                }

                for (int64_t v = 0; v < n_seq; ++v) {
                    if (key_seq[k*LLAMA_MAX_SEQ + seqs[v]]) {
                        scope_blk[v].push_back(b);
                    }
                }
            }

            for (int64_t d = 0; d < n_dead; ++d) {
                for (int64_t v = 0; v < n_seq; ++v) {
                    if (spare_has(d, seqs[v], INT64_MAX)) {
                        scope_blk[v].push_back(n_bid + (int32_t) d);
                    }
                }
            }

            const int64_t n_rows_valid = (int64_t) get_pooled_rows();
            int64_t n_bad = 0;

            for (int64_t v = 0; v < n_seq; ++v) {
                const auto & list = scope_blk[v];

                GGML_ASSERT((int64_t) list.size() <= n_blk && "qsa: scoped tables sized at graph build; see qsa_scope_n_blocks");

                // an empty cell, else the first cell without the sequence
                int32_t hidden = pad;
                for (int64_t j = 0; hidden < 0 && j < n_kv; ++j) {
                    if (!cells.seq_has((uint32_t) j, seqs[v])) {
                        hidden = (int32_t) j;
                    }
                }
                hidden = std::max(hidden, 0);

                int32_t * dst_rows  = (int32_t *) scope_rows->data  + v*n_blk;
                int32_t * dst_cells = (int32_t *) scope_cells->data + v*r*n_blk;

                for (int64_t j = 0; j < n_blk; ++j) {
                    const int32_t b = j < (int64_t) list.size() ? list[j] : -1;

                    int64_t row = b >= 0 && b < n_bid ? pooled_row_of(blk_seq[b], bid_idx[b]/r) : dustbin;

                    // [TAG_QSA_BLK_ROWS_CHECK]
                    if (row < 0 || row >= n_rows_valid) {
                        row = dustbin;
                        n_bad++;
                    }

                    dst_rows[j] = (int32_t) row;

                    const int32_t * src = b < 0 ? nullptr : b < n_bid ? &st.bid_cells[(size_t) b*r] : &dead_cells[(size_t) (b - n_bid)*r];

                    for (int64_t k = 0; k < r; ++k) {
                        dst_cells[j*r + k] = src != nullptr ? src[k] : hidden;
                    }

                    // [TAG_QSA_SCOPE_CHUNKS] the blk_start/blk_spare values of this block for this sequence
                    if (dev_scope) {
                        constexpr float never = 1e30f;

                        float start = never;
                        if (b >= 0 && b < n_bid) {
                            if (cells.seq_has((uint32_t) bid_cell[b], seqs[v])) {
                                start = (float) bid_idx[b];
                            }
                        } else if (b >= n_bid) {
                            for (int64_t k = 0; k < r; ++k) {
                                const int32_t c = src[k];
                                if (c >= 0 && !cells.is_empty(c) && cells.seq_has(c, seqs[v])) {
                                    start = std::min(start, (float) st.cell_idx[c]);
                                }
                            }
                        }

                        ((float *) tbl.scope_start->data)[v*n_blk + j] = start;
                        ((float *) tbl.scope_spare->data)[v*n_blk + j] = b >= n_bid ? 1.0f : 0.0f;
                    }
                }
            }

            if (n_bad > 0) {
                LLAMA_LOG_WARN("%s: %" PRId64 " scoped block rows out of range, sent to the dustbin\n", __func__, n_bad);
            }
        }

        // resolve which blocks the graph must (re)pool this ubatch: for every sequence with
        // tokens here, the range from its watermark to its last complete block. Complete blocks
        // are immutable, so rows below a watermark stay valid; rollbacks arrive as
        // seq_rm/state_read, which clamp the watermark before this runs. A sequence with no
        // tokens in this ubatch is left alone - its rows are masked by the bias either way.
        if (dirty_cells != nullptr) {
            const int64_t n_dirty_max = dirty_rows->ne[0];

            int32_t * dst_d_cells = (int32_t *) dirty_cells->data + s*(r*n_dirty_max);
            int32_t * dst_d_pos   = (int32_t *) dirty_pos->data;
            int64_t * dst_d_rows  = (int64_t *) dirty_rows->data  + s*n_dirty_max;

            // sequences carrying tokens in this stream, in first-appearance order
            std::vector<llama_seq_id> seqs;
            if (pooled_is_keyed_by_seq()) {
                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const llama_seq_id sq = ubatch->seq_id[s*n_tps + ii][0];
                    if (std::find(seqs.begin(), seqs.end(), sq) == seqs.end()) {
                        seqs.push_back(sq);
                    }
                }
            } else {
                seqs.push_back(seq_of_stream);
            }

            int64_t n_filled = 0;

            // last bid of each sequence; bids are in position-block order
            std::vector<int32_t> seq_last_bid(LLAMA_MAX_SEQ, -1);
            for (int32_t t = 0; t < n_bid && !blk_seq.empty(); ++t) {
                seq_last_bid[blk_seq[t]] = t;
            }

            for (const llama_seq_id sq : seqs) {
                const int32_t t_last = blk_seq.empty() ? n_bid - 1 : seq_last_bid[sq];
                const int64_t n_complete = t_last >= 0 ? (int64_t) bid_idx[t_last]/r + 1 : 0;

                auto & w = pooled_valid(sq);
                w = std::min(w, n_complete);

                for (int64_t b = w; b < n_complete; ++b) {
                    GGML_ASSERT(n_filled < n_dirty_max &&
                            "dirty tables sized at graph build; see qsa_pooled_n_dirty_max");

                    // the bid of block b for this sequence; an incomplete block below the complete
                    // end pools nothing this time and keeps its stale row, masked by the bias
                    int32_t t = -1;
                    for (auto it = std::lower_bound(bid_idx.begin(), bid_idx.end(), (int32_t) (b*r)); it != bid_idx.end() && *it == b*r; ++it) {
                        const int32_t u = (int32_t) (it - bid_idx.begin());
                        if (blk_seq.empty() || blk_seq[u] == sq) {
                            t = u;
                        }
                    }

                    dst_d_rows[n_filled] = pooled_row_of(sq, b);

                    for (int64_t sec = 0; sec < 4; ++sec) {
                        dst_d_pos[sec*(n_dirty_max*n_ns) + s*n_dirty_max + n_filled] =
                                t >= 0 ? st.bid_pos[t*4 + sec] : 0;
                    }
                    for (int64_t j = 0; j < r; ++j) {
                        dst_d_cells[n_filled*r + j] = t >= 0 ? st.bid_cells[t*r + j] : 0;
                    }

                    n_filled++;
                }

                w = n_complete;
            }

            // unused slots write the shared dustbin row, which nothing ever reads
            for (int64_t i = n_filled; i < n_dirty_max; ++i) {
                dst_d_rows[i] = dustbin;

                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_d_pos[sec*(n_dirty_max*n_ns) + s*n_dirty_max + i] = 0;
                }
                for (int64_t j = 0; j < r; ++j) {
                    dst_d_cells[i*r + j] = 0;
                }
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked && st.ranked_seqs.test(seq_id)) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                const auto & order = st.seq_order[seq_id];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (dev_bias) {
                ((int32_t *) tbl.tok_seq->data)[s*n_tps + ii] = (int32_t) seq_id;
            }

            if (dev_bias || dev_scope) {
                ((float   *) tbl.tok_q->data)  [s*n_tps + ii] = (float) q;
                ((float   *) tbl.tok_m->data)  [s*n_tps + ii] = (float) ((q + 1) % r);
            }

            if (dev_vis) {
                int32_t * meta = (int32_t *) tbl.q_meta->data + 2*(s*n_tps + ii);
                meta[0] = (int32_t) seq_id;
                meta[1] = (int32_t) q;
            }

            if (scope_bias != nullptr) {
                const int64_t n_blk = scope_bias->ne[0];
                const auto &  list  = scope_blk[ii/scope_bias->ne[1]];

                float * cur_scope_bias = (float *) scope_bias->data + i*n_blk;

                // the whole-pool rule below, on the listed blocks only
                for (int64_t j = 0; j < n_blk; ++j) {
                    const int32_t b = j < (int64_t) list.size() ? list[j] : -1;

                    cur_scope_bias[j] = b < 0     ? -INFINITY
                                      : b >= n_bid ? (spare_has(b - n_bid, seq_id, q) ? 1e9f : -INFINITY)
                                      : bid_idx[b] >  q          ? -INFINITY
                                      : bid_idx[b] >= tail_start ? 1e9f
                                      : 0.0f;
                }

                continue;
            }

            if (dst_bias == nullptr) {
                continue;
            }

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and, when causal, future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                const uint8_t * key_in = &key_seq[seq_id];

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !key_in[st.bid_key[b]*LLAMA_MAX_SEQ]) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // [TAG_QSA_BLOCK_TOPK] the causal cut has to happen before the selection, not
                    // after it. The per-cell expansion this replaced added the attention mask to
                    // the cells and only then ran top-k, so future cells could never be picked;
                    // block-level top-k sees these scores raw, and a block that starts past the
                    // query would take a budget slot on its 1e9 and be masked away afterwards.
                    // A whole prefill ubatch sits past its own first query, so that alone cost
                    // those queries nearly all of their history.
                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = !causal_attn             ? 0.0f
                                    : bid_idx[b] >  q          ? -INFINITY
                                    : bid_idx[b] >= tail_start ? 1e9f
                                    : 0.0f;
                }

                // spare blocks hold unpooled cells (incomplete tails). A spare block is visible
                // and gets the tail value (1e9f) if it contains at least one causally visible cell
                // for this sequence; otherwise -inf so foreign sequence tails are not selected.
                for (int64_t d = 0; d < n_dead; ++d) {
                    cur_blk_bias[n_bid + d] = spare_has(d, seq_id, causal_attn ? q : INT64_MAX) ? 1e9f : -INFINITY;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = st.cell_idx[j];

                    if (!causal_attn) {
                        // every visible block competes on score and the unpooled cells are always selected
                        v = st.blk_of[j] < 0 ? 1e9f : 0.0f;
                    } else if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (st.blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> rep_gen; // per global cell, the generation that last marked a pool with that rep
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.j_next  = 0;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
        }

        // Pools start at the first valid token
        size_t j = sq.j_next;
        while (j + kpool <= sq.cells.size()) {
            const llama_pos p0 = sq.cells[j].first;
            if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                ++j;
                continue;
            }
            bool ok = true;
            for (uint32_t k = 1; k < kpool; ++k) {
                if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                sq.pools.push_back((uint32_t) j);
                j += kpool;
            } else {
                ++j;
            }
        }
        sq.j_next = j;

        lay.n_pool_real += (uint32_t) sq.pools.size();
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

uint32_t llama_memory_hybrid_idx_context::get_s0() const {
    return get_idx() ? get_idx()->get_s0() : 0;
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool causal_attn,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows,
        ggml_tensor * blk_rows,
        const llama_qsa_device_inputs * dev,
        ggml_tensor * scope_rows,
        ggml_tensor * scope_cells,
        ggml_tensor * scope_bias) const {
    GGML_ASSERT(mem != nullptr);
    GGML_ASSERT(get_idx() != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch,
            get_idx()->get_n_kv(), get_n_stream(), ratio, blk_bias, causal_attn,
            dirty_cells, dirty_pos, dirty_rows, blk_rows, dev,
            scope_rows, scope_cells, scope_bias);
}

uint32_t llama_memory_hybrid_idx_context::qsa_scope_n_blocks(const llama_ubatch & ubatch, uint32_t ratio, uint32_t n_seq_vis, uint32_t & n_seq) const {
    n_seq = 0;

    if (mem == nullptr || get_idx() == nullptr || get_n_stream() != 1) {
        return 0;
    }

    return mem->qsa_scope_n_blocks(ubatch, get_idx()->get_n_kv(), ratio, n_seq_vis, n_seq);
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    GGML_ASSERT(k->ne[0] == 3*n_embd);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, 2*n_embd, n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k,   n_embd, n_cells, k->nb[1], ggml_row_size(k->type, 2*n_embd));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

// k-pool DSA indexer (glm5-next)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(),  st.is_new.end(),  0);
        std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);
    st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
    // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const auto & sq  = lay.seqs[s];
        const size_t rep = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + kpool - 1].second;
        if (st.rep_gen[rep] != st.generation) {
            st.rep_gen[rep] = st.generation;
            st.is_new[pool_start[s] + k] = st.generation;
            ++st.n_new;
        }
    };

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j].first + (llama_pos) kpool <= p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_pad(kpool_cur().n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            pcell[ip] = (int32_t) gcell(sq, rep);

            for (uint32_t k = 0; k < kpool; ++k) {
                pidx[(size_t) ip*kpool + k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                nrep[i_new] = gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        std::vector<int64_t> reps(pcell, pcell + pool_end.size());
        std::sort(reps.begin(), reps.end());

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (std::binary_search(reps.begin(), reps.end(), pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            std::fill(row + p0, row + p0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}

ggml_tensor * llama_memory_hybrid_idx_context::get_pooled_k(int32_t il) const {
    return mem != nullptr && get_idx() != nullptr ? mem->get_pooled_k(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::get_pooled_rows() const {
    return mem != nullptr ? mem->get_pooled_rows() : 0;
}

bool llama_memory_hybrid_idx_context::pooled_is_keyed_by_seq() const {
    return mem != nullptr && mem->pooled_is_keyed_by_seq();
}

uint32_t llama_memory_hybrid_idx_context::qsa_pooled_n_dirty_max(const llama_ubatch & ubatch, uint32_t ratio) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr);

    const bool keyed_by_seq = mem->pooled_is_keyed_by_seq();

    // the reserve pass builds worst-case graphs from a mock ubatch with no seq/pos data;
    // give it the per-ubatch bound (the refill after a state load resizes on a live ubatch).
    // Keyed by sequence the worst case is every sequence on its first ubatch after a restore,
    // each repooling its own range, so the bound multiplies rather than adds.
    if (ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr || ubatch.pos == nullptr) {
        return ((ubatch.n_tokens + ratio - 1)/ratio + 1) * (keyed_by_seq ? mem->pooled_n_seq() : 1);
    }

    const uint32_t n_stream = get_n_stream();
    const uint32_t n_tps    = n_stream > 0 ? ubatch.n_tokens / n_stream : ubatch.n_tokens;

    uint32_t max_dirty = 1;

    for (uint32_t s = 0; s < n_stream; ++s) {
        if (s * n_tps >= ubatch.n_tokens) break;

        // keyed by sequence, each sequence in the stream fills its own slice of the tables, so the
        // stream needs the SUM over them; otherwise a stream carries exactly one sequence
        std::vector<llama_seq_id> seqs;
        for (uint32_t i = s * n_tps; i < (s + 1) * n_tps && i < ubatch.n_tokens; ++i) {
            const llama_seq_id sq = ubatch.seq_id[i][0];
            if (std::find(seqs.begin(), seqs.end(), sq) == seqs.end()) {
                seqs.push_back(sq);
            }
            if (!keyed_by_seq) {
                break;
            }
        }

        int64_t stream_dirty = 0;

        for (const llama_seq_id seq : seqs) {
            llama_pos q_max = -1;
            for (uint32_t i = s * n_tps; i < (s + 1) * n_tps && i < ubatch.n_tokens; ++i) {
                if (!keyed_by_seq || ubatch.seq_id[i][0] == seq) {
                    q_max = std::max(q_max, ubatch.pos[i]);
                }
            }

            const auto & cells = mem->get_mem_idx()->get_cells(seq);

            // [TAG_QSA_MROPE_RANK] set_input_qsa buckets a sequence by position, or by rank once an
            // mrope image repeats a position: an image advances the cell count far faster than
            // the position span, so bound by the larger of the two. get_used() would count the
            // whole unified pool, so the bound is taken from this sequence's own cells
            const int64_t n_live = std::max<int64_t>((int64_t) cells.seq_pos_max(seq) + 1,
                                                     (int64_t) cells.seq_pos_cells(seq).size());

            const int64_t n_complete = std::max<int64_t>((int64_t) (q_max + 1)/ratio, n_live/ratio);
            const int64_t w          = std::min(mem->pooled_valid(seq), n_complete);

            stream_dirty += std::max<int64_t>(1, n_complete - w);
        }

        max_dirty = std::max(max_dirty, (uint32_t) std::max<int64_t>(1, stream_dirty));
    }

    return max_dirty;
}
