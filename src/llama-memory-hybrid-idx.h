#pragma once

#include "llama-memory-hybrid.h"

#include "ggml-backend.h"

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

// TODO: this memory module is pending complete reimplementation - do not use for model other than Qwen4

// [TAG_QSA_DEVICE_INPUTS] tables the graph derives the per-query QSA bias and visibility from,
// in place of inputs that grow with n_tokens times the context. Any of them may be null.
//   blk_start F32 [n_blocks, n_seq, ns]  first causal index of each block as seen by each sequence:
//                                        bid_idx for a block the sequence shares, the lowest visible
//                                        index of a spare block, +1e30 otherwise
//   blk_spare F32 [n_blocks, 1, ns]      1 for a spare block (unpooled cells), else 0
//   tok_seq   I32 [n_tps, ns]            sequence of each token
//   tok_q     F32 [1, n_tps, ns]         causal index of each token (its rank under an image, else its position)
//   tok_m     F32 [1, n_tps, ns]         (q + 1) % ratio: how far the token's own incomplete block reaches back
//   cell_idx  I32 [n_kv, n_seq, 1, ns]   causal index of each cell per sequence, INT32_MAX when empty or foreign
//   q_meta    I32 [2, n_tps, 1, ns]      (sequence, causal index) of each token
//   zero_mask F16 [n_kv, 1, 1, 1]        the one mask row the attention op reads at visible cells
struct llama_qsa_device_inputs {
    ggml_tensor * blk_start = nullptr;
    ggml_tensor * blk_spare = nullptr;
    ggml_tensor * tok_seq   = nullptr;
    ggml_tensor * tok_q     = nullptr;
    ggml_tensor * tok_m     = nullptr;

    ggml_tensor * cell_idx  = nullptr;
    ggml_tensor * q_meta    = nullptr;
    ggml_tensor * zero_mask = nullptr;
};

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
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
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx,
                 uint32_t   kv_unified_per_slot = 0);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // causal_attn selects the rule: causal forces the query's own block on, non-causal lets every visible block compete on score
    // n_kv/n_stream are the graph's, not the cache's: get_n_kv() is a padded live window well
    // below cells.size(), and every tensor here was sized from it
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch,
                       uint32_t n_kv, uint32_t n_stream, uint32_t ratio,
                       bool blk_bias, bool causal_attn,
                       ggml_tensor * dirty_cells = nullptr,
                       ggml_tensor * dirty_pos   = nullptr,
                       ggml_tensor * dirty_rows  = nullptr,
                       ggml_tensor * blk_rows    = nullptr,
                       const llama_qsa_device_inputs * dev = nullptr) const;

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Which cells of a sequence make up which pool of kpool consecutive positions.
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

    // [TAG_QSA_POOLED_CACHE] cache of the indexer's block summary keys (mean-pooled,
    // normalized, roped), one f32 row per position block, written by the graph via set_rows.
    // Only complete blocks are scored and a complete block's members never change, so rows are
    // write-once per content epoch. Validity is a per-sequence block watermark; seq_rm clamps
    // it and replay recomputes the range. Rows at or beyond the watermark may hold stale but
    // finite data and are masked by the -inf bias. Single-stream memories only.
    ggml_tensor * get_pooled_k(int32_t il) const;              // nullptr: no indexer / multi-stream
    uint32_t get_pooled_rows() const { return pooled_rows; }   // rows per stream, incl. trailing dustbin row

    // A unified cache holds every sequence in one stream, so the position block alone no longer
    // identifies a row: two agents at the same positions would share it and overwrite each other.
    // Rows are then keyed on (seq_id, position block) and the reader gathers rather than viewing
    // a contiguous range. With one sequence per stream the base is 0 and nothing changes.
    bool     pooled_is_keyed_by_seq() const { return pooled_seq_stride > 0; }
    uint32_t get_pooled_seq_stride()  const { return pooled_seq_stride; }

    int64_t pooled_row_base(llama_seq_id seq_id) const {
        return (int64_t) pooled_seq_stride * seq_id;
    }

    // row holding block blk of seq_id, or the shared dustbin when blk is past the range reserved
    // for one sequence. A block scored from the dustbin is wrong for that block alone; a row taken
    // from the next sequence's range would corrupt another agent. The server admits no prompt long
    // enough to reach this, so it is a bound on --kv-unified-per-slot, not a live path.
    int64_t pooled_row_of(llama_seq_id seq_id, int64_t blk) const {
        const int64_t lim = pooled_seq_stride > 0 ? (int64_t) pooled_seq_stride : (int64_t) pooled_rows - 1;

        if (blk < 0 || blk >= lim) {
            return (int64_t) pooled_rows - 1;
        }

        return pooled_row_base(seq_id) + blk;
    }

    // sequences the store has row ranges for; 1 when a stream owns one sequence
    uint32_t pooled_n_seq() const {
        return pooled_seq_stride > 0 ? (pooled_rows - 1)/pooled_seq_stride : 1;
    }

    // blocks of seq_id whose pooled rows are known valid; mutable like a cache's bookkeeping
    int64_t & pooled_valid(llama_seq_id seq_id) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();

    // [TAG_QSA_POOLED_CACHE] storage + watermarks; empty unless the model has an indexer
    // one buffer per device: each layer's rows must live with that layer's indexer cache,
    // or every decode step would copy the rows across the (slow) inter-GPU links
    std::vector<ggml_context_ptr>        pooled_ctxs;
    std::vector<ggml_backend_buffer_ptr> pooled_bufs;
    std::map<int32_t, ggml_tensor *> pooled_k;

    uint32_t pooled_rows  = 0;
    uint32_t pooled_ratio = 0;

    // rows reserved per sequence when the cache is unified; 0 when a stream owns one sequence
    // and the position block indexes the store directly
    uint32_t pooled_seq_stride = 0;

    // set once seq_cp has put one block's cells in several sequences: their rows are then no
    // longer independent, so a later removal has to invalidate every watermark, not just one
    bool pooled_shared = false;

    mutable std::unordered_map<llama_seq_id, int64_t> pooled_w;

    // [TAG_QSA_INPUT_STATE] what set_input_qsa derives from the cells of one stream. With a single
    // stream it is kept between ubatches and updated from the cells' journal, so a decode step
    // touches its few new cells instead of rescanning the pool
    struct qsa_input_state {
        bool valid = false;                     // true: can be updated in place
        const llama_kv_cells * cells = nullptr;
        uint64_t mark = 0;                      // journal position the state reflects
        int64_t  n_kv = 0;
        int64_t  r    = 0;
        int64_t  n_seq_vis = 0;

        std::vector<llama_seq_id>              seq_present;
        std::vector<llama_kv_cells::seq_set_t> key_set;

        std::vector<int32_t> cell_key;          // -1: empty
        std::vector<int32_t> cell_idx;          // causal index of a non-empty cell
        std::vector<int32_t> blk_of;            // only when a per-cell table needs it

        std::vector<int64_t>  grp_ext;
        std::vector<int64_t>  grp_base;
        std::vector<int32_t>  grp_first;
        std::vector<int32_t>  grp_slot0;
        std::vector<uint64_t> grp_slots;
        std::vector<int32_t>  grp_bid;

        std::vector<int32_t> bid_idx;
        std::vector<int32_t> bid_cell;
        std::vector<int32_t> bid_slot0;
        std::vector<int32_t> bid_key;
        std::vector<int64_t> bid_grp;
        std::vector<int32_t> bid_pos;           // [4] per bid
        std::vector<int32_t> bid_cells;         // [r] per bid

        std::vector<int32_t> unpooled;          // ascending
        int32_t pad = -1;                       // first empty cell

        std::vector<int32_t> vis;               // [n_kv] per sequence row

        bool oor    = false;
        bool dup    = false;
        bool ranked = false;
        llama_kv_cells::seq_set_t ranked_seqs;
        std::vector<int32_t> rank;
        std::vector<std::vector<int32_t>> seq_order;

        std::vector<uint8_t>      key_seq;      // [n_key*LLAMA_MAX_SEQ] whether key k's cells belong to sequence sq
        std::vector<llama_seq_id> key_low;      // lowest sequence of key k
    };

    mutable qsa_input_state qsa_st;

    // brings st up to date with cells: in place when a decode step only added cells, else rebuilt.
    // keep: st outlives this ubatch and may be updated in place next time
    void qsa_sync(qsa_input_state & st, const llama_kv_cells & cells, const llama_ubatch * ubatch,
                  int64_t n_kv, int64_t r, int64_t n_seq_vis, bool keep, bool need_blk_of, bool blk_bias) const;

    // clamp helpers, one per llama_memory_i operation that can invalidate rows
    void pooled_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void pooled_reset(llama_seq_id seq_id);   // -1 resets every sequence
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;
    uint32_t get_s0() const;

    // glm5-next, complete pools of kpool consecutive positions per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    // sel_mask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the live selection slots, -inf for the dead ones
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

    // [TAG_QSA_POOLED_CACHE] the dirty_* tensors are optional: when given, the fill also
    // resolves which blocks must be (re)pooled this ubatch — the range from the sequence's
    // watermark to its last complete block — and advances the watermark.
    //   dirty_cells I32 [ratio*n_dirty_max, ns] cells of each block to (re)pool, 0-padded
    //   dirty_pos   I32 [4*n_dirty_max*ns]      mrope position rows of those blocks
    //   dirty_rows  I64 [n_dirty_max*ns]        pooled-cache rows to write, dustbin-padded
    //   blk_rows    I32 [n_blocks*ns]           pooled-cache row each block reads, dustbin-padded
    //                                           (unified only; otherwise the reader views a range)
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool causal_attn,
                       ggml_tensor * dirty_cells = nullptr,
                       ggml_tensor * dirty_pos   = nullptr,
                       ggml_tensor * dirty_rows  = nullptr,
                       ggml_tensor * blk_rows    = nullptr,
                       const llama_qsa_device_inputs * dev = nullptr) const;

    // [TAG_QSA_POOLED_CACHE] true when store rows are keyed on (seq_id, position block) rather
    // than the position block alone, which a unified pool requires - see the memory class
    bool pooled_is_keyed_by_seq() const;

    // [TAG_QSA_POOLED_CACHE] pooled tensor for il, or nullptr when the cache is unavailable
    // (no indexer, multi-stream memory, or a non-batch context)
    ggml_tensor * get_pooled_k(int32_t il) const;

    uint32_t get_pooled_rows() const;

    // capacity the dirty tables need for this ubatch: completed blocks plus pending refill
    // below the watermark; stable at 1 during steady decode so graph reuse holds
    uint32_t qsa_pooled_n_dirty_max(const llama_ubatch & ubatch, uint32_t ratio) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
