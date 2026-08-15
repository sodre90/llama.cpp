// Sequence-parallel prefill, v2: N ranks each compute their OWN rows only, exchanging KV per layer.
//
// v1 proved the row split is bit-exact but is serial by construction -- rank k waits for k-1 to hand
// over its whole cache -- so it delivers correctness and no speedup. v2 replaces the one whole-cache
// handover with 43 per-layer exchanges, which is what lets every rank run at once.
//
// WHAT GOES ON THE WIRE, AND WHY THAT EXACT SET. Milestone 0 recorded every SET_ROWS of a real
// prefill instead of trusting the "one cpy_kv per layer" assumption in the design: a plain layer
// writes the cache once, an HCA layer four times, a CSA+LID layer eight. Only the four cache_k_l*
// writes travel. The dsv4_*_state_* accumulators stay local -- they are order-dependent running
// scans, not row data, and the N=2 proof already showed a 128-aligned split is bit-exact without
// them. The indexer mask stays local too: its destination is [1, lid_cells, n_ubatch], per-ubatch
// scratch that each rank rebuilds from the LID cache it receives.
//
// WHERE THE BARRIER GOES, taken from the graph rather than chosen. Within a layer every cache write
// precedes every cross-row read, and the main kv-<il> write is the last of them:
//     L2:  SET_ROWS csa -> SET_ROWS lid -> SET_ROWS main -> LIGHTNING_INDEXER -> FLASH_ATTN_EXT
// So one blocking exchange in the post-eval callback of the main write covers all four sub-caches.
//
// CELLS ARE NAMED BY THE GRAPH, NEVER COMPUTED. Each SET_ROWS carries its own index tensor, so the
// payload is literally "the cells these tensors name". Hand arithmetic would be wrong anyway: at
// n_ctx 1792 the CSA sub-cache holds 512 cells, not 1792/4.
//
// BUT A CELL NUMBER IS NOT PORTABLE, which is the bug that failed the first run of this driver. The
// token cache is sliding-window with n_swa=128 on every layer, so reserve_external claims only the
// 128 positions before a rank's block: on rank 1 the prefix lives in cells [0,128) and its OWN rows
// start at cell 128, while on rank 0 those same positions are cells [0,1024). Shipping cell numbers
// verbatim wrote the prefix straight over the receiver's own rows, and the bounds check passed
// because it compared against the cache's physical size rather than against ownership.
//
// So rows travel labelled by POSITION, and each rank translates using only what it can observe about
// its own layout: F = the lowest cell its own graph writes for that sub-cache, which is exactly the
// size of the region reserve_external claimed for it. For the sliding-window token cache
// pos = cell + block_start - F; for a compressed cache pos = cell * ratio, the cell index being the
// window number. Neither side needs to know the other's geometry or n_swa.
//
// THE HALO, which is why a block starts before its own first row. CSA and LID are built with
// overlap=true in dsv4_build_comp_plan, so compressing a window reads the ratio tokens BEFORE it:
// prev_start = source_start - ratio, resolved through state_source_idx, which falls back to the
// persisted compressor ring for any position outside the current ubatch. A rank whose block starts at
// B therefore needs positions [B-ratio, B) that only its predecessor computed. Milestone 0 recorded
// those state tensors and called them rank-local; that was wrong, and it cost max|d| 0.409.
//
// The dependency is bounded -- one window, not a running scan -- so each rank simply RECOMPUTES the
// tokens at its left edge and throws their outputs away. That needs no extra message and no second
// barrier. The halo's own first window is still compressed against a ring the rank does not have, so
// that one window is wrong; it is also the one window its predecessor computed correctly, so the
// exchange overwrites it. That is why compressed slots accept cell <= F while the token cache accepts
// cell < F. HCA is overlap=false and needs none of this.
//
// THE HALO IS 128, NOT ratio. The compressor's reach is 4, so 4 looks sufficient and is not: the halo
// moves where the block STARTS, and a start off the 128 grid breaks HCA's 128-row windows. Halo 4 was
// measured and made things worse (max|d| 0.409 -> 0.464), which was misread as evidence against the
// compressor ring for a whole diagnostic round. A halo is a split boundary and obeys the same rule as
// every other split boundary.
//
// SENDS BLOCK, ON PURPOSE. Every rank sends in ascending receiver order and receives in ascending
// sender order, so the send graph is a DAG with a consistent order and cannot deadlock; a full socket
// buffer just becomes the barrier.
//
// AND THAT BARRIER IS THE ONE THING HERE THAT DOES NOT SCALE. Measured at N=2: rank 1 spent 0.164 s
// across 43 exchanges against 47.6 s of compute, while rank 0 spent 21.05 s of its 57.04 s "in
// exchange". That 21 s is not wire time -- 50 MiB over a measured 1.88 GiB/s fabric is ~30 ms, so it
// is ~700x the transfer -- it is backpressure: the full socket buffer throttles the sender to the
// receiver's per-layer drain rate. At N=10 the lowest rank serialises nine of these and becomes the
// straggler. The fix is to decouple "done computing layer l" from "peer consumed layer l", which a
// sender thread does directly; a tree would instead route traffic THROUGH compute ranks mid-layer and
// couple their schedules, and a symmetric collective would add the shallow-waits-on-deep dependency
// this causal DAG deliberately does not have.
#include "sp-prefill.h"

#include "ggml.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace sp {

static const uint32_t SP_MAGIC = 0x53504C32;   // "SPL2"

// The four cache_k_l* writes, identified by the SOURCE name because all four share the destination
// name cache_k_l<il>. Anything not in this table is rank-local and must not be exchanged.
static int payload_slot(const char * src_name) {
    if (!strncmp(src_name, "kv-",                     3)) return 0;
    if (!strncmp(src_name, "csa_state_compress",     18)) return 1;
    if (!strncmp(src_name, "lid_state_compress_rot", 22)) return 2;
    if (!strncmp(src_name, "hca_state_compress",     18)) return 3;
    return -1;
}

struct pending_write {
    int      slot;
    uint8_t * base;        // destination cache buffer
    size_t   stride;       // bytes between cells
    size_t   row_bytes;    // bytes actually written per cell
    int64_t  n_cells;
    uint32_t max_cell;
    std::vector<uint32_t> cells;
};

struct wire_header { uint32_t magic, layer, n_writes, total_bytes; };
struct wire_write  { uint32_t slot, n_cells, row_bytes, reserved; };

// The claimed region of a sub-cache is [0, F), where F is the lowest cell this rank's own graph
// writes. Everything below is derived from that and from the rank's own block, never from the peer's.
struct cache_geometry {
    uint32_t first_own_cell;
    int64_t  ratio;
};

// DSV4_CSA_RATIO for the CSA and LID caches, DSV4_HCA_RATIO for HCA. These are architectural, and the
// library treats them as constants too; what must never be assumed is a CELL COUNT, which is padded.
static int64_t slot_ratio(int slot) { return slot == 0 ? 1 : (slot == 3 ? 128 : 4); }

// ---------------------------------------------------------------------------------------------

static bool send_all(int fd, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    while (n) {
        ssize_t k = send(fd, b, n, MSG_NOSIGNAL);
        if (k <= 0) { if (errno == EINTR) continue; return false; }
        b += k; n -= (size_t) k;
    }
    return true;
}

static bool recv_all(int fd, void * p, size_t n) {
    uint8_t * b = (uint8_t *) p;
    while (n) {
        ssize_t k = recv(fd, b, n, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; }
        b += k; n -= (size_t) k;
    }
    return true;
}

static void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

static int listen_on(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(port);
    if (bind(s, (sockaddr *) &a, sizeof a) || listen(s, 16)) { perror("bind/listen"); return -1; }
    return s;
}

static int connect_to(const std::string & host, int port, int tries) {
    for (int i = 0; i < tries; i++) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &a.sin_addr);
        if (connect(s, (sockaddr *) &a, sizeof a) == 0) { set_nodelay(s); return s; }
        close(s);
        usleep(500000);
    }
    return -1;
}

// ---------------------------------------------------------------------------------------------

struct sp_state {
    int rank = 0, world = 1;
    int begin = 0, n_rows = 0;
    int own_begin = 0;   // first row this rank OWNS; begin is where it starts COMPUTING
    std::vector<int> to_higher;    // sockets to ranks > rank, ascending
    std::vector<int> from_lower;   // sockets from ranks < rank, ascending

    int cur_layer = -1;
    std::vector<pending_write> pend;
    std::vector<uint8_t> rbuf;

    // The sender runs on its own thread so that finishing a layer never waits on a peer draining it.
    // A layer's payload is identical for every higher rank, so one buffer is queued and fanned out.
    std::deque<std::vector<uint8_t>> outbox;
    std::mutex              out_mu;
    std::condition_variable out_cv;
    bool                    out_closed = false;
    std::thread             sender;

    stats st;
};

static sp_state g;
static bool  g_armed = true;

// A peer dying used to call exit(1), which is right for a benchmark driver and fatal for a server:
// one worker hiccup took the whole process down. Instead the session is marked failed, every
// subsequent exchange becomes a no-op so the in-flight llama_decode can unwind instead of blocking on
// a dead socket, and the caller turns that into a failed REQUEST. The decode's output is meaningless
// once this is set, which is exactly why the caller must check it rather than trust rc.
static bool g_failed = false;

static void fail(const char * fmt, ...) {
    if (!g_failed) {
        va_list ap;
        va_start(ap, fmt);
        fprintf(stderr, "!! sp: ");
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        va_end(ap);
    }
    g_failed = true;
}

bool failed() { return g_failed; }

static bool  g_trace = false;
static FILE *g_dump  = nullptr;   // per-cell hashes, so a diff names the exact cell rather than a layer
static int   g_occ[64] = {0};

void set_trace(bool on, FILE * cell_dump) { g_trace = on; g_dump = cell_dump; }
void arm()      { g_armed = true;  }
void disarm()   { g_armed = false; }
bool is_tail()  { return g.rank == g.world - 1; }
int  world()    { return g.world; }
const stats & get_stats() { return g.st; }

static bool traced_name(const char * n) {
    static const char * const WANT[] = { "attn_wo_a-", "ffn_norm-", "hc_attn_pre-", "hc_pre-",
                                         "csa_state_kv-", "csa_state_compress-", "l_last-" };
    for (const char * w : WANT) if (!strncmp(n, w, strlen(w))) return true;
    return false;
}

// A SUM is not a usable checksum on these tensors. The compressor buffers are sized per token while
// only every ratio'th window is real, and the dead lanes hold an -inf mask sentinel; one -inf makes
// the sum -inf on BOTH sides, so any difference in the live data prints as a match. That is exactly
// how csa_state_kv-2 was read as identical while its consumer diverged. Hash the live lanes instead,
// and print how many there are, so a differing lane count is itself visible rather than silent.
static void trace_tensor(const ggml_tensor * t) {
    // A flat walk over ggml_nelements is only valid on a contiguous tensor; on a strided view it
    // reads padding, which is how hc_mixes appeared to diverge while its layer's output did not.
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return;
    const float * v = (const float *) t->data;
    const int64_t n = ggml_nelements(t);
    uint64_t h = 1469598103934665603ULL;
    int64_t n_fin = 0;
    double sum = 0, amax = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!std::isfinite(v[i])) continue;
        n_fin++;
        sum += v[i];
        if (fabs(v[i]) > amax) amax = fabs(v[i]);
        uint32_t bits; memcpy(&bits, &v[i], 4);
        for (int b = 0; b < 4; b++) { h ^= (bits >> (b * 8)) & 0xff; h *= 1099511628211ULL; }
    }
    const int il = atoi(strrchr(t->name, '-') + 1);
    printf("== TRACE %-14s l%-3d occ%d  n=%-9" PRId64 " live=%-9" PRId64 " h=%016" PRIx64 " sum=%+.6f amax=%.6f\n",
           t->name, il, ++g_occ[il & 63], n, n_fin, h, sum, amax);
}

static int layer_of(const char * dst_name) {
    const char * p = strstr(dst_name, "cache_k_l");
    return p ? atoi(p + 9) : -1;
}

static void record(ggml_tensor * t, int slot) {
    const ggml_tensor * idx = t->src[1];
    pending_write w;
    w.slot      = slot;
    w.base      = (uint8_t *) t->data;
    w.stride    = t->nb[1];
    w.row_bytes = ggml_row_size(t->type, t->ne[0]);
    w.n_cells   = idx->ne[0];
    w.max_cell  = (uint32_t) t->ne[1];
    w.cells.resize((size_t) w.n_cells);
    if (idx->type == GGML_TYPE_I64) {
        const int64_t * v = (const int64_t *) idx->data;
        for (int64_t i = 0; i < w.n_cells; i++) w.cells[i] = (uint32_t) v[i];
    } else {
        const int32_t * v = (const int32_t *) idx->data;
        for (int64_t i = 0; i < w.n_cells; i++) w.cells[i] = (uint32_t) v[i];
    }
    g.pend.push_back(std::move(w));
}

static cache_geometry geometry_of(const pending_write & w) {
    cache_geometry cg;
    cg.first_own_cell = w.cells[0];
    for (uint32_t c : w.cells) cg.first_own_cell = std::min(cg.first_own_cell, c);
    cg.ratio = slot_ratio(w.slot);
    return cg;
}

// Slot 0 is the sliding-window token cache: its claimed region holds the F positions immediately
// before the block. Every other slot is a compressed cache whose cell index IS the window index.
static int64_t position_of(const pending_write & w, const cache_geometry & cg, uint32_t cell) {
    return w.slot == 0 ? (int64_t) cell + g.begin - cg.first_own_cell
                       : (int64_t) cell * cg.ratio;
}

static int64_t cell_of(const pending_write & w, const cache_geometry & cg, int64_t pos) {
    if (w.slot == 0) return pos - (g.begin - (int64_t) cg.first_own_cell);
    return pos % cg.ratio ? -1 : pos / cg.ratio;
}

// The token cache's claimed region is [0, F). A compressed cache also accepts F itself: that is the
// halo's own first window, which this rank computed without the ring it needed and the sender did not.
static bool in_claimed_region(int slot, int64_t c, uint32_t first_own_cell) {
    return c >= 0 && c < (int64_t) first_own_cell + (slot == 0 ? 0 : 1);
}

// Hash the cache region a rank did NOT compute, keyed by position so a rank and the reference are
// directly comparable even though their cell numbering differs. This answers the question the logit
// gate cannot: are the received bytes right, or is the bug in how they are read?
// The token cache keyed by POSITION rather than by cell. Cell numbers are not comparable across ranks
// -- reserve_external's claim offsets rank 1's numbering -- but positions are, and this is the only
// measurement that separates "the received bytes are wrong" from "the same bytes are read in a
// different order". Two hypotheses for the decode divergence died without it: the compressor state
// ring (refuted, the first divergence is at layer 0, which is raw-only) and n_ubatch (refuted,
// matching it left max|delta| identical to ten significant figures).
static void trace_token_cache(int il, const pending_write & w, const cache_geometry & cg) {
    const int64_t end_pos = g.begin + g.n_rows;
    const auto rowhash = [&](int64_t pos) {
        const int64_t c = cell_of(w, cg, pos);
        if (c < 0 || c >= (int64_t) w.max_cell) return (uint64_t) 0;
        const uint8_t * b = w.base + (size_t) c * w.stride;
        uint64_t h = 1469598103934665603ULL;
        for (size_t k = 0; k < w.row_bytes; k++) { h ^= b[k]; h *= 1099511628211ULL; }
        return h;
    };
    // The last 128 positions are exactly the sliding window a decode token at end_pos will read.
    uint64_t win = 1469598103934665603ULL;
    for (int64_t p = end_pos - 128; p < end_pos; p++) {
        const uint64_t rh = rowhash(p);
        win ^= rh; win *= 1099511628211ULL;
    }
    printf("== TOKCACHE l%-3d window[%" PRId64 ",%" PRId64 ") h=%016llx  first=%016llx last=%016llx\n",
           il, end_pos - 128, end_pos, (unsigned long long) win,
           (unsigned long long) rowhash(end_pos - 128), (unsigned long long) rowhash(end_pos - 1));
}

static void trace_cache(int il) {
    for (const auto & w : g.pend) {
        const cache_geometry cg = geometry_of(w);
        if (w.slot == 0) { trace_token_cache(il, w, cg); continue; }
        uint32_t last_own = 0;
        for (uint32_t c : w.cells) last_own = std::max(last_own, c);
        // A compressed cell index IS the global window number, so these ranges mean the same thing on
        // every rank and on the reference. [0,F) is what this rank received; [0,last] adds what it
        // computed itself, which is where a missing compressor ring would show up.
        const auto chk = [&](uint32_t lo, uint32_t hi) {
            uint64_t h = 1469598103934665603ULL;
            for (uint32_t c = lo; c < hi; c++) {
                const uint8_t * b = w.base + (size_t) c * w.stride;
                for (size_t k = 0; k < w.row_bytes; k++) { h ^= b[k]; h *= 1099511628211ULL; }
            }
            return h;
        };
        printf("== CACHE l%-3d slot%d F=%-5u last=%-5u recv=%016llx all=%016llx\n",
               il, w.slot, cg.first_own_cell, last_own,
               (unsigned long long) chk(0, cg.first_own_cell),
               (unsigned long long) chk(0, last_own + 1));
        if (g_dump) {
            for (uint32_t c = 0; c <= last_own; c++) {
                const int32_t  hdr[4] = { il, w.slot, (int32_t) c, 0 };
                const uint64_t h      = chk(c, c + 1);
                fwrite(hdr, sizeof hdr, 1, g_dump);
                fwrite(&h,  sizeof h,   1, g_dump);
            }
        }
    }
}

// Drains the outbox in FIFO order, so each socket still sees layers strictly in order and the
// receiver's header check on the layer number stays valid.
static void sender_loop() {
    for (;;) {
        std::vector<uint8_t> buf;
        {
            std::unique_lock<std::mutex> lk(g.out_mu);
            g.out_cv.wait(lk, [] { return g.out_closed || !g.outbox.empty(); });
            if (g.outbox.empty()) return;               // closed and drained
            buf = std::move(g.outbox.front());
            g.outbox.pop_front();
        }
        for (int fd : g.to_higher) {
            if (!send_all(fd, buf.data(), buf.size())) { fail("send failed"); return; }
            g.st.bytes_sent += buf.size();
        }
    }
}

// A one-byte all-to-all on the sockets that already exist. Ranks connect before loading the model and
// the load time varies by several seconds between hosts, so without this the ranks enter llama_decode
// at different wall-clock times and every per-rank duration is measured against a different t0 -- the
// N=2 run's 57.04 s and 47.6 s only reconcile through a ~9 s load skew. One byte never fills a socket
// buffer, so sending to everyone before reading from anyone cannot deadlock.
void barrier(const char * what) {
    if (g.world == 1 || g_failed) return;
    const uint8_t tok = 0xB0;
    for (int fd : g.to_higher)  if (!send_all(fd, &tok, 1)) { fail("barrier send failed"); return; }
    for (int fd : g.from_lower) if (!send_all(fd, &tok, 1)) { fail("barrier send failed"); return; }
    uint8_t got = 0;
    for (int fd : g.to_higher)  if (!recv_all(fd, &got, 1)) { fail("barrier recv failed"); return; }
    for (int fd : g.from_lower) if (!recv_all(fd, &got, 1)) { fail("barrier recv failed"); return; }
    printf("== barrier %s passed\n", what);
}

static void flush_layer(int il) {
    if (g.world == 1 || g_failed) { if (g_trace) trace_cache(il); g.pend.clear(); return; }
    const auto t0 = std::chrono::steady_clock::now();

    if (!g.to_higher.empty()) {
        // SEND ONLY WHAT THIS RANK OWNS. A rank computes `halo` rows before its own block, and those
        // rows are also owned -- and computed correctly -- by its predecessor. Shipping them makes a
        // higher rank apply the same positions twice, and because senders are applied in ascending
        // rank order the halo copy lands LAST and overwrites the owner's. It is also the worse copy:
        // the halo's first compressed window is exactly the one built against a ring this rank does
        // not have. Invisible at N=2, where rank 0 has no halo and rank 1 sends to nobody.
        std::vector<std::vector<uint32_t>> owned(g.pend.size());
        size_t n = sizeof(wire_header);
        for (size_t i = 0; i < g.pend.size(); i++) {
            const auto & w = g.pend[i];
            const cache_geometry cg = geometry_of(w);
            for (uint32_t c : w.cells) {
                if (position_of(w, cg, c) >= g.own_begin) owned[i].push_back(c);
            }
            n += sizeof(wire_write) + owned[i].size() * (sizeof(int64_t) + w.row_bytes);
        }
        std::vector<uint8_t> buf(n);
        uint8_t * p = buf.data();
        wire_header h{SP_MAGIC, (uint32_t) il, (uint32_t) g.pend.size(), (uint32_t) n};
        memcpy(p, &h, sizeof h); p += sizeof h;
        for (size_t i = 0; i < g.pend.size(); i++) {
            const auto & w = g.pend[i];
            const cache_geometry cg = geometry_of(w);
            wire_write ww{(uint32_t) w.slot, (uint32_t) owned[i].size(), (uint32_t) w.row_bytes, 0};
            memcpy(p, &ww, sizeof ww); p += sizeof ww;
            int64_t * pos = (int64_t *) p; p += owned[i].size() * sizeof(int64_t);
            for (size_t k = 0; k < owned[i].size(); k++) pos[k] = position_of(w, cg, owned[i][k]);
            for (uint32_t c : owned[i]) { memcpy(p, w.base + (size_t) c * w.stride, w.row_bytes); p += w.row_bytes; }
        }
        {
            std::lock_guard<std::mutex> lk(g.out_mu);
            g.outbox.push_back(std::move(buf));
        }
        g.out_cv.notify_one();
    }
    const auto t1 = std::chrono::steady_clock::now();
    g.st.send_s += std::chrono::duration<double>(t1 - t0).count();

    for (int fd : g.from_lower) {
        wire_header h{};
        if (!recv_all(fd, &h, sizeof h) || h.magic != SP_MAGIC || (int) h.layer != il) {
            fail("bad header at layer %d (magic %08x layer %u)", il, h.magic, h.layer);
            return;
        }
        g.rbuf.resize(h.total_bytes - sizeof h);
        if (!recv_all(fd, g.rbuf.data(), g.rbuf.size())) { fail("recv body failed"); return; }
        g.st.bytes_recv += h.total_bytes;

        const uint8_t * p = g.rbuf.data();
        for (uint32_t i = 0; i < h.n_writes; i++) {
            wire_write ww; memcpy(&ww, p, sizeof ww); p += sizeof ww;
            const int64_t * pos = (const int64_t *) p; p += (size_t) ww.n_cells * sizeof(int64_t);
            const pending_write * dst = nullptr;
            for (const auto & w : g.pend) if (w.slot == (int) ww.slot) { dst = &w; break; }
            if (!dst || dst->row_bytes != ww.row_bytes) {
                fail("layer %d slot %u has no matching local write", il, ww.slot);
                return;
            }
            const cache_geometry cg = geometry_of(*dst);
            for (uint32_t k = 0; k < ww.n_cells; k++, p += ww.row_bytes) {
                const int64_t c = cell_of(*dst, cg, pos[k]);
                if (!in_claimed_region(dst->slot, c, cg.first_own_cell)) continue;
                memcpy(dst->base + (size_t) c * dst->stride, p, ww.row_bytes);
                g.st.rows_applied++;
            }
        }
    }

    if (g_trace) trace_cache(il);
    g.pend.clear();
    g.st.recv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    g.st.exchanges++;
}

bool wants(const ggml_tensor * t) {
    return (g_armed && t->op == GGML_OP_SET_ROWS && layer_of(t->name) >= 0) ||
           (g_trace && traced_name(t->name));
}

bool on_tensor(ggml_tensor * t) {
    if (t->op != GGML_OP_SET_ROWS) { trace_tensor(t); return true; }
    if (!g_armed) return true;
    const int slot = payload_slot(t->src[0]->name);
    if (slot < 0) return true;                 // a cache_k write we deliberately do not exchange
    const int il = layer_of(t->name);
    if (il != g.cur_layer) { g.pend.clear(); g.cur_layer = il; }
    record(t, slot);
    if (slot == 0) flush_layer(il);            // the main kv write is the last one in every layer
    return true;
}

bool eval_callback(ggml_tensor * t, bool ask, void *) {
    return ask ? wants(t) : on_tensor(t);
}

double drain() {
    const auto t0 = std::chrono::steady_clock::now();
    if (g.sender.joinable()) {
        { std::lock_guard<std::mutex> lk(g.out_mu); g.out_closed = true; }
        g.out_cv.notify_one();
        g.sender.join();
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void reset() {
    drain();                       // idempotent: a joined sender is not joinable
    for (int fd : g.to_higher)  close(fd);
    for (int fd : g.from_lower) close(fd);
    g.to_higher.clear();
    g.from_lower.clear();
    g.pend.clear();
    g.rbuf.clear();
    g.outbox.clear();
    g.out_closed = false;
    g.cur_layer  = -1;
    g.st         = stats{};
    // Leave the hook DISARMED. A server installs one cb_eval for its whole life and runs ordinary
    // single-node requests through it between distributed ones; re-arming here would leave the next
    // ordinary request armed with a stale world > 1 and hang it against peers that no longer exist.
    // connect() arms, which ties armed-ness to having a session rather than to having been reset.
    g_armed      = false;
    g_failed     = false;
    for (int & o : g_occ) o = 0;
}

// ---------------------------------------------------------------------------------------------

bool plan(const config & cfg, split & out, std::string & err) {
    const int world = cfg.hosts.empty() ? 1 : (int) cfg.hosts.size();
    char buf[256];

    // The fleet is 34% heterogeneous, so equal blocks make the slowest node pace everyone. sizes
    // takes the per-rank row counts from sp-schedule.py's solve_partition; without it the split is
    // uniform, which is only right on a homogeneous fleet.
    if (!cfg.sizes.empty()) {
        if ((int) cfg.sizes.size() != world) {
            snprintf(buf, sizeof buf, "sizes has %zu entries for %d ranks", cfg.sizes.size(), world);
            err = buf; return false;
        }
        int sum = 0;
        for (int s : cfg.sizes) {
            if (s % 128) { snprintf(buf, sizeof buf, "block %d is not a multiple of 128", s); err = buf; return false; }
            sum += s;
        }
        if (sum != cfg.rows) {
            snprintf(buf, sizeof buf, "sizes sums to %d, not %d", sum, cfg.rows);
            err = buf; return false;
        }
    }
    const int chunk = cfg.rows / world;
    if (cfg.sizes.empty() && chunk % 128) {
        snprintf(buf, sizeof buf, "chunk %d is not a multiple of 128", chunk); err = buf; return false;
    }
    if (cfg.halo % 128) {
        snprintf(buf, sizeof buf, "halo %d is not a multiple of 128; it moves the block start off the HCA grid", cfg.halo);
        err = buf; return false;
    }

    if (cfg.sizes.empty()) {
        out.begin = cfg.rank * chunk;
        out.end   = (cfg.rank == world - 1) ? cfg.rows : out.begin + chunk;
    } else {
        out.begin = 0;
        for (int j = 0; j < cfg.rank; j++) out.begin += cfg.sizes[j];
        out.end = out.begin + cfg.sizes[cfg.rank];
    }
    out.first = out.begin - (cfg.rank ? cfg.halo : 0);   // the overlap compressor's reach, recomputed
    return true;
}

std::vector<int> partition(int rows, int world) {
    // Blocks must be 128-aligned, so the unit of division is a granule of 128 rows, never a row.
    // Spare granules go to the LOW ranks because prefill cost grows with depth: a deep block costs
    // more per row than a shallow one, so the cheap direction to add work is downward. This is a
    // uniform split, not the speed-weighted one solve_partition() produces -- on a heterogeneous
    // fleet the solved partition is worth roughly 10%, and --sp-sizes exists to pass it in.
    std::vector<int> out(world, 0);
    if (world <= 0 || rows < 128) return out;
    const int granules = rows / 128;
    for (int i = 0; i < world; i++) {
        out[i] = (granules / world + (i < granules % world ? 1 : 0)) * 128;
    }
    return out;
}

bool connect(const config & cfg, const split & s, std::string & err) {
    g_armed     = true;   // armed-ness tracks having a session, so reset() can safely leave it off
    g.rank      = cfg.rank;
    g.world     = cfg.hosts.empty() ? 1 : (int) cfg.hosts.size();
    g.begin     = s.first;
    g.own_begin = s.begin;
    g.n_rows    = s.end - s.first;

    if (g.world == 1) return true;

    // Ranks connect low-to-high: rank i listens, every j > i dials in. Data only ever flows upward,
    // because attention is causal and a rank never needs rows after its own.
    int srv = -1;
    // Every exit runs through this, because a caller that retries after a half-built session would
    // otherwise fan the next request out over both the new sockets and the stale ones.
    auto abandon = [&](std::string why) {
        err = std::move(why);
        if (srv >= 0) close(srv);
        reset();
        return false;
    };

    if (g.rank < g.world - 1) {
        srv = listen_on(cfg.port + g.rank);
        if (srv < 0) return abandon("listen failed");
    }
    for (int j = 0; j < g.rank; j++) {
        const int c = connect_to(cfg.hosts[j], cfg.port + j, 120);
        if (c < 0) return abandon("could not reach " + cfg.hosts[j]);
        g.from_lower.push_back(c);
    }
    // A bare accept() waits forever, so one rank that never arrives hangs every rank below it -- and
    // the head, which is still inside a request. Ranks that are coming are already loaded and arrive
    // within seconds of each other, so anything past a minute is a rank that is not coming.
    for (int j = g.rank + 1; j < g.world; j++) {
        pollfd pfd{srv, POLLIN, 0};
        if (poll(&pfd, 1, 60000) != 1) return abandon("timed out waiting for a higher rank to dial in");
        const int c = accept(srv, nullptr, nullptr);
        if (c < 0) return abandon("accept failed");
        set_nodelay(c);
        g.to_higher.push_back(c);
    }
    if (srv >= 0) close(srv);
    g.sender = std::thread(sender_loop);
    return true;
}

}  // namespace sp
