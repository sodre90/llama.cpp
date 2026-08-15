// Sequence-parallel prefill as a reusable component.
//
// Extracted from the sp-rank driver so that llama-server can drive the same exchange. The rationale
// for WHAT travels, WHERE the barrier goes, and WHY a cell number is not portable lives in
// sp-prefill.cpp, next to the code it constrains.
//
// ONE SESSION PER PROCESS, deliberately. State is file-static rather than an object: a process is one
// rank, and a rank serves one request at a time because sequence-parallel prefill already saturates
// every core on every node. Making this instantiable would add threading risk to code whose only
// real property is that it is bit-exact.
//
// USAGE, in the order the calls must happen:
//     sp::plan(cfg, split, err)      -- pure; no sockets, no model
//     sp::connect(cfg, err)          -- BEFORE the model loads, so peers meet while loads overlap
//     cp.cb_eval = sp::eval_callback -- or call sp::wants/sp::on_tensor from your own callback
//     sp::barrier("pre-decode")      -- shared t0; also proves every peer is alive
//     llama_decode(...)              -- the exchange happens inside, per layer
//     sp::disarm()                   -- BEFORE any decode step; see the note on eval_callback
//     sp::drain()                    -- pay for bytes still queued
#pragma once

#include <cstdio>
#include <string>
#include <vector>

struct ggml_tensor;

namespace sp {

struct config {
    std::vector<std::string> hosts;   // empty means solo: every hook below becomes a no-op
    std::vector<int>         sizes;   // per-rank row counts; empty means a uniform split
    int rank = 0;
    int port = 5400;
    int halo = 128;                   // a split boundary, so a multiple of DSV4_HCA_RATIO
    int rows = 0;
};

// begin/end are the rows this rank OWNS and is the only rank to send. first is where it starts
// COMPUTING: the halo rows in [first, begin) are recomputed locally and their outputs discarded.
struct split {
    int first = 0;
    int begin = 0;
    int end   = 0;
};

struct stats {
    size_t bytes_sent = 0, bytes_recv = 0, rows_applied = 0;
    double send_s = 0, recv_s = 0;    // time lost to the exchange vs time waiting on a peer
    int    exchanges = 0;
};

// A 128-aligned uniform split of `rows` over `world` ranks, spare granules to the low ranks. plan()
// rejects any block that is not a multiple of 128, and rows/world almost never is, so a server
// serving arbitrary prompt lengths must build sizes with this rather than leave config::sizes empty.
std::vector<int> partition(int rows, int world);

// Validates sizes/halo alignment and derives this rank's block. Pure: safe to call before anything
// is loaded, and the server calls it per request to pick a partition.
bool plan(const config & cfg, split & out, std::string & err);

// Ranks connect low-to-high: rank i listens, every j > i dials in. Call before loading the model --
// the connect retries for two minutes, and overlapping it with an 8-11 minute load is free.
bool connect(const config & cfg, const split & sp_split, std::string & err);

// Prefill exchanges KV per layer; decode must not. Disarm between the two, or a rank will block
// forever trying to receive from peers that have finished and closed.
void arm();
void disarm();

bool is_tail();
int  world();

// Compose these into your own cb_eval if you also want tracing; use eval_callback directly if not.
bool wants(const ggml_tensor * t);
bool on_tensor(ggml_tensor * t);
bool eval_callback(ggml_tensor * t, bool ask, void * user);

// One-byte all-to-all on the sockets that already exist. Gives every rank a shared t0, which per-rank
// timings are meaningless without: model load times differ by several seconds across hosts.
void barrier(const char * what);

// Closes the outbox and joins the sender. Returns seconds spent, which is work this rank caused and
// has not yet paid for.
double drain();

// Tears the session down so another connect() can follow: closes sockets, clears counters, re-arms.
// A one-shot driver never needs this; a worker serving request after request does, and forgetting it
// would leave the previous request's sockets in to_higher and fan every layer out to dead peers.
// Call after drain().
void reset();

const stats & get_stats();

// Per-layer activation and cache tracing. Diagnostic only; costs nothing when off.
void set_trace(bool on, FILE * cell_dump);

}  // namespace sp
