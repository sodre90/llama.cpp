#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    // Burn up to max_us waiting for the socket to become readable, so that the following recv_data
    // finds the data already there instead of sleeping and being woken. Returns early once readable.
    void spin_until_readable(int64_t max_us);

    socket_ptr accept();

    // Port the socket actually bound to, for a listener created on port 0.
    uint16_t local_port();

    // Sizing the peer receive buffers above the largest message the mesh will carry is what keeps a
    // send-all-then-recv-all exchange from deadlocking, see RPC_MESH_CHUNK_BYTES.
    bool set_buffer_size(size_t bytes);

    // What the kernel actually granted. setsockopt reports success even when net.core.wmem_max clamps
    // the request, so the size that governs the deadlock bound can only be learned by reading it back.
    size_t send_buffer_size();

    // A collective has no reply for the client to time out on, so a dead peer has to surface here.
    bool set_recv_timeout(int seconds);

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    // Set once the peer answers that it keeps no tensor cache, so that set_tensor stops paying a
    // full serial hash pass over every tensor to ask a question whose answer cannot change.
    std::atomic<bool> tensor_cache_absent{false};

    // Held across a whole tensor write, because two loader threads filling different shards can still
    // land on one socket when a peer exposes several devices, and a half-written command is not
    // recoverable. Only the write paths take it, since nothing else runs off the calling thread: graph
    // collectives deliberately split send from recv across sockets, and buffer setup and teardown
    // happen before and after the parallel fill.
    std::mutex cmd_mutex;

    static socket_ptr create_server(const char * host, int port, int backlog = 1);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
