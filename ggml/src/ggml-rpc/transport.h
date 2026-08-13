#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

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
    // send-all-then-recv-all exchange from deadlocking, see RPC_MESH_MAX_BYTES.
    bool set_buffer_size(size_t bytes);

    // A collective has no reply for the client to time out on, so a dead peer has to surface here.
    bool set_recv_timeout(int seconds);

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port, int backlog = 1);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
