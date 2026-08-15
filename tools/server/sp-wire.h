// The control protocol between the head and its prefill workers.
//
// Separate from sp-prefill.h because that header is about the KV exchange between ranks, which is a
// property of the model. This is about who tells whom to start, which is a property of the
// deployment. The worker and the head are the only two users.
#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace spwire {

static const uint32_t MAGIC = 0x5350574B;   // "SPWK"

struct request {
    uint32_t magic;
    uint32_t rank, world, rows, halo, port, threads, n_hosts, n_tokens;
};

struct reply {
    uint32_t magic;
    int32_t  rc;
    double   seconds;
};

inline bool read_all(int fd, void * p, size_t n) {
    uint8_t * b = (uint8_t *) p;
    while (n) {
        ssize_t k = recv(fd, b, n, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; }
        b += k; n -= (size_t) k;
    }
    return true;
}

inline bool write_all(int fd, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    while (n) {
        ssize_t k = send(fd, b, n, MSG_NOSIGNAL);
        if (k <= 0) { if (errno == EINTR) continue; return false; }
        b += k; n -= (size_t) k;
    }
    return true;
}

// A blocking connect() is not a liveness probe. A host that is merely down answers with RST and fails
// in microseconds, but a host that is powered off or partitioned swallows the SYN, and the kernel then
// retries for over a minute before returning. That is fine for a benchmark driver waiting on a fleet
// that is still booting; it is fatal for a server deciding, per request, who is alive. Every attempt
// here is bounded, which is what makes degrading onto the surviving ranks worth doing.
inline int dial(const std::string & host, int port, int tries, int timeout_ms = 60000) {
    for (int i = 0; i < tries; i++) {
        const int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return -1;

        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &a.sin_addr);

        const int flags = fcntl(s, F_GETFL, 0);
        fcntl(s, F_SETFL, flags | O_NONBLOCK);

        bool up = connect(s, (sockaddr *) &a, sizeof a) == 0;
        if (!up && errno == EINPROGRESS) {
            pollfd pfd{s, POLLOUT, 0};
            if (poll(&pfd, 1, timeout_ms) == 1) {
                int soerr = 0;
                socklen_t len = sizeof soerr;
                up = getsockopt(s, SOL_SOCKET, SO_ERROR, &soerr, &len) == 0 && soerr == 0;
            }
        }
        if (up) {
            fcntl(s, F_SETFL, flags);
            int one = 1;
            setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            return s;
        }
        close(s);
        usleep(200000);
    }
    return -1;
}

// The token ids travel whole to every worker rather than sliced to each worker's block, because a
// rank also recomputes a halo before its block and the halo is the component's business, not the
// caller's. Sixteen thousand ids is 64 kB; slicing would trade a real correctness coupling for
// nothing.
inline bool send_request(int fd, const request & rq,
                         const std::vector<std::string> & hosts,
                         const std::vector<int> & sizes,
                         const std::vector<int32_t> & tokens) {
    if (!write_all(fd, &rq, sizeof rq)) return false;
    for (const auto & h : hosts) {
        const uint32_t len = (uint32_t) h.size();
        if (!write_all(fd, &len, sizeof len))     return false;
        if (len && !write_all(fd, h.data(), len)) return false;
    }
    if (!sizes.empty()  && !write_all(fd, sizes.data(),  sizes.size()  * sizeof(int)))     return false;
    if (!tokens.empty() && !write_all(fd, tokens.data(), tokens.size() * sizeof(int32_t))) return false;
    return true;
}

}  // namespace spwire
