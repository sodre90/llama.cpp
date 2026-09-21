#pragma once

// [TAG_CUDA_ADDRMAP] Device address map: every device / pinned-host allocation the CUDA backend
// makes is recorded (base, size, kind) together with a ring of recent frees, and the whole map is
// dumped by ggml_cuda_error before the abort. An NVRM Xid 31 MMU fault in dmesg carries the
// faulting virtual address; matching it against this dump names the buffer (or the freed buffer)
// the kernel was reading. Allocations >= 1 MiB are also logged as they happen (WARN level).

#include <cstddef>

void ggml_cuda_addrmap_alloc(const char * kind, const void * ptr, size_t size);
void ggml_cuda_addrmap_free(const void * ptr);
void ggml_cuda_addrmap_dump(void);
