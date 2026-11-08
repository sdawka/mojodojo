// C API of the phantom NPU driver library (libphantom.dylib, built from
// capi.mojo). Device addresses are opaque 64-bit values, not host pointers.
// Every call is synchronous.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int64_t phantom_open(int64_t tile_n); // handle > 0, or < 0 on error
int64_t phantom_close(int64_t h);
uint64_t phantom_alloc(int64_t h, int64_t bytes); // 0 on error
int64_t phantom_free(int64_t h, uint64_t addr);
int64_t phantom_memcpy_h2d(int64_t h, uint64_t dst, const void *src, int64_t n);
int64_t phantom_memcpy_d2h(int64_t h, void *dst, uint64_t src, int64_t n);
// C[m,n] (int32) = A[m,k] (int8) @ B[k,n] (int8); returns cycles, < 0 on error.
int64_t phantom_launch_matmul_i8(int64_t h, uint64_t a, uint64_t b, uint64_t c,
                                 int64_t m, int64_t k, int64_t n);
int64_t phantom_cycles(int64_t h);
int64_t phantom_total_cycles(int64_t h);
int64_t phantom_launches(int64_t h);
/* Execution mode. 0 (default): step the cycle-accurate register model; cycles
 * are measured. 1: functional int32 matmul (same bits); cycles come from the
 * DESIGN.md formula ceil(M/N)*ceil(Nout/N)*(K+3N-1). Returns 0 or -1. */
#define PHANTOM_MODE_CYCLE_ACCURATE 0
#define PHANTOM_MODE_FUNCTIONAL 1
int64_t phantom_set_mode(int64_t h, int64_t mode);
int64_t phantom_get_mode(int64_t h);
int64_t phantom_tile_n(int64_t h);
int64_t phantom_last_error(int64_t h, char *buf, int64_t buflen);
#ifdef __cplusplus
}
#endif
