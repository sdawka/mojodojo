// Plain-C client of libphantom.dylib: proves the driver library is usable from
// any host language. Random int8 matmuls vs a C reference + cycle formula.
#include "phantom.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


static uint64_t s = 0x1234567;
static uint64_t rnd(void) { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1DULL; }

int main(void) {
  int fails = 0, total = 0;
  int tiles[] = {4, 8};
  for (int ti = 0; ti < 2; ti++) {
    int N = tiles[ti];
    int64_t h = phantom_open(N);
    for (int it = 0; it < 6; it++) {
      int64_t M = 1 + rnd() % 21, K = 1 + rnd() % 40, NO = 1 + rnd() % 19;
      int8_t *A = malloc(M * K), *B = malloc(K * NO);
      int32_t *C = malloc(M * NO * 4), *R = calloc(M * NO, 4);
      for (int i = 0; i < M * K; i++) A[i] = (int8_t)rnd();
      for (int i = 0; i < K * NO; i++) B[i] = (int8_t)rnd();
      for (int i = 0; i < M; i++) for (int j = 0; j < NO; j++) for (int k = 0; k < K; k++)
        R[i * NO + j] += (int32_t)A[i * K + k] * (int32_t)B[k * NO + j];
      uint64_t da = phantom_alloc(h, M * K), db = phantom_alloc(h, K * NO), dc = phantom_alloc(h, M * NO * 4);
      phantom_memcpy_h2d(h, da, A, M * K);
      phantom_memcpy_h2d(h, db, B, K * NO);
      int64_t cyc = phantom_launch_matmul_i8(h, da, db, dc, M, K, NO);
      phantom_memcpy_d2h(h, C, dc, M * NO * 4);
      int64_t want = ((M + N - 1) / N) * ((NO + N - 1) / N) * (K + 3 * N - 1);
      int ok = cyc == want;
      for (int i = 0; i < M * NO; i++) ok &= C[i] == R[i];
      total++; fails += !ok;
      printf("%s N=%d M=%lld K=%lld N_out=%lld cycles=%lld formula=%lld\n", ok ? "ok  " : "FAIL",
             N, (long long)M, (long long)K, (long long)NO, (long long)cyc, (long long)want);
      free(A); free(B); free(C); free(R);
    }
    // error path: out-of-bounds copy must fail and set a message
    char buf[256];
    uint64_t small = phantom_alloc(h, 4);
    int64_t st = phantom_memcpy_h2d(h, small, buf, 64);
    phantom_last_error(h, buf, sizeof buf);
    total++; fails += st == 0;
    printf("%s bounds check -> %lld (%s)\n", st ? "ok  " : "FAIL", (long long)st, buf);
    phantom_close(h);
  }
  printf("capi: %d/%d passed\n", total - fails, total);
  return fails != 0;
}
