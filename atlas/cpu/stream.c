// CPU memory bandwidth: STREAM-like read / write / copy / triad over large arrays, multithreaded.
// Bytes counted as program-visible traffic (read=1x, write=1x, copy=2x, triad=3x per element);
// write-allocate traffic is NOT counted (Apple cores may avoid it for streaming stores anyway).
#include "common.h"
#include <arm_neon.h>

typedef struct {
  double *a, *b, *c;
  size_t n;
  int op;  // 0 read, 1 write, 2 copy, 3 triad
  double sink[64];
} sargs_t;

static void kern(int tid, int nt, void *p) {
  sargs_t *s = (sargs_t *)p;
  size_t chunk = (s->n / nt) & ~(size_t)63;
  size_t lo = (size_t)tid * chunk, hi = (tid == nt - 1) ? s->n : lo + chunk;
  double *a = s->a, *b = s->b, *c = s->c;
  switch (s->op) {
    case 0: {  // read: 8 independent vector accumulators
      float64x2_t acc[8];
      for (int k = 0; k < 8; k++) acc[k] = vdupq_n_f64(0);
      size_t i = lo;
      for (; i + 16 <= hi; i += 16)
        for (int k = 0; k < 8; k++) acc[k] = vaddq_f64(acc[k], vld1q_f64(a + i + 2 * k));
      double t = 0;
      for (int k = 0; k < 8; k++) t += vaddvq_f64(acc[k]);
      for (; i < hi; i++) t += a[i];
      s->sink[tid] = t;
      break;
    }
    case 1:
      for (size_t i = lo; i < hi; i++) a[i] = 1.0;
      break;
    case 2:
      for (size_t i = lo; i < hi; i++) c[i] = a[i];
      break;
    case 3:
      for (size_t i = lo; i < hi; i++) a[i] = b[i] + 3.0 * c[i];
      break;
  }
}

int main(int argc, char **argv) {
  // Default 3 x 256 MiB arrays (>> 16 MiB P-L2 + SLC) unless ATLAS_SMOKE.
  size_t mib = (size_t)env_int("ATLAS_STREAM_MIB", atlas_smoke() ? 64 : 256);
  size_t n = mib * 1024 * 1024 / sizeof(double);
  int reps = atlas_reps(), warm = atlas_warmup();
  sargs_t s = {0};
  s.n = n;
  posix_memalign((void **)&s.a, 16384, n * sizeof(double));
  posix_memalign((void **)&s.b, 16384, n * sizeof(double));
  posix_memalign((void **)&s.c, 16384, n * sizeof(double));
  for (size_t i = 0; i < n; i++) s.a[i] = 1.0, s.b[i] = 2.0, s.c[i] = 0.5;  // fault in

  const char *ops[] = {"read", "write", "copy", "triad"};
  const double mult[] = {1, 1, 2, 3};
  struct { int t; qos_class_t q; } cfg[] = {
      {1, QOS_CLASS_USER_INTERACTIVE}, {2, QOS_CLASS_USER_INTERACTIVE},
      {4, QOS_CLASS_USER_INTERACTIVE}, {6, QOS_CLASS_BACKGROUND},
      {8, QOS_CLASS_USER_INTERACTIVE}, {10, QOS_CLASS_USER_INTERACTIVE}};
  double *times = malloc(sizeof(double) * reps), *gbs = malloc(sizeof(double) * reps);
  for (size_t ci = 0; ci < sizeof(cfg) / sizeof(cfg[0]); ci++) {
    for (int op = 0; op < 4; op++) {
      s.op = op;
      run_parallel(cfg[ci].t, cfg[ci].q, warm, reps, kern, &s, times);
      for (int r = 0; r < reps; r++) gbs[r] = mult[op] * n * sizeof(double) / times[r] / 1e9;
      char extra[256];
      snprintf(extra, sizeof extra,
               "\"threads\":%d,\"qos\":\"%s\",\"array_mib\":%zu,\"bytes_per_rep\":%.0f",
               cfg[ci].t, qos_name(cfg[ci].q), mib, mult[op] * n * sizeof(double));
      emit("cpu_bandwidth", "cpu", ops[op], "f64", "GB/s", stats_of(gbs, reps), extra);
    }
  }
  return 0;
}
