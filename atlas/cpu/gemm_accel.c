// Accelerate GEMM throughput: cblas_sgemm (fp32; Accelerate dispatches to the SME/AMX unit on
// M4/M5) in multi- and single-threaded BLAS modes, and BNNSMatMul for fp16 / bf16 / fp32
// (BNNSMatMul is deprecated in favour of BNNSGraph but still works and needs no model file).
// Shapes come from stdin: "tag M N K" lines (see shapes.py). C[M,N] = A[M,K] * W[N,K]^T.
// Weights are cycled through enough copies to exceed ~256 MiB so memory-bound shapes are
// measured cache-cold (decode-realistic) instead of from L2/SLC.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include "common.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static size_t ncopies_for(size_t wbytes) {
  size_t target = (size_t)env_int("ATLAS_COLD_MIB", 256) << 20;
  size_t n = (target + wbytes - 1) / wbytes;
  return n < 1 ? 1 : (n > 64 ? 64 : n);
}

static void fill(void *p, size_t bytes) {
  uint16_t *q = (uint16_t *)p;  // small finite values for any 16/32-bit float type
  for (size_t i = 0; i < bytes / 2; i++) q[i] = 0x3c00 ^ (uint16_t)(i & 0x00ff);
}

// Times `fn(copy_index)` with inner-loop batching so each rep lasts >= min_rep_s.
typedef int (*call_fn)(size_t c, void *ctx);

static int time_reps(call_fn fn, void *ctx, size_t ncopies, double flops, double bytes,
                     double *gflops, double *gbs, int reps, int warm) {
  // Calibrate inner iterations.
  double min_rep = atlas_smoke() ? 0.02 : 0.1;
  size_t c = 0;
  for (int w = 0; w < warm; w++)
    if (fn(c++ % ncopies, ctx)) return -1;
  double t0 = now_s();
  if (fn(c++ % ncopies, ctx)) return -1;
  double one = now_s() - t0;
  int inner = one > 0 ? (int)(min_rep / one) + 1 : 1;
  if (inner > 100000) inner = 100000;
  for (int r = 0; r < reps; r++) {
    double s = now_s();
    for (int i = 0; i < inner; i++) fn(c++ % ncopies, ctx);
    double dt = (now_s() - s) / inner;
    gflops[r] = flops / dt / 1e9;
    gbs[r] = bytes / dt / 1e9;
  }
  return inner;
}

typedef struct {
  int M, N, K;
  void *A, *C;
  void **W;
  BNNSDataType dt;
  size_t esz;
} gctx_t;

static int call_sgemm(size_t c, void *p) {
  gctx_t *g = p;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, g->M, g->N, g->K, 1.0f, g->A, g->K,
              g->W[c], g->K, 0.0f, g->C, g->N);
  return 0;
}

static BNNSNDArrayDescriptor desc2d(void *data, BNNSDataType dt, size_t rows, size_t cols) {
  BNNSNDArrayDescriptor d;
  memset(&d, 0, sizeof d);
  d.layout = BNNSDataLayoutRowMajorMatrix;  // size[0]=cols, size[1]=rows
  d.size[0] = cols;
  d.size[1] = rows;
  d.data = data;
  d.data_type = dt;
  return d;
}

static int call_bnns(size_t c, void *p) {
  gctx_t *g = p;
  BNNSNDArrayDescriptor a = desc2d(g->A, g->dt, g->M, g->K);
  BNNSNDArrayDescriptor w = desc2d(g->W[c], g->dt, g->N, g->K);
  BNNSNDArrayDescriptor o = desc2d(g->C, g->dt, g->M, g->N);
  return BNNSMatMul(false, true, 1.0f, &a, &w, &o, NULL, NULL);
}

int main(void) {
  int reps = atlas_reps(), warm = atlas_warmup();
  char tag[128];
  int M, N, K;
  double *gf = malloc(sizeof(double) * 1000), *gb = malloc(sizeof(double) * 1000);
  while (scanf("%127s %d %d %d", tag, &M, &N, &K) == 4) {
    struct {
      const char *engine, *dtype, *mode;
      BNNSDataType dt;
      size_t esz;
      int bnns;
      enum BLAS_THREADING th;
    } cfg[] = {
        {"accelerate_blas", "f32", "multi_threaded", BNNSDataTypeFloat32, 4, 0, BLAS_THREADING_MULTI_THREADED},
        {"accelerate_blas", "f32", "single_threaded", BNNSDataTypeFloat32, 4, 0, BLAS_THREADING_SINGLE_THREADED},
        {"accelerate_bnns", "f32", "default", BNNSDataTypeFloat32, 4, 1, BLAS_THREADING_MULTI_THREADED},
        {"accelerate_bnns", "f16", "default", BNNSDataTypeFloat16, 2, 1, BLAS_THREADING_MULTI_THREADED},
        {"accelerate_bnns", "bf16", "default", BNNSDataTypeBFloat16, 2, 1, BLAS_THREADING_MULTI_THREADED},
    };
    for (size_t ci = 0; ci < sizeof cfg / sizeof cfg[0]; ci++) {
      // Skip the very slow single-threaded / BNNS paths for the huge lm_head prefill.
      double flops = 2.0 * M * N * K;
      if (flops > 5e11 && ci > 0 && atlas_smoke()) continue;
      BLASSetThreading(cfg[ci].th);
      gctx_t g = {M, N, K, NULL, NULL, NULL, cfg[ci].dt, cfg[ci].esz};
      size_t wbytes = (size_t)N * K * g.esz;
      size_t nc = ncopies_for(wbytes);
      g.A = malloc((size_t)M * K * g.esz);
      g.C = malloc((size_t)M * N * g.esz);
      g.W = malloc(sizeof(void *) * nc);
      fill(g.A, (size_t)M * K * g.esz);
      for (size_t i = 0; i < nc; i++) { g.W[i] = malloc(wbytes); fill(g.W[i], wbytes); }
      double bytes = (double)wbytes + (double)(M * K + M * N) * g.esz;
      int inner = time_reps(cfg[ci].bnns ? call_bnns : call_sgemm, &g, nc, flops, bytes, gf, gb,
                            reps, warm);
      char extra[512];
      if (inner < 0) {
        snprintf(extra, sizeof extra,
                 "\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"mode\":\"%s\",\"error\":\"BNNSMatMul returned nonzero (unsupported)\"",
                 tag, M, N, K, cfg[ci].mode);
        stats_t z = {0};
        emit("gemm", cfg[ci].engine, "matmul", cfg[ci].dtype, "GFLOP/s", z, extra);
      } else {
        stats_t bw = stats_of(gb, reps);
        snprintf(extra, sizeof extra,
                 "\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"mode\":\"%s\",\"weight_copies\":%zu,"
                 "\"inner_iters\":%d,\"eff_gbps_median\":%.4g,\"eff_gbps_max\":%.4g,\"bytes_per_call\":%.0f",
                 tag, M, N, K, cfg[ci].mode, nc, inner, bw.median, bw.max, bytes);
        emit("gemm", cfg[ci].engine, "matmul", cfg[ci].dtype, "GFLOP/s", stats_of(gf, reps), extra);
      }
      for (size_t i = 0; i < nc; i++) free(g.W[i]);
      free(g.W); free(g.A); free(g.C);
    }
  }
  return 0;
}
