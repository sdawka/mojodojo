// CPU compute ceilings: register-only microkernels (no memory traffic) for
//   NEON:  FMLA f32 / f16, FMLAL (f16->f32), BFDOT / BFMMLA (bf16->f32), SDOT / SMMLA (i8->i32)
//   SME:   FMOPA outer products into ZA (f32, f16->f32, bf16->f32, f16->f16, i8->i32)
// Each kernel issues many independent ops so throughput, not latency, is measured.
// Thread configs probe whether units are per-core (NEON) or shared per cluster (SME).
#include "common.h"
#include <arm_neon.h>
#include <arm_sme.h>

#define NACC 24  // independent accumulators (32 vector regs total)

static volatile float g_sink_f;
static volatile int32_t g_sink_i;

typedef struct {
  long iters;
  int which;
} pargs_t;

// ---------------- NEON kernels: return ops per outer iteration ----------------
#define NEON_BODY(T, INIT, OP, RED)                      \
  T acc[NACC];                                           \
  for (int k = 0; k < NACC; k++) acc[k] = INIT;          \
  for (long i = 0; i < iters; i++) {                     \
    _Pragma("clang loop unroll(full)") for (int k = 0; k < NACC; k++) acc[k] = OP; \
  }                                                      \
  RED

static void neon_f32(long iters) {
  float32x4_t x = vdupq_n_f32(1.0001f), y = vdupq_n_f32(0.9999f);
  __asm__ volatile("" : "+w"(x), "+w"(y));
  NEON_BODY(float32x4_t, vdupq_n_f32(0), vfmaq_f32(acc[k], x, y),
            float s = 0; for (int k = 0; k < NACC; k++) s += vaddvq_f32(acc[k]); g_sink_f = s;)
}
static void neon_f16(long iters) {
  float16x8_t x = vdupq_n_f16(1.0f), y = vdupq_n_f16(0.001f);
  __asm__ volatile("" : "+w"(x), "+w"(y));
  NEON_BODY(float16x8_t, vdupq_n_f16(0), vfmaq_f16(acc[k], x, y),
            float s = 0; for (int k = 0; k < NACC; k++) s += (float)vgetq_lane_f16(acc[k], 0); g_sink_f = s;)
}
static void neon_fmlal(long iters) {  // f16 x f16 -> f32 widening (low half): 4 MACs
  float16x8_t x = vdupq_n_f16(1.0f), y = vdupq_n_f16(0.001f);
  __asm__ volatile("" : "+w"(x), "+w"(y));
  NEON_BODY(float32x4_t, vdupq_n_f32(0), vfmlalq_low_f16(acc[k], x, y),
            float s = 0; for (int k = 0; k < NACC; k++) s += vaddvq_f32(acc[k]); g_sink_f = s;)
}
static void neon_bfdot(long iters) {  // 4 lanes x 2-way = 8 MACs
  bfloat16x8_t x = vreinterpretq_bf16_u16(vdupq_n_u16(0x3f80)), y = x;
  __asm__ volatile("" : "+w"(x), "+w"(y));
  NEON_BODY(float32x4_t, vdupq_n_f32(0), vbfdotq_f32(acc[k], x, y),
            float s = 0; for (int k = 0; k < NACC; k++) s += vaddvq_f32(acc[k]); g_sink_f = s;)
}
static void neon_bfmmla(long iters) {  // 2x4 . 4x2 = 16 MACs
  bfloat16x8_t x = vreinterpretq_bf16_u16(vdupq_n_u16(0x3f80)), y = x;
  __asm__ volatile("" : "+w"(x), "+w"(y));
  NEON_BODY(float32x4_t, vdupq_n_f32(0), vbfmmlaq_f32(acc[k], x, y),
            float s = 0; for (int k = 0; k < NACC; k++) s += vaddvq_f32(acc[k]); g_sink_f = s;)
}
static void neon_sdot(long iters) {  // 4 lanes x 4-way = 16 MACs
  int8x16_t x = vdupq_n_s8(1), y = vdupq_n_s8(1);
  __asm__ volatile("" : "+w"(x), "+w"(y));
  NEON_BODY(int32x4_t, vdupq_n_s32(0), vdotq_s32(acc[k], x, y),
            int32_t s = 0; for (int k = 0; k < NACC; k++) s += vaddvq_s32(acc[k]); g_sink_i = s;)
}
static void neon_smmla(long iters) {  // 2x8 . 8x2 = 32 MACs
  int8x16_t x = vdupq_n_s8(1), y = vdupq_n_s8(1);
  __asm__ volatile("" : "+w"(x), "+w"(y));
  NEON_BODY(int32x4_t, vdupq_n_s32(0), vmmlaq_s32(acc[k], x, y),
            int32_t s = 0; for (int k = 0; k < NACC; k++) s += vaddvq_s32(acc[k]); g_sink_i = s;)
}

// ---------------- SME kernels (streaming mode, ZA outer products) ----------------
// SVL = 512 bits on M4/M5 (hw.optional.arm.sme_max_svl_b = 64 bytes), so a za32 tile is 16x16.
static float g_za_out[64];

__arm_locally_streaming __arm_new("za") static void sme_f32(long iters) {
  svbool_t p = svptrue_b32();
  svfloat32_t a = svdup_f32(1.0f), b = svdup_f32(1e-7f);
  for (long i = 0; i < iters; i++) {
    svmopa_za32_f32_m(0, p, p, a, b); svmopa_za32_f32_m(1, p, p, a, b);
    svmopa_za32_f32_m(2, p, p, a, b); svmopa_za32_f32_m(3, p, p, a, b);
  }
  svst1_hor_za32(0, 0, p, g_za_out);
}
__arm_locally_streaming __arm_new("za") static void sme_f16f32(long iters) {
  svbool_t p = svptrue_b16();
  svfloat16_t a = svdup_f16(1.0f), b = svdup_f16(1e-3f);
  for (long i = 0; i < iters; i++) {
    svmopa_za32_f16_m(0, p, p, a, b); svmopa_za32_f16_m(1, p, p, a, b);
    svmopa_za32_f16_m(2, p, p, a, b); svmopa_za32_f16_m(3, p, p, a, b);
  }
  svst1_hor_za32(0, 0, svptrue_b32(), g_za_out);
}
__arm_locally_streaming __arm_new("za") static void sme_bf16f32(long iters) {
  svbool_t p = svptrue_b16();
  svbfloat16_t a = svreinterpret_bf16_u16(svdup_u16(0x3f80)), b = a;
  for (long i = 0; i < iters; i++) {
    svmopa_za32_bf16_m(0, p, p, a, b); svmopa_za32_bf16_m(1, p, p, a, b);
    svmopa_za32_bf16_m(2, p, p, a, b); svmopa_za32_bf16_m(3, p, p, a, b);
  }
  svst1_hor_za32(0, 0, svptrue_b32(), g_za_out);
}
__arm_locally_streaming __arm_new("za") static void sme_f16f16(long iters) {  // FEAT_SME_F16F16
  svbool_t p = svptrue_b16();
  svfloat16_t a = svdup_f16(1.0f), b = svdup_f16(1e-3f);
  for (long i = 0; i < iters; i++) {
    svmopa_za16_f16_m(0, p, p, a, b); svmopa_za16_f16_m(1, p, p, a, b);
    svmopa_za16_f16_m(0, p, p, a, b); svmopa_za16_f16_m(1, p, p, a, b);
  }
  svst1_hor_za16(0, 0, p, g_za_out);
}
__arm_locally_streaming __arm_new("za") static void sme_i8i32(long iters) {
  svbool_t p = svptrue_b8();
  svint8_t a = svdup_s8(1), b = svdup_s8(1);
  for (long i = 0; i < iters; i++) {
    svmopa_za32_s8_m(0, p, p, a, b); svmopa_za32_s8_m(1, p, p, a, b);
    svmopa_za32_s8_m(2, p, p, a, b); svmopa_za32_s8_m(3, p, p, a, b);
  }
  svst1_hor_za32(0, 0, svptrue_b32(), g_za_out);
}

typedef struct {
  const char *name, *engine, *dtype;
  void (*fn)(long);
  double ops_per_iter;  // FLOPs (or int OPs), 1 MAC = 2 ops
} kdesc_t;

// SVL-dependent op counts computed at runtime.
static kdesc_t K[16];
static int NK = 0;

static void kern(int tid, int nt, void *p) {
  (void)tid; (void)nt;
  pargs_t *a = (pargs_t *)p;
  K[a->which].fn(a->iters);
}

int main(void) {
  int svl_b = 64;  // bytes; verified against sysctl by scripts/inventory.py
  int t32 = svl_b / 4, t16 = svl_b / 2;  // za32 tile dim, za16 tile dim
  K[NK++] = (kdesc_t){"fmla", "cpu_neon", "f32", neon_f32, NACC * 4 * 2.0};
  K[NK++] = (kdesc_t){"fmla", "cpu_neon", "f16", neon_f16, NACC * 8 * 2.0};
  K[NK++] = (kdesc_t){"fmlal", "cpu_neon", "f16->f32", neon_fmlal, NACC * 4 * 2.0};
  K[NK++] = (kdesc_t){"bfdot", "cpu_neon", "bf16->f32", neon_bfdot, NACC * 8 * 2.0};
  K[NK++] = (kdesc_t){"bfmmla", "cpu_neon", "bf16->f32", neon_bfmmla, NACC * 16 * 2.0};
  K[NK++] = (kdesc_t){"sdot", "cpu_neon", "i8->i32", neon_sdot, NACC * 16 * 2.0};
  K[NK++] = (kdesc_t){"smmla", "cpu_neon", "i8->i32", neon_smmla, NACC * 32 * 2.0};
  K[NK++] = (kdesc_t){"fmopa", "cpu_sme", "f32", sme_f32, 4.0 * t32 * t32 * 2};
  K[NK++] = (kdesc_t){"fmopa", "cpu_sme", "f16->f32", sme_f16f32, 4.0 * t32 * t32 * 2 * 2};
  K[NK++] = (kdesc_t){"bfmopa", "cpu_sme", "bf16->f32", sme_bf16f32, 4.0 * t32 * t32 * 2 * 2};
  K[NK++] = (kdesc_t){"fmopa", "cpu_sme", "f16", sme_f16f16, 4.0 * t16 * t16 * 2};
  K[NK++] = (kdesc_t){"smopa", "cpu_sme", "i8->i32", sme_i8i32, 4.0 * t32 * t32 * 4 * 2};

  int reps = atlas_reps(), warm = atlas_warmup();
  long iters = atlas_smoke() ? 2000000 : 20000000;
  struct { int t; qos_class_t q; } cfg[] = {
      {1, QOS_CLASS_USER_INTERACTIVE}, {2, QOS_CLASS_USER_INTERACTIVE},
      {4, QOS_CLASS_USER_INTERACTIVE}, {6, QOS_CLASS_BACKGROUND},
      {10, QOS_CLASS_USER_INTERACTIVE}};
  double *times = malloc(sizeof(double) * reps), *g = malloc(sizeof(double) * reps);
  for (int k = 0; k < NK; k++) {
    // SME units are fewer/heavier; scale iterations so each rep is ~0.1-1 s.
    long it = K[k].engine[4] == 's' ? iters / 8 : iters;
    for (size_t c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
      pargs_t a = {it, k};
      run_parallel(cfg[c].t, cfg[c].q, warm, reps, kern, &a, times);
      for (int r = 0; r < reps; r++) g[r] = K[k].ops_per_iter * it * cfg[c].t / times[r] / 1e9;
      char extra[256];
      snprintf(extra, sizeof extra, "\"threads\":%d,\"qos\":\"%s\",\"instr\":\"%s\",\"iters\":%ld",
               cfg[c].t, qos_name(cfg[c].q), K[k].name, it);
      emit("cpu_peak", K[k].engine, "peak_fma", K[k].dtype,
           strchr(K[k].dtype, 'i') ? "GOP/s" : "GFLOP/s", stats_of(g, reps), extra);
    }
  }
  return 0;
}
