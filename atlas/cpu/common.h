// Shared timing / statistics / JSON helpers for the atlas C benchmarks.
// Every benchmark: warmup, then >= ATLAS_REPS timed reps; report median/min/max/stddev.
// Records go to stdout as one JSON object per line; scripts/wrap.py adds metadata.
#pragma once
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_s(void) {
  return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9;
}

static inline int env_int(const char *name, int dflt) {
  const char *v = getenv(name);
  return (v && *v) ? atoi(v) : dflt;
}

// ATLAS_SMOKE=1 shrinks sizes/reps for validation-only runs.
static inline int atlas_smoke(void) { return env_int("ATLAS_SMOKE", 0); }
static inline int atlas_reps(void) {
  int r = env_int("ATLAS_REPS", atlas_smoke() ? 5 : 15);
  return r < 5 ? 5 : r;
}
static inline int atlas_warmup(void) { return env_int("ATLAS_WARMUP", atlas_smoke() ? 1 : 3); }

static int cmp_dbl(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

typedef struct {
  double median, min, max, mean, stddev;
  int n;
} stats_t;

// Stats over per-rep *throughput* values (higher is better).
static stats_t stats_of(double *v, int n) {
  stats_t s = {0};
  double *t = malloc(sizeof(double) * n);
  memcpy(t, v, sizeof(double) * n);
  qsort(t, n, sizeof(double), cmp_dbl);
  s.n = n;
  s.min = t[0];
  s.max = t[n - 1];
  s.median = (n % 2) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
  double sum = 0, sq = 0;
  for (int i = 0; i < n; i++) sum += t[i];
  s.mean = sum / n;
  for (int i = 0; i < n; i++) sq += (t[i] - s.mean) * (t[i] - s.mean);
  s.stddev = n > 1 ? sqrt(sq / (n - 1)) : 0;
  free(t);
  return s;
}

// Emits a record. `extra` is a JSON fragment (without braces) or "".
static void emit(const char *bench, const char *engine, const char *op, const char *dtype,
                 const char *unit, stats_t s, const char *extra) {
  printf("{\"bench\":\"%s\",\"engine\":\"%s\",\"op\":\"%s\",\"dtype\":\"%s\",\"unit\":\"%s\","
         "\"median\":%.6g,\"min\":%.6g,\"max\":%.6g,\"mean\":%.6g,\"stddev\":%.6g,\"reps\":%d%s%s}\n",
         bench, engine, op, dtype, unit, s.median, s.min, s.max, s.mean, s.stddev, s.n,
         (extra && *extra) ? "," : "", extra ? extra : "");
  fflush(stdout);
}

// ---------- simple fork/join thread pool with a spin barrier ----------
// (macOS has no pthread_barrier_t, so emulate one.)
typedef struct {
  pthread_mutex_t m;
  pthread_cond_t c;
  int count, need, gen;
} barrier_t;

static void barrier_init(barrier_t *b, int n) {
  pthread_mutex_init(&b->m, NULL);
  pthread_cond_init(&b->c, NULL);
  b->count = 0;
  b->need = n;
  b->gen = 0;
}
static void barrier_wait(barrier_t *b) {
  pthread_mutex_lock(&b->m);
  int g = b->gen;
  if (++b->count == b->need) {
    b->gen++;
    b->count = 0;
    pthread_cond_broadcast(&b->c);
  } else {
    while (g == b->gen) pthread_cond_wait(&b->c, &b->m);
  }
  pthread_mutex_unlock(&b->m);
}

// ---------- fork/join timed runner ----------
// Runs kernel(tid, nthreads, arg) on `nthreads` threads, `warmup` untimed + `reps` timed
// iterations, all threads released together by a barrier. times[r] = wall seconds of rep r
// (slowest thread). qos: QOS_CLASS_USER_INTERACTIVE steers to P-cores, QOS_CLASS_BACKGROUND
// to E-cores (macOS has no hard affinity; this is a scheduler hint only).
#include <pthread/qos.h>

typedef void (*kernel_fn)(int tid, int nthreads, void *arg);

typedef struct {
  int tid, nthreads, warmup, reps;
  qos_class_t qos;
  kernel_fn fn;
  void *arg;
  barrier_t *bar;
  double *times;
} runner_ctx_t;

static void *runner_thread(void *p) {
  runner_ctx_t *c = (runner_ctx_t *)p;
  pthread_set_qos_class_self_np(c->qos, 0);
  for (int r = 0; r < c->warmup + c->reps; r++) {
    barrier_wait(c->bar);
    double t0 = now_s();
    c->fn(c->tid, c->nthreads, c->arg);
    barrier_wait(c->bar);
    double t1 = now_s();
    if (c->tid == 0 && r >= c->warmup) c->times[r - c->warmup] = t1 - t0;
  }
  return NULL;
}

static void run_parallel(int nthreads, qos_class_t qos, int warmup, int reps, kernel_fn fn,
                         void *arg, double *times) {
  barrier_t bar;
  barrier_init(&bar, nthreads);
  pthread_t th[64];
  runner_ctx_t ctx[64];
  for (int i = 0; i < nthreads; i++) {
    ctx[i] = (runner_ctx_t){i, nthreads, warmup, reps, qos, fn, arg, &bar, times};
    if (i) pthread_create(&th[i], NULL, runner_thread, &ctx[i]);
  }
  runner_thread(&ctx[0]);
  for (int i = 1; i < nthreads; i++) pthread_join(th[i], NULL);
}

static const char *qos_name(qos_class_t q) {
  return q == QOS_CLASS_BACKGROUND ? "background(E-cores)" : "user_interactive(P-pref)";
}
