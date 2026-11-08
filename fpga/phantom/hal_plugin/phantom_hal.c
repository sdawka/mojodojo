// Phantom NPU driver plugin for the Modular HAL (`SDK/include/HAL/M_driver_*`,
// interface 0.1.0, modular@2841b316).
//
// One device ("phantom-npu", arch "phantom-n<N>"), backed by libphantom.dylib
// (the cycle-accurate tile simulator compiled from Mojo). Memory lives in the
// simulator's device memory; copies are synchronous; queues are in-order
// streams. The one executable function is "phantom_matmul_i8", taking six
// arguments (a, b, c device addresses as uint64; m, k, n as int64), loaded
// from any static bundle with file_type "phantom" (or "object").
//
// Built against the HAL headers in third_party (MAX License); this file is
// our own code and includes no Modular source.
#include "HAL/M_driver_api.h"
#include "phantom.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT M_DRIVER_API_EXPORT
#define OK M_DRIVER_STATUS_SUCCESS

struct M_driver_driver { int64_t sim; int64_t tile_n; char last_error[256]; };
struct M_driver_device { struct M_driver_driver *drv; };
struct M_driver_context { struct M_driver_driver *drv; };
struct M_driver_memory { uint64_t addr; uint64_t size; void *host; int pinned; };
struct M_driver_queue { struct M_driver_context *ctx; int64_t last_cycles; int64_t launches; };
struct M_driver_event { int dummy; };
struct M_driver_runtime_bundle { int dummy; };
struct M_driver_function { int kind; };

static struct M_driver_driver *g_drv; // single driver instance per process
static struct M_driver_device g_dev;
static int g_refs; // one simulator per process, shared by every M_driver_create

static int logon(void) { static int v = -1; if (v < 0) v = getenv("PHANTOM_HAL_LOG") != NULL; return v; }
#define HLOG(...) do { if (logon()) { fprintf(stderr, "[phantom-hal] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

static M_driver_status_t sim_err(const char *what) {
  if (g_drv) {
    char buf[200];
    phantom_last_error(g_drv->sim, buf, sizeof buf);
    snprintf(g_drv->last_error, sizeof g_drv->last_error, "%s: %s", what, buf);
  }
  return M_DRIVER_STATUS_CUSTOM_CODES_BASE;
}

// ------------------------------------------------------------------ driver
EXPORT M_driver_status_t M_driver_status_message(M_driver_driver_t driver, M_driver_status_t status,
                                                 char *buf, int64_t n) {
  if (!buf || n <= 0) return M_DRIVER_STATUS_INVALID_ARG;
  const char *msg = "phantom: error";
  switch (status) {
  case M_DRIVER_STATUS_SUCCESS: msg = "success"; break;
  case M_DRIVER_STATUS_UNIMPLEMENTED: msg = "phantom: not implemented by the phantom NPU"; break;
  case M_DRIVER_STATUS_UNSUPPORTED: msg = "phantom: unsupported on the phantom NPU"; break;
  case M_DRIVER_STATUS_BAD_FUNCTION: msg = "phantom: unknown function (only phantom_matmul_i8)"; break;
  case M_DRIVER_STATUS_INVALID_ARG: msg = "phantom: invalid argument"; break;
  case M_DRIVER_STATUS_BAD_DEVICE: msg = "phantom: bad device id"; break;
  case M_DRIVER_STATUS_CUSTOM_CODES_BASE: msg = driver ? driver->last_error : (g_drv ? g_drv->last_error : msg); break;
  default: break;
  }
  snprintf(buf, (size_t)n, "%s", msg);
  return OK;
}

EXPORT M_driver_status_t M_driver_create(M_driver_version *version, M_driver_driver_t *driver) {
  if (!version || !driver) return M_DRIVER_STATUS_INVALID_ARG;
  HLOG("M_driver_create(host interface %u.%u.%u)", version->major, version->minor, version->patch);
  if (version->major != M_DRIVER_INTERFACE_VERSION_MAJOR) return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;
  if (version->minor > M_DRIVER_INTERFACE_VERSION_MINOR) {
    version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
    version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
  }
  if (!g_drv) {
    const char *t = getenv("PHANTOM_TILE_N");
    int64_t n = t ? atoll(t) : 8;
    int64_t sim = phantom_open(n);
    if (sim <= 0) return M_DRIVER_STATUS_NO_DEVICES;
    g_drv = calloc(1, sizeof *g_drv);
    g_drv->sim = sim;
    g_drv->tile_n = n;
    g_dev.drv = g_drv;
    HLOG("opened simulator: %lld x %lld int8 tile", (long long)n, (long long)n);
  }
  g_refs++;
  *driver = g_drv;
  return OK;
}

// The simulator is a process-wide singleton, like a physical device. The
// NeoDriver client may create and destroy driver handles several times in one
// process (Host() and RawDriver.load() each create one, and value copies may
// destroy more than once), so destroy only drops a reference and never frees:
// a stale handle keeps pointing at valid state.
EXPORT M_driver_status_t M_driver_destroy(M_driver_driver_t driver) {
  if (driver && driver == g_drv) {
    if (g_refs > 0) g_refs--;
    HLOG("M_driver_destroy: refs=%d, %lld launches, %lld simulated cycles so far", g_refs,
         (long long)phantom_launches(g_drv->sim), (long long)phantom_total_cycles(g_drv->sim));
  }
  return OK;
}

EXPORT M_driver_status_t M_driver_property(M_driver_driver_t h, const char *name, void *value) {
  if (!name || !value) return M_DRIVER_STATUS_INVALID_ARG;
  if (!strcmp(name, M_DRIVER_PROPERTY_API)) { *(const char **)value = "phantom"; return OK; }
  if (!strcmp(name, M_DRIVER_PROPERTY_VERSION)) {
    M_driver_version *v = value;
    v->major = M_DRIVER_INTERFACE_VERSION_MAJOR; v->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
    v->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
    return OK;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

// ------------------------------------------------------------------ device
EXPORT M_driver_status_t M_driver_device_count(M_driver_driver_t d, int64_t *count) {
  if (!count) return M_DRIVER_STATUS_INVALID_ARG;
  *count = 1;
  return OK;
}
EXPORT M_driver_status_t M_driver_device_get(M_driver_driver_t d, int64_t id, M_driver_device_t *dev) {
  if (!dev) return M_DRIVER_STATUS_INVALID_ARG;
  if (id != 0) return M_DRIVER_STATUS_BAD_DEVICE;
  *dev = &g_dev;
  return OK;
}
EXPORT M_driver_status_t M_driver_device_attribute(M_driver_device_t dev, int32_t attr, int32_t *value) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}
EXPORT M_driver_status_t M_driver_device_property(M_driver_device_t dev, const char *name, void *value) {
  if (!dev || !name || !value) return M_DRIVER_STATUS_INVALID_ARG;
  if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_NAME)) {
    snprintf(value, M_DRIVER_DEVICE_PROPERTY_MAX_LEN, "phantom-npu (simulated %lldx%lld int8 systolic tile)",
             (long long)dev->drv->tile_n, (long long)dev->drv->tile_n);
    return OK;
  }
  if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_ARCH)) {
    snprintf(value, M_DRIVER_DEVICE_PROPERTY_MAX_LEN, "phantom-n%lld", (long long)dev->drv->tile_n);
    return OK;
  }
  if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED)) { *(int64_t *)value = 0; return OK; }
  if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE)) { *(uint64_t *)value = 1ull << 31; return OK; }
  if (!strcmp(name, "phantom_tile_n")) { *(int64_t *)value = dev->drv->tile_n; return OK; }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}
EXPORT M_driver_status_t M_driver_device_can_access_peer(M_driver_device_t d, M_driver_device_t p, bool *out) {
  if (!out) return M_DRIVER_STATUS_INVALID_ARG;
  *out = false;
  return OK;
}

// ------------------------------------------------------------------ context
EXPORT M_driver_status_t M_driver_context_create(M_driver_device_t device, M_driver_context_t *ctx) {
  if (!device || !ctx) return M_DRIVER_STATUS_INVALID_ARG;
  struct M_driver_context *c = calloc(1, sizeof *c);
  c->drv = device->drv;
  *ctx = c;
  return OK;
}
EXPORT M_driver_status_t M_driver_context_destroy(M_driver_context_t ctx) { free(ctx); return OK; }
EXPORT M_driver_status_t M_driver_context_set_current(M_driver_context_t ctx) { return OK; }
EXPORT M_driver_status_t M_driver_context_get_current_driver_context(M_driver_context_t c, M_driver_native_handle_t *h) {
  if (!h) return M_DRIVER_STATUS_INVALID_ARG;
  *h = (M_driver_native_handle_t)c->drv->sim;
  return OK;
}
EXPORT M_driver_status_t M_driver_context_set_current_driver_context(M_driver_context_t c, M_driver_native_handle_t h) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}
EXPORT M_driver_status_t M_driver_context_memory_info(M_driver_context_t c, uint64_t *free_b, uint64_t *total) {
  if (!free_b || !total) return M_DRIVER_STATUS_INVALID_ARG;
  *free_b = 1ull << 31; *total = 1ull << 31;
  return OK;
}
EXPORT M_driver_status_t M_driver_context_enable_peer_access(M_driver_context_t c, M_driver_context_t p) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}

// ------------------------------------------------------------------ memory
EXPORT M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t ctx, uint64_t size, M_driver_memory_t *mem) {
  if (!ctx || !mem || size == 0) return M_DRIVER_STATUS_INVALID_ARG;
  uint64_t a = phantom_alloc(ctx->drv->sim, (int64_t)size);
  if (!a) return sim_err("alloc");
  struct M_driver_memory *m = calloc(1, sizeof *m);
  m->addr = a; m->size = size;
  *mem = m;
  HLOG("alloc %llu bytes -> device address 0x%llx", (unsigned long long)size, (unsigned long long)a);
  return OK;
}
EXPORT M_driver_status_t M_driver_memory_free_sync(M_driver_context_t ctx, M_driver_memory_t mem) {
  if (!ctx || !mem) return M_DRIVER_STATUS_INVALID_ARG;
  if (!mem->pinned && phantom_free(ctx->drv->sim, mem->addr) != 0) return sim_err("free");
  free(mem->host);
  free(mem);
  return OK;
}
EXPORT M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t q, uint64_t size, M_driver_memory_t *mem) {
  return M_driver_memory_alloc_sync(q ? q->ctx : NULL, size, mem);
}
EXPORT M_driver_status_t M_driver_memory_free_async(M_driver_queue_t q, M_driver_memory_t mem) {
  return M_driver_memory_free_sync(q ? q->ctx : NULL, mem);
}
EXPORT M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t ctx, uint64_t size, M_driver_memory_t *mem) {
  if (!mem || size == 0) return M_DRIVER_STATUS_INVALID_ARG;
  struct M_driver_memory *m = calloc(1, sizeof *m);
  m->host = calloc(1, size); m->size = size; m->pinned = 1; m->addr = (uint64_t)(uintptr_t)m->host;
  *mem = m;
  return OK;
}
EXPORT M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t ctx, M_driver_memory_t mem) {
  return M_driver_memory_free_sync(ctx, mem);
}
EXPORT M_driver_status_t M_driver_memory_wrap(M_driver_context_t c, uint64_t address, uint64_t size, bool owning,
                                              M_driver_memory_t *mem) {
  return M_DRIVER_STATUS_UNSUPPORTED; // device memory is not host-addressable
}
EXPORT M_driver_status_t M_driver_memory_unwrap(M_driver_context_t c, M_driver_memory_t mem, uint64_t *address) {
  if (!mem || !address) return M_DRIVER_STATUS_INVALID_ARG;
  *address = mem->addr;
  return OK;
}
EXPORT M_driver_status_t M_driver_memory_property(M_driver_memory_t mem, const char *name, void *value) {
  if (!mem || !name || !value) return M_DRIVER_STATUS_INVALID_ARG;
  if (!strcmp(name, M_DRIVER_MEMORY_PROPERTY_ADDRESS)) { *(uint64_t *)value = mem->addr; return OK; }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

// ------------------------------------------------------------------ copies
static M_driver_status_t h2d(M_driver_context_t ctx, M_driver_memory_view_t dst, const void *src) {
  if (!ctx || !dst || !dst->memory || !src) return M_DRIVER_STATUS_INVALID_ARG;
  if (dst->memory->pinned) { memcpy((char *)dst->memory->host + dst->offset, src, dst->size); return OK; }
  if (phantom_memcpy_h2d(ctx->drv->sim, dst->memory->addr + dst->offset, src, (int64_t)dst->size)) return sim_err("h2d");
  HLOG("copy h2d %llu bytes", (unsigned long long)dst->size);
  return OK;
}
static M_driver_status_t d2h(M_driver_context_t ctx, void *dst, M_driver_memory_view_t src) {
  if (!ctx || !src || !src->memory || !dst) return M_DRIVER_STATUS_INVALID_ARG;
  if (src->memory->pinned) { memcpy(dst, (char *)src->memory->host + src->offset, src->size); return OK; }
  if (phantom_memcpy_d2h(ctx->drv->sim, dst, src->memory->addr + src->offset, (int64_t)src->size)) return sim_err("d2h");
  HLOG("copy d2h %llu bytes", (unsigned long long)src->size);
  return OK;
}
static M_driver_status_t d2d(M_driver_context_t ctx, M_driver_memory_view_t dst, M_driver_memory_view_t src) {
  if (!dst || !src || dst->size != src->size) return M_DRIVER_STATUS_INVALID_ARG;
  void *tmp = malloc(src->size ? src->size : 1);
  M_driver_status_t st = d2h(ctx, tmp, src);
  if (st == OK) st = h2d(ctx, dst, tmp);
  free(tmp);
  return st;
}
EXPORT M_driver_status_t M_driver_copy_to_device_sync(M_driver_context_t c, M_driver_memory_view_t d, const void *s) { return h2d(c, d, s); }
EXPORT M_driver_status_t M_driver_copy_from_device_sync(M_driver_context_t c, void *d, M_driver_memory_view_t s) { return d2h(c, d, s); }
EXPORT M_driver_status_t M_driver_copy_intra_device_sync(M_driver_context_t c, M_driver_memory_view_t d, M_driver_memory_view_t s) { return d2d(c, d, s); }
EXPORT M_driver_status_t M_driver_copy_inter_device_sync(M_driver_context_t c, M_driver_memory_view_t d, M_driver_memory_view_t s,
                                                         M_driver_context_t sc) { return M_DRIVER_STATUS_UNSUPPORTED; }
EXPORT M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t q, M_driver_memory_view_t d, const void *s) { return h2d(q ? q->ctx : NULL, d, s); }
EXPORT M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t q, void *d, M_driver_memory_view_t s) { return d2h(q ? q->ctx : NULL, d, s); }
EXPORT M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t q, M_driver_memory_view_t d, M_driver_memory_view_t s) { return d2d(q ? q->ctx : NULL, d, s); }
EXPORT M_driver_status_t M_driver_queue_copy_inter_device(M_driver_queue_t q, M_driver_memory_view_t d, M_driver_memory_view_t s,
                                                          M_driver_context_t sc) { return M_DRIVER_STATUS_UNSUPPORTED; }

static M_driver_status_t fill_bytes(M_driver_context_t ctx, M_driver_memory_view_t dst, uint64_t value, uint64_t vsize) {
  if (!dst || vsize == 0 || vsize > 8 || dst->size % vsize) return M_DRIVER_STATUS_INVALID_ARG;
  char *tmp = malloc(dst->size ? dst->size : 1);
  for (uint64_t i = 0; i < dst->size; i += vsize) memcpy(tmp + i, &value, vsize);
  M_driver_status_t st = h2d(ctx, dst, tmp);
  free(tmp);
  return st;
}
EXPORT M_driver_status_t M_driver_set_memory_sync(M_driver_context_t c, M_driver_memory_view_t d, uint8_t v) { return fill_bytes(c, d, v, 1); }
EXPORT M_driver_status_t M_driver_fill_sync(M_driver_context_t c, M_driver_memory_view_t d, uint64_t v, uint64_t n) { return fill_bytes(c, d, v, n); }
EXPORT M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t q, M_driver_memory_view_t d, uint8_t v) { return fill_bytes(q ? q->ctx : NULL, d, v, 1); }
EXPORT M_driver_status_t M_driver_queue_fill(M_driver_queue_t q, M_driver_memory_view_t d, uint64_t v, uint64_t n) { return fill_bytes(q ? q->ctx : NULL, d, v, n); }

// ------------------------------------------------------------------ queues & events
EXPORT M_driver_status_t M_driver_queue_create(M_driver_context_t ctx, M_driver_queue_t *q) {
  if (!ctx || !q) return M_DRIVER_STATUS_INVALID_ARG;
  struct M_driver_queue *x = calloc(1, sizeof *x);
  x->ctx = ctx;
  *q = x;
  return OK;
}
EXPORT M_driver_status_t M_driver_queue_destroy(M_driver_context_t c, M_driver_queue_t q) { free(q); return OK; }
EXPORT M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t q) { return q ? OK : M_DRIVER_STATUS_BAD_QUEUE; }
EXPORT M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t q, bool *s) { if (!s) return M_DRIVER_STATUS_INVALID_ARG; *s = true; return OK; }
EXPORT M_driver_status_t M_driver_queue_property(M_driver_queue_t q, const char *name, void *value) {
  if (!q || !name || !value) return M_DRIVER_STATUS_INVALID_ARG;
  if (!strcmp(name, "phantom_last_cycles")) { *(int64_t *)value = q->last_cycles; return OK; }
  if (!strcmp(name, "phantom_launches")) { *(int64_t *)value = q->launches; return OK; }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}
EXPORT M_driver_status_t M_driver_queue_launch_host_func(M_driver_queue_t q, void (*fn)(void *), void *ud) {
  if (!fn) return M_DRIVER_STATUS_INVALID_ARG;
  fn(ud); // in-order and synchronous: everything before it has completed
  return OK;
}
EXPORT M_driver_status_t M_driver_queue_wait_value64(M_driver_queue_t q, uint64_t a, uint64_t v) { return M_DRIVER_STATUS_UNSUPPORTED; }
EXPORT M_driver_status_t M_driver_queue_record_event(M_driver_queue_t q, M_driver_event_t e) { return OK; }
EXPORT M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t q, M_driver_event_t *e, uint32_t n) { return OK; }
EXPORT M_driver_status_t M_driver_event_create(M_driver_context_t c, M_driver_event_flags_t f, M_driver_event_t *e) {
  if (!e) return M_DRIVER_STATUS_INVALID_ARG;
  *e = calloc(1, sizeof(struct M_driver_event));
  return OK;
}
EXPORT M_driver_status_t M_driver_event_destroy(M_driver_context_t c, M_driver_event_t e) { free(e); return OK; }
EXPORT M_driver_status_t M_driver_event_synchronize(M_driver_context_t c, M_driver_event_t e) { return OK; }
EXPORT M_driver_status_t M_driver_is_event_ready(M_driver_context_t c, M_driver_event_t e, bool *r) { if (!r) return M_DRIVER_STATUS_INVALID_ARG; *r = true; return OK; }

// ------------------------------------------------------------------ bundles & functions
static struct M_driver_runtime_bundle g_bundle;
static struct M_driver_function g_matmul = {1};

EXPORT M_driver_status_t M_driver_bundle_load(M_driver_context_t c, M_driver_static_bundle_t b,
                                              M_driver_bundle_compilation_options_t o, M_driver_runtime_bundle_t *rb) {
  if (!b || !rb) return M_DRIVER_STATUS_INVALID_ARG;
  // The tile is fixed-function: a bundle carries no code, only a manifest.
  HLOG("bundle_load file_type=%.*s, %llu bytes", (int)b->file_type_len, b->file_type ? b->file_type : "",
       (unsigned long long)b->mapped_data.size);
  *rb = &g_bundle;
  return OK;
}
EXPORT M_driver_status_t M_driver_bundle_unload(M_driver_context_t c, M_driver_runtime_bundle_t b) { return OK; }
EXPORT M_driver_status_t M_driver_function_load(M_driver_context_t c, M_driver_runtime_bundle_t b, const char *name,
                                                uint64_t len, M_driver_function_t *f) {
  if (!name || !f) return M_DRIVER_STATUS_INVALID_ARG;
  HLOG("function_load '%.*s'", (int)len, name);
  if (len == strlen("phantom_matmul_i8") && !memcmp(name, "phantom_matmul_i8", len)) { *f = &g_matmul; return OK; }
  return M_DRIVER_STATUS_BAD_FUNCTION;
}
EXPORT M_driver_status_t M_driver_function_unload(M_driver_context_t c, M_driver_function_t f) { return OK; }
EXPORT M_driver_status_t M_driver_function_property(M_driver_context_t c, M_driver_function_t f, const char *n, void *v) {
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}
EXPORT M_driver_status_t M_driver_function_set_attribute(M_driver_context_t c, M_driver_function_t f, int32_t a, int32_t v) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}
EXPORT M_driver_status_t M_driver_function_occupancy_max_active_blocks(M_driver_context_t c, M_driver_function_t f, int32_t bs,
                                                                      uint64_t sm, int32_t *nb) {
  if (!nb) return M_DRIVER_STATUS_INVALID_ARG;
  *nb = 1;
  return OK;
}

EXPORT M_driver_status_t M_driver_queue_execute(M_driver_queue_t q, M_driver_function_t f,
                                                M_driver_queue_execute_config_t *cfg, void **args, uint64_t *sizes,
                                                uint32_t n) {
  if (!q || f != &g_matmul || !args || !sizes) return M_DRIVER_STATUS_INVALID_ARG;
  if (n != 6) return M_DRIVER_STATUS_INVALID_ARG;
  for (uint32_t i = 0; i < 6; i++) if (sizes[i] != 8) return M_DRIVER_STATUS_INVALID_ARG;
  uint64_t a = *(uint64_t *)args[0], b = *(uint64_t *)args[1], c = *(uint64_t *)args[2];
  int64_t m = *(int64_t *)args[3], k = *(int64_t *)args[4], nn = *(int64_t *)args[5];
  int64_t cyc = phantom_launch_matmul_i8(q->ctx->drv->sim, a, b, c, m, k, nn);
  if (cyc < 0) return sim_err("phantom_matmul_i8");
  q->last_cycles = cyc;
  q->launches++;
  HLOG("execute phantom_matmul_i8 M=%lld K=%lld N=%lld (mode %s) -> %lld cycles", (long long)m, (long long)k,
       (long long)nn, cfg && cfg->mode == M_DRIVER_QUEUE_EXECUTE_MODE_NPU ? "NPU" : "GPU", (long long)cyc);
  return OK;
}

// ------------------------------------------------------------------ TMA (not applicable)
EXPORT M_driver_status_t M_driver_tensor_map_encode_tiled(M_driver_context_t c, void *tm, int32_t dt, uint32_t r, uint64_t g,
                                                          const uint64_t *gd, const uint64_t *gs, const uint32_t *bd,
                                                          const uint32_t *es, int32_t il, int32_t sw, int32_t l2, int32_t oob) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}
EXPORT M_driver_status_t M_driver_tensor_map_encode_im2col(M_driver_context_t c, void *tm, int32_t dt, uint32_t r, uint64_t g,
                                                           const uint64_t *gd, const uint64_t *gs, const int32_t *lo,
                                                           const int32_t *hi, int32_t cpp, int32_t ppc, const uint32_t *es,
                                                           int32_t il, int32_t sw, int32_t l2, int32_t oob) {
  return M_DRIVER_STATUS_UNSUPPORTED;
}
