// SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING)
// #includes upstream r_draw.c (Copyright(C) 1993-1996 Id Software, Inc.; 2005-2014 Simon Howard).
// r_draw_gpu.c -- tier 3 (X-DOOM-3): DOOM's column/span drawing pass on the GPU.
//
// The eight R_Draw* entry points become *recorders*. Each call is range-checked
// exactly like upstream (same I_Error), applies upstream's side effects on the
// engine globals (fuzz clamps dc_yl/dc_yh and advances fuzzpos; SpanLow doubles
// ds_x1/ds_x2), and appends one fixed-size record to a per-frame command list.
// The texture bytes a call will read (column hull, 64x64 flat, colormap,
// translation table) are snapshotted into a per-frame arena at record time, so
// the GPU sees exactly the bytes the C drawer would have read at that moment.
// Snapshots are deduplicated by (pointer, length) and re-validated with memcmp.
//
// R_RenderPlayerView (wrapped in shim/r_main_gpu.c) calls r_gpu_frame_begin /
// r_gpu_frame_end around the C front end (BSP, clipping, planes, sprites). At
// frame end the list is replayed in ONE GPU dispatch by the Mojo Metal kernel
// in src/metal/gpu_replay.mojo (per pixel: last record covering it in
// submission order; columns with fuzz: in order, fuzz serial), and the 8-bit screen is copied back before the status
// bar / HUD / menus draw on top.
//
// -DDRAW_GPU_VERIFY: the recorders also run the upstream C drawer (renamed C_*)
// on the real screen; the GPU replays onto a snapshot of the pre-render screen
// and the two 64,000-byte screens are compared at the end of every view render.
// Side effects are compared per call. A randomised self-test (overlapping
// draws of all eight kinds, full-range fixed point) runs at the first frame.
#define R_DrawColumn              C_R_DrawColumn
#define R_DrawColumnLow           C_R_DrawColumnLow
#define R_DrawFuzzColumn          C_R_DrawFuzzColumn
#define R_DrawFuzzColumnLow       C_R_DrawFuzzColumnLow
#define R_DrawTranslatedColumn    C_R_DrawTranslatedColumn
#define R_DrawTranslatedColumnLow C_R_DrawTranslatedColumnLow
#define R_DrawSpan                C_R_DrawSpan
#define R_DrawSpanLow             C_R_DrawSpanLow
#include "r_draw.c"
#undef R_DrawColumn
#undef R_DrawColumnLow
#undef R_DrawFuzzColumn
#undef R_DrawFuzzColumnLow
#undef R_DrawTranslatedColumn
#undef R_DrawTranslatedColumnLow
#undef R_DrawSpan
#undef R_DrawSpanLow

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "i_video.h"

void R_DrawColumn(void);
void R_DrawColumnLow(void);
void R_DrawFuzzColumn(void);
void R_DrawFuzzColumnLow(void);
void R_DrawTranslatedColumn(void);
void R_DrawTranslatedColumnLow(void);
void R_DrawSpan(void);
void R_DrawSpanLow(void);

// --- Mojo side (src/metal/gpu_replay.mojo). Layout constants must match it. ---
#define GPU_REC 12                 /* int32 fields per record */
#define GPU_REC_CAP 65536          /* records per dispatch */
#define GPU_ARENA_CAP (8 << 20)    /* snapshot bytes per dispatch */
#define GPU_ARENA_HDR 256          /* arena[0..200): fuzzoffset[50] as int32 */
void *mojo_gpu_init(void);
int32_t mojo_gpu_flush(void *state, const int32_t *recs, int32_t nrec, const uint8_t *arena,
                       int32_t arena_used, uint8_t *screen);
void mojo_gpu_report(void *state, int32_t flushes);

enum { K_COL, K_COL_LOW, K_TCOL, K_TCOL_LOW, K_FUZZ, K_FUZZ_LOW, K_SPAN, K_SPAN_LOW };
#define NPIX (SCREENWIDTH * SCREENHEIGHT)

static void *g_st;
static int32_t *g_rec;
static uint8_t *g_arena;
static int g_n, g_used;
static uint8_t *g_target;   /* screen the replay applies to */
static int g_in_frame;

/* stats */
static unsigned long long s_frames, s_flushes, s_records, s_arena, s_pixels_dedup_hits;
static unsigned long long s_overflow, s_outside, s_fallback;
static double s_flush_ns;

#define HASH 16384
static struct { const byte *p; int len, off; unsigned gen; } g_hash[HASH];
static unsigned g_gen = 1;

static double now_ns(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static void arena_reset(void) {
    g_n = 0;
    g_gen++;
    memcpy(g_arena, fuzzoffset, sizeof fuzzoffset);
    g_used = GPU_ARENA_HDR;
}

void r_gpu_init(void) {
    if (g_st) return;
    /* plain cached memory: blob() re-reads snapshots (GPU staging may be write-combined) */
    g_rec = malloc(sizeof(int32_t) * GPU_REC * GPU_REC_CAP);
    g_arena = malloc(GPU_ARENA_CAP);
    g_st = mojo_gpu_init();
    if (!g_st || !g_rec || !g_arena) I_Error("r_gpu_init: GPU init failed");
    arena_reset();
}

static void gpu_flush(void) {
    if (!g_n) return;
    double t0 = now_ns();
    if (mojo_gpu_flush(g_st, g_rec, g_n, g_arena, g_used, g_target))
        I_Error("r_gpu: GPU replay failed");
    s_flush_ns += now_ns() - t0;
    s_flushes++;
    s_records += g_n;
    s_arena += g_used;
    arena_reset();
}

/* Make room for one record plus `bytes` of snapshots (flushes early if full). */
static void reserve(int bytes) {
    if (g_n + 1 > GPU_REC_CAP || g_used + bytes > GPU_ARENA_CAP) {
        s_overflow++;
        gpu_flush();
    }
}

/* Snapshot len bytes at p into the arena (dedup + memcmp re-validation). */
static int blob(const byte *p, int len) {
    uintptr_t k = (uintptr_t)p * 0x9E3779B97F4A7C15ull + (uintptr_t)len;
    unsigned h = (unsigned)(k >> 50) & (HASH - 1);
    for (int probe = 0; probe < 8; probe++, h = (h + 1) & (HASH - 1)) {
        if (g_hash[h].gen != g_gen) break;
        if (g_hash[h].p == p && g_hash[h].len == len) {
            if (!memcmp(g_arena + g_hash[h].off, p, len)) { s_pixels_dedup_hits++; return g_hash[h].off; }
            break;  /* contents changed since: take a fresh snapshot */
        }
    }
    int off = g_used;
    memcpy(g_arena + off, p, len);
    g_used += (len + 3) & ~3;
    g_hash[h].p = p; g_hash[h].len = len; g_hash[h].off = off; g_hash[h].gen = g_gen;
    return off;
}

/* Screen offset of a dest pointer, or -1 if it is not inside I_VideoBuffer. */
static int screen_off(const byte *dest) {
    ptrdiff_t d = dest - I_VideoBuffer;
    return (d >= 0 && d < NPIX) ? (int)d : -1;
}

/* Hull [lo, hi] of the texel indices (frac>>16, optionally &127) one column reads. */
static void texel_hull(fixed_t frac, fixed_t step, int count, int mask, int *lo, int *hi) {
    int64_t f0 = frac, f1 = (int64_t)frac + (int64_t)count * step;
    int64_t a = f0 < f1 ? f0 : f1, b = f0 < f1 ? f1 : f0;
    if (a >= INT32_MIN && b <= INT32_MAX) {   /* no int32 wraparound inside the column */
        int t0 = (int)(a >> FRACBITS), t1 = (int)(b >> FRACBITS);
        if (!mask) { *lo = t0; *hi = t1; return; }
        if (t1 - t0 >= 127 || (t0 & 127) > (t1 & 127)) { *lo = 0; *hi = 127; return; }
        *lo = t0 & 127; *hi = t1 & 127; return;
    }
    /* wraps (only with extreme iscale, e.g. the self-test): walk it like the drawer does */
    int mn = INT32_MAX, mx = INT32_MIN;
    uint32_t f = (uint32_t)frac;
    for (int i = 0; i <= count; i++) {
        int t = ((int32_t)f) >> FRACBITS;
        if (mask) t &= 127;
        if (t < mn) mn = t;
        if (t > mx) mx = t;
        f += (uint32_t)step;
    }
    *lo = mn; *hi = mx;
}

static int32_t *new_rec(void) {
    int32_t *r = g_rec + (size_t)g_n * GPU_REC;
    g_n++;
    return r;
}

/* Returns 0 if recorded, 1 if the geometry is outside what the GPU path models. */
static int rec_column(int kind) {
    int count = dc_yh - dc_yl;
    if (count < 0) return 0;
    int low = (kind == K_COL_LOW || kind == K_TCOL_LOW);
    int tr = (kind == K_TCOL || kind == K_TCOL_LOW);
    int x = low ? dc_x << 1 : dc_x;
    /* upstream RANGECHECK: ColumnLow checks dc_x, TranslatedColumnLow checks x<<1 */
    int xc = (kind == K_TCOL_LOW) ? x : dc_x;
    if ((unsigned)xc >= SCREENWIDTH || dc_yl < 0 || dc_yh >= SCREENHEIGHT)
        I_Error("R_DrawColumn: %i to %i at %i", dc_yl, dc_yh, xc);
    if (x + low >= SCREENWIDTH) return 1;
    int off = screen_off(ylookup[dc_yl] + columnofs[x]);
    if (off < 0 || off + count * SCREENWIDTH >= NPIX) return 1;
    int x0 = off % SCREENWIDTH;
    if (low && (columnofs[x + 1] != columnofs[x] + 1 || x0 + 1 >= SCREENWIDTH)) return 1;
    fixed_t frac = (fixed_t)((unsigned)dc_texturemid + (unsigned)(dc_yl - centery) * (unsigned)dc_iscale);
    int lo, hi;
    texel_hull(frac, dc_iscale, count, !tr, &lo, &hi);
    reserve((hi - lo + 1) + 256 + 256 + 12);
    int src = blob(dc_source + lo, hi - lo + 1) - lo;
    int cm = blob(dc_colormap, 256);
    int trn = tr ? blob(dc_translation, 256) : 0;
    int32_t *r = new_rec();
    r[0] = kind; r[1] = x0; r[2] = x0 + low; r[3] = off / SCREENWIDTH; r[4] = count + 1;
    r[5] = frac; r[6] = dc_iscale; r[7] = src; r[8] = cm; r[9] = trn; r[10] = r[11] = 0;
    return 0;
}

static int rec_fuzz(int kind) {
    int low = (kind == K_FUZZ_LOW);
    if (!dc_yl) dc_yl = 1;
    if (dc_yh == viewheight - 1) dc_yh = viewheight - 2;
    int count = dc_yh - dc_yl;
    if (count < 0) return 0;
    int x = low ? dc_x << 1 : dc_x;
    if ((unsigned)x >= SCREENWIDTH || dc_yl < 0 || dc_yh >= SCREENHEIGHT)
        I_Error("R_DrawFuzzColumn: %i to %i at %i", dc_yl, dc_yh, dc_x);
    if (x + low >= SCREENWIDTH) return 1;
    int off = screen_off(ylookup[dc_yl] + columnofs[x]);
    /* fuzz reads one row above / below every pixel */
    if (off < SCREENWIDTH || off + (count + 1) * SCREENWIDTH >= NPIX) return 1;
    int x0 = off % SCREENWIDTH;
    if (low && (columnofs[x + 1] != columnofs[x] + 1 || x0 + 1 >= SCREENWIDTH)) return 1;
    reserve(256 + 12);
    int cm = blob(colormaps + 6 * 256, 256);
    int32_t *r = new_rec();
    r[0] = kind; r[1] = x0; r[2] = x0 + low; r[3] = off / SCREENWIDTH; r[4] = count + 1;
    r[5] = r[6] = r[7] = 0; r[8] = cm; r[9] = fuzzpos; r[10] = r[11] = 0;
    fuzzpos = (fuzzpos + count + 1) % FUZZTABLE;   /* upstream: ++fuzzpos, wrap at FUZZTABLE */
    return 0;
}

static int rec_span(int kind) {
    int low = (kind == K_SPAN_LOW);
    if (ds_x2 < ds_x1 || ds_x1 < 0 || ds_x2 >= SCREENWIDTH || (unsigned)ds_y > SCREENHEIGHT)
        I_Error("R_DrawSpan: %i to %i at %i", ds_x1, ds_x2, ds_y);
    unsigned position = ((ds_xfrac << 10) & 0xffff0000) | ((ds_yfrac >> 6) & 0x0000ffff);
    unsigned step = ((ds_xstep << 10) & 0xffff0000) | ((ds_ystep >> 6) & 0x0000ffff);
    int count = ds_x2 - ds_x1;
    int x1 = low ? ds_x1 << 1 : ds_x1;
    int npx = low ? 2 * (count + 1) : count + 1;
    if (ds_y >= SCREENHEIGHT || x1 + npx > SCREENWIDTH) return 1;   /* (unsigned)ds_y == SCREENHEIGHT passes upstream's check */
    int off = screen_off(ylookup[ds_y] + columnofs[x1]);
    if (off < 0) return 1;
    int x0 = off % SCREENWIDTH;
    if (x0 + npx > SCREENWIDTH || columnofs[x1 + npx - 1] != columnofs[x1] + npx - 1) return 1;
    reserve(4096 + 256 + 12);
    int src = blob(ds_source, 4096);   /* spot = ((pos>>4)&0xfc0)|(pos>>26) is in [0, 4095] */
    int cm = blob(ds_colormap, 256);
    int32_t *r = new_rec();
    r[0] = kind; r[1] = x0; r[2] = x0 + npx - 1; r[3] = off / SCREENWIDTH; r[4] = count + 1;
    r[5] = (int32_t)position; r[6] = (int32_t)step; r[7] = src; r[8] = cm; r[9] = 0; r[10] = r[11] = 0;
    if (low) { ds_x1 <<= 1; ds_x2 <<= 1; }   /* upstream side effect */
    return 0;
}

static int record(int kind) {
    switch (kind) {
    case K_COL: case K_COL_LOW: case K_TCOL: case K_TCOL_LOW: return rec_column(kind);
    case K_FUZZ: case K_FUZZ_LOW: return rec_fuzz(kind);
    default: return rec_span(kind);
    }
}

typedef void (*drawfn)(void);
static drawfn c_fn[8] = {C_R_DrawColumn, C_R_DrawColumnLow, C_R_DrawTranslatedColumn,
                         C_R_DrawTranslatedColumnLow, C_R_DrawFuzzColumn, C_R_DrawFuzzColumnLow,
                         C_R_DrawSpan, C_R_DrawSpanLow};
static const char *k_names[8] = {"Column", "ColumnLow", "TranslatedColumn", "TranslatedColumnLow",
                                 "FuzzColumn", "FuzzColumnLow", "Span", "SpanLow"};
static unsigned long long s_calls[8];

/* side-effect state the drawers may change */
typedef struct { int yl, yh, fp, x1, x2; } fx_t;
static fx_t fx_get(void) { fx_t s = {dc_yl, dc_yh, fuzzpos, ds_x1, ds_x2}; return s; }
static void fx_set(fx_t s) { dc_yl = s.yl; dc_yh = s.yh; fuzzpos = s.fp; ds_x1 = s.x1; ds_x2 = s.x2; }

#ifndef DRAW_GPU_VERIFY
static void draw(int kind) {
    s_calls[kind]++;
    if (!g_st) r_gpu_init();
    if (!g_in_frame) g_target = I_VideoBuffer;
    fx_t s0 = fx_get();
    if (record(kind)) {   /* geometry the GPU path does not model: run upstream C in order */
        s_fallback++;
        fx_set(s0);
        gpu_flush();
        c_fn[kind]();
        return;
    }
    if (!g_in_frame) { s_outside++; gpu_flush(); }   /* drawer called outside a view render */
}
#else
static unsigned long long v_frames, v_bad_frames, v_bad_pixels, v_fx_bad, v_st_calls, v_st_bad_px, v_st_batches, v_st_bad_batches;
static uint8_t v_shadow[NPIX], v_orig[NPIX];
static int v_hooked;
static void v_report(void);
static void v_selftest(void);

static void draw(int kind) {
    s_calls[kind]++;
    if (!g_st) r_gpu_init();
    fx_t s0 = fx_get();
    int fb = record(kind);
    fx_t s1 = fx_get();
    fx_set(s0);
    c_fn[kind]();                       /* upstream C, on the real screen */
    fx_t s2 = fx_get();
    if (memcmp(&s1, &s2, sizeof s1)) v_fx_bad++;
    if (fb) {
        /* GPU path can't model it: apply the C result to the shadow too, in order */
        s_fallback++;
        gpu_flush();
        memcpy(v_shadow, I_VideoBuffer, NPIX);
    }
    if (!g_in_frame) s_outside++;
}

static unsigned long long v_diff(const uint8_t *a, const uint8_t *b) {
    unsigned long long n = 0;
    for (int i = 0; i < NPIX; i++) n += a[i] != b[i];
    return n;
}
#endif

void R_DrawColumn(void)              { draw(K_COL); }
void R_DrawColumnLow(void)           { draw(K_COL_LOW); }
void R_DrawTranslatedColumn(void)    { draw(K_TCOL); }
void R_DrawTranslatedColumnLow(void) { draw(K_TCOL_LOW); }
void R_DrawFuzzColumn(void)          { draw(K_FUZZ); }
void R_DrawFuzzColumnLow(void)       { draw(K_FUZZ_LOW); }
void R_DrawSpan(void)                { draw(K_SPAN); }
void R_DrawSpanLow(void)             { draw(K_SPAN_LOW); }

static int s_hooked;
static void s_report(void) {
    printf("GPU frames=%llu flushes=%llu records=%llu arena_bytes=%llu dedup_hits=%llu "
           "overflow_flushes=%llu outside_calls=%llu fallbacks=%llu flush_us_avg=%.1f\n",
           s_frames, s_flushes, s_records, s_arena, s_pixels_dedup_hits, s_overflow, s_outside,
           s_fallback, s_flushes ? s_flush_ns / s_flushes / 1e3 : 0.0);
    for (int i = 0; i < 8; i++)
        if (s_calls[i]) printf("GPU calls %s=%llu\n", k_names[i], s_calls[i]);
    fflush(stdout);
    if (g_st) mojo_gpu_report(g_st, (int32_t)s_flushes);
}

void r_gpu_frame_begin(void) {
    if (!s_hooked) { s_hooked = 1; I_AtExit(s_report, true); }
    if (!g_st) r_gpu_init();
    gpu_flush();   /* nothing should be pending; keep order if something is */
#ifdef DRAW_GPU_VERIFY
    if (!v_hooked) { v_hooked = 1; I_AtExit(v_report, true); v_selftest(); }
    memcpy(v_shadow, I_VideoBuffer, NPIX);
    g_target = v_shadow;
#else
    g_target = I_VideoBuffer;
#endif
    g_in_frame = 1;
}

void r_gpu_frame_end(void) {
    gpu_flush();
    g_in_frame = 0;
    s_frames++;
#ifdef DRAW_GPU_VERIFY
    unsigned long long d = v_diff(v_shadow, I_VideoBuffer);
    v_frames++;
    if (d) { v_bad_frames++; v_bad_pixels += d; }
    g_target = I_VideoBuffer;
#endif
}

#ifdef DRAW_GPU_VERIFY
static void v_report(void) {
    printf("GPUVERIFY selftest batches=%llu calls=%llu bad_batches=%llu bad_pixels=%llu\n",
           v_st_batches, v_st_calls, v_st_bad_batches, v_st_bad_px);
    printf("GPUVERIFY frames=%llu mismatched_frames=%llu mismatched_pixels=%llu sidefx_mismatches=%llu fallbacks=%llu\n",
           v_frames, v_bad_frames, v_bad_pixels, v_fx_bad, s_fallback);
    fflush(stdout);
}

// Randomised self-test (same generator as the tier-2 verifier): batches of
// overlapping draws of all eight kinds, replayed on the GPU in one dispatch
// per batch and compared against upstream C applied call by call.
static unsigned v_rng = 12345;
static unsigned v_rand(void) { v_rng = v_rng * 1103515245u + 12345u; return (v_rng >> 8) ^ (v_rng << 13); }
static int v_range(int lo, int hi) { return lo + (int)(v_rand() % (unsigned)(hi - lo + 1)); }
static void v_selftest(void) {
    static byte src[65536];
    for (int i = 0; i < 65536; i++) src[i] = (byte)v_rand();
    int vw = viewwidth, vh = viewheight, fp = fuzzpos;
    unsigned long long fb0 = s_fallback;
    const int BATCH = 2000, NB = 100;
    memcpy(v_orig, I_VideoBuffer, NPIX);
    for (int b = 0; b < NB; b++) {
        memcpy(v_shadow, I_VideoBuffer, NPIX);
        g_target = v_shadow;
        for (int n = 0; n < BATCH; n++) {
            int low = n & 1, kind = (n >> 1) % 4, wmax = (scaledviewwidth >> low) - 1, hmax = vh - 1;
            viewheight = vh;
            dc_colormap = colormaps + 256 * v_range(0, 31);
            ds_colormap = dc_colormap;
            dc_translation = translationtables + 256 * v_range(0, 2);
            dc_iscale = (fixed_t)v_rand(); dc_texturemid = (fixed_t)v_rand();
            dc_x = v_range(0, wmax);
            dc_yl = v_range(0, hmax); dc_yh = v_range(dc_yl, hmax);
            ds_xfrac = (fixed_t)v_rand(); ds_yfrac = (fixed_t)v_rand();
            ds_xstep = (fixed_t)v_rand(); ds_ystep = (fixed_t)v_rand();
            ds_y = v_range(0, hmax);
            ds_x1 = v_range(0, wmax); ds_x2 = v_range(ds_x1, wmax);
            if (kind == 0) { dc_source = src + v_range(0, 65536 - 128);
                             if (low) R_DrawColumnLow(); else R_DrawColumn(); }
            if (kind == 1) { dc_source = src + 32768;  /* no &127: index is frac>>16 in [-32768,32767] */
                             if (low) R_DrawTranslatedColumnLow(); else R_DrawTranslatedColumn(); }
            if (kind == 2) { ds_source = src + v_range(0, 65536 - 4096);
                             if (low) R_DrawSpanLow(); else R_DrawSpan(); }
            if (kind == 3) { viewheight = v_range(3, vh);  /* fuzz reads one row above/below */
                             dc_yl = v_range(0, viewheight - 1); dc_yh = v_range(dc_yl, viewheight - 1);
                             if (dc_yh > viewheight - 2 && dc_yh != viewheight - 1) dc_yh = viewheight - 2;
                             fuzzpos = v_range(0, FUZZTABLE - 1);
                             if (low) R_DrawFuzzColumnLow(); else R_DrawFuzzColumn(); }
            v_st_calls++;
        }
        gpu_flush();
        unsigned long long d = v_diff(v_shadow, I_VideoBuffer);
        v_st_batches++;
        if (d) { v_st_bad_batches++; v_st_bad_px += d; }
        memcpy(I_VideoBuffer, v_orig, NPIX);
    }
    viewwidth = vw; viewheight = vh; fuzzpos = fp;
    printf("GPUVERIFY selftest done (%d batches x %d randomised calls, %llu fallbacks)\n", NB, BATCH,
           s_fallback - fb0);
    v_report();
    s_outside = 0;
    for (int i = 0; i < 8; i++) s_calls[i] = 0;
    s_flushes = s_records = s_arena = s_overflow = s_pixels_dedup_hits = 0; s_flush_ns = 0;
}
#endif
