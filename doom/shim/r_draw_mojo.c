// SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING)
// #includes upstream r_draw.c (Copyright(C) 1993-1996 Id Software, Inc.; 2005-2014 Simon Howard).
// r_draw_mojo.c -- tier 2 (X-DOOM-2): route DOOM's column/span drawers to Mojo.
//
// Includes the unmodified upstream r_draw.c with its six drawers renamed to C_*
// (so every other r_draw.c function and global stays exactly upstream), then
// defines R_Draw* as marshalling stubs: they read the engine globals and pass
// them as arguments to Mojo kernels (Mojo has no access to C globals). All the
// per-pixel work and fixed-point maths happen in Mojo (src/drawers.mojo).
// Mojo returns nonzero when upstream's RANGECHECK would fire; we raise the same
// I_Error here so error semantics match too.
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

int mojo_draw_column(byte **, int *, int, int, int, int, fixed_t, fixed_t, byte *, byte *, byte *);
int mojo_draw_column_low(byte **, int *, int, int, int, int, fixed_t, fixed_t, byte *, byte *, byte *);
int mojo_draw_translated_column(byte **, int *, int, int, int, int, fixed_t, fixed_t, byte *, byte *, byte *);
int mojo_draw_translated_column_low(byte **, int *, int, int, int, int, fixed_t, fixed_t, byte *, byte *, byte *);
int mojo_draw_fuzz_column(byte **ylookup, int *columnofs, int x, int *yl, int *yh,
                          int viewheight, byte *colormaps, int *fuzzoffset, int *fuzzpos);
int mojo_draw_fuzz_column_low(byte **, int *, int, int *, int *, int, byte *, int *, int *);
int mojo_draw_span(byte **ylookup, int *columnofs, int y, int *x1, int *x2,
                   fixed_t xfrac, fixed_t yfrac, fixed_t xstep, fixed_t ystep,
                   byte *source, byte *colormap);
int mojo_draw_span_low(byte **, int *, int, int *, int *, fixed_t, fixed_t, fixed_t, fixed_t, byte *, byte *);

void R_DrawColumn(void);
void R_DrawColumnLow(void);
void R_DrawFuzzColumn(void);
void R_DrawFuzzColumnLow(void);
void R_DrawTranslatedColumn(void);
void R_DrawTranslatedColumnLow(void);
void R_DrawSpan(void);
void R_DrawSpanLow(void);

#define COL_ARGS ylookup, columnofs, dc_x, dc_yl, dc_yh, centery, dc_iscale, dc_texturemid, dc_source, dc_colormap
static void M_R_DrawColumn(void) {
    if (mojo_draw_column(COL_ARGS, NULL))
        I_Error("R_DrawColumn: %i to %i at %i", dc_yl, dc_yh, dc_x);
}
static void M_R_DrawColumnLow(void) {
    if (mojo_draw_column_low(COL_ARGS, NULL))
        I_Error("R_DrawColumn: %i to %i at %i", dc_yl, dc_yh, dc_x);
}
static void M_R_DrawTranslatedColumn(void) {
    if (mojo_draw_translated_column(COL_ARGS, dc_translation))
        I_Error("R_DrawColumn: %i to %i at %i", dc_yl, dc_yh, dc_x);
}
static void M_R_DrawTranslatedColumnLow(void) {
    if (mojo_draw_translated_column_low(COL_ARGS, dc_translation))
        I_Error("R_DrawColumn: %i to %i at %i", dc_yl, dc_yh, dc_x << 1);
}
static void M_R_DrawFuzzColumn(void) {
    if (mojo_draw_fuzz_column(ylookup, columnofs, dc_x, &dc_yl, &dc_yh, viewheight,
                              colormaps, fuzzoffset, &fuzzpos))
        I_Error("R_DrawFuzzColumn: %i to %i at %i", dc_yl, dc_yh, dc_x);
}
static void M_R_DrawFuzzColumnLow(void) {
    if (mojo_draw_fuzz_column_low(ylookup, columnofs, dc_x, &dc_yl, &dc_yh, viewheight,
                                  colormaps, fuzzoffset, &fuzzpos))
        I_Error("R_DrawFuzzColumn: %i to %i at %i", dc_yl, dc_yh, dc_x);
}
static void M_R_DrawSpan(void) {
    if (mojo_draw_span(ylookup, columnofs, ds_y, &ds_x1, &ds_x2, ds_xfrac, ds_yfrac,
                       ds_xstep, ds_ystep, ds_source, ds_colormap))
        I_Error("R_DrawSpan: %i to %i at %i", ds_x1, ds_x2, ds_y);
}
static void M_R_DrawSpanLow(void) {
    if (mojo_draw_span_low(ylookup, columnofs, ds_y, &ds_x1, &ds_x2, ds_xfrac, ds_yfrac,
                           ds_xstep, ds_ystep, ds_source, ds_colormap))
        I_Error("R_DrawSpan: %i to %i at %i", ds_x1, ds_x2, ds_y);
}

// --------------------------------------------------------------------------
// Public entry points. Default build: straight to the Mojo stubs above.
// -DDRAW_VERIFY build: differential check of every drawer call during a run.
// The C drawer (upstream code, renamed C_*) and the Mojo drawer are both run
// on the same input pixels and engine state; outputs (pixels + side effects on
// dc_yl/dc_yh/fuzzpos/ds_x1/ds_x2) must be identical. The C result is kept.
// This sidesteps upstream's run-to-run nondeterminism (it reads uninitialised
// zone memory at startup), which makes whole-frame hashes noisy for ~30 frames.
// --------------------------------------------------------------------------
#ifndef DRAW_VERIFY
void R_DrawColumn(void)              { M_R_DrawColumn(); }
void R_DrawColumnLow(void)           { M_R_DrawColumnLow(); }
void R_DrawFuzzColumn(void)          { M_R_DrawFuzzColumn(); }
void R_DrawFuzzColumnLow(void)       { M_R_DrawFuzzColumnLow(); }
void R_DrawTranslatedColumn(void)    { M_R_DrawTranslatedColumn(); }
void R_DrawTranslatedColumnLow(void) { M_R_DrawTranslatedColumnLow(); }
void R_DrawSpan(void)                { M_R_DrawSpan(); }
void R_DrawSpanLow(void)             { M_R_DrawSpanLow(); }
#else
#include <stdio.h>
#include <string.h>
#define VMAX (2 * SCREENWIDTH * SCREENHEIGHT)
static unsigned long long v_calls[8], v_bad[8];
static const char *v_names[8] = {"Column", "ColumnLow", "FuzzColumn", "FuzzColumnLow",
                                 "TranslatedColumn", "TranslatedColumnLow", "Span", "SpanLow"};
static int v_hooked;
typedef struct { byte *p[VMAX / 64]; int n; } region_t;  /* <= 2 cols * 200 or 640 px */
static void v_report(void) {
    unsigned long long c = 0, b = 0;
    for (int i = 0; i < 8; i++) {
        c += v_calls[i]; b += v_bad[i];
        if (v_calls[i]) printf("VERIFY %s calls=%llu mismatches=%llu\n", v_names[i], v_calls[i], v_bad[i]);
    }
    printf("VERIFY total calls=%llu mismatches=%llu\n", c, b);
    fflush(stdout);
}
static void v_selftest(void);
static void v_hook(void) { if (!v_hooked) { v_hooked = 1; I_AtExit(v_report, true); v_selftest(); } }
static void reg_cols(region_t *r, int x, int yl, int yh, int ncols) {
    r->n = 0;
    if (yl < 0) yl = 0;
    if (yh > SCREENHEIGHT - 1) yh = SCREENHEIGHT - 1;
    if (x < 0 || x + ncols > SCREENWIDTH) return;
    for (int y = yl; y <= yh; y++)
        for (int c = 0; c < ncols; c++) r->p[r->n++] = ylookup[y] + columnofs[x + c];
}
static void reg_span(region_t *r, int y, int x1, int x2) {
    r->n = 0;
    if (y < 0 || y >= SCREENHEIGHT) return;
    if (x1 < 0) x1 = 0;
    if (x2 > SCREENWIDTH - 1) x2 = SCREENWIDTH - 1;
    for (int x = x1; x <= x2; x++) r->p[r->n++] = ylookup[y] + columnofs[x];
}
static void v_save(region_t *r, byte *b) { for (int i = 0; i < r->n; i++) b[i] = *r->p[i]; }
static void v_load(region_t *r, byte *b) { for (int i = 0; i < r->n; i++) *r->p[i] = b[i]; }

#define VERIFY(K, REGION, CFN, MFN, ST)                                        \
    do {                                                                       \
        region_t r; byte orig[VMAX / 64], cres[VMAX / 64];                     \
        int st0[4], stc[4];                                                    \
        v_hook(); REGION; v_save(&r, orig); ST(st0, 0);                        \
        CFN(); v_save(&r, cres); ST(stc, 0);                                   \
        v_load(&r, orig); ST(st0, 1);                                          \
        MFN();                                                                 \
        byte mres[VMAX / 64]; int stm[4]; v_save(&r, mres); ST(stm, 0);        \
        v_calls[K]++;                                                          \
        if (memcmp(cres, mres, r.n) || memcmp(stc, stm, sizeof stc)) v_bad[K]++; \
        v_load(&r, cres); ST(stc, 1);                                          \
    } while (0)
#define ST_NONE(a, set) do { if (!(set)) memset(a, 0, 4 * sizeof(int)); } while (0)
#define ST_FUZZ(a, set) do { if (set) { dc_yl = a[0]; dc_yh = a[1]; fuzzpos = a[2]; } \
                             else { a[0] = dc_yl; a[1] = dc_yh; a[2] = fuzzpos; a[3] = 0; } } while (0)
#define ST_SPAN(a, set) do { if (set) { ds_x1 = a[0]; ds_x2 = a[1]; } \
                             else { a[0] = ds_x1; a[1] = ds_x2; a[2] = a[3] = 0; } } while (0)

void R_DrawColumn(void) {
    VERIFY(0, reg_cols(&r, dc_x, dc_yl, dc_yh, 1), C_R_DrawColumn, M_R_DrawColumn, ST_NONE);
}
void R_DrawColumnLow(void) {
    VERIFY(1, reg_cols(&r, dc_x << 1, dc_yl, dc_yh, 2), C_R_DrawColumnLow, M_R_DrawColumnLow, ST_NONE);
}
void R_DrawFuzzColumn(void) {
    VERIFY(2, reg_cols(&r, dc_x, dc_yl, dc_yh, 1), C_R_DrawFuzzColumn, M_R_DrawFuzzColumn, ST_FUZZ);
}
void R_DrawFuzzColumnLow(void) {
    VERIFY(3, reg_cols(&r, dc_x << 1, dc_yl, dc_yh, 2), C_R_DrawFuzzColumnLow, M_R_DrawFuzzColumnLow, ST_FUZZ);
}
void R_DrawTranslatedColumn(void) {
    VERIFY(4, reg_cols(&r, dc_x, dc_yl, dc_yh, 1), C_R_DrawTranslatedColumn, M_R_DrawTranslatedColumn, ST_NONE);
}
void R_DrawTranslatedColumnLow(void) {
    VERIFY(5, reg_cols(&r, dc_x << 1, dc_yl, dc_yh, 2), C_R_DrawTranslatedColumnLow, M_R_DrawTranslatedColumnLow, ST_NONE);
}
void R_DrawSpan(void) {
    VERIFY(6, reg_span(&r, ds_y, ds_x1, ds_x2), C_R_DrawSpan, M_R_DrawSpan, ST_SPAN);
}
void R_DrawSpanLow(void) {
    VERIFY(7, reg_span(&r, ds_y, ds_x1 << 1, (ds_x2 << 1) + 1), C_R_DrawSpanLow, M_R_DrawSpanLow, ST_SPAN);
}

// Randomised self-test, run once at the first drawer call (engine initialised):
// all eight drawers, full-range int32 iscale/texturemid/xfrac/... (exercises
// wraparound), every colormap/translation table. The translated drawers are
// only reached in multiplayer, so the demo alone never calls them.
static unsigned v_rng = 12345;
static unsigned v_rand(void) { v_rng = v_rng * 1103515245u + 12345u; return (v_rng >> 8) ^ (v_rng << 13); }
static int v_range(int lo, int hi) { return lo + (int)(v_rand() % (unsigned)(hi - lo + 1)); }
static void v_selftest(void) {
    static byte src[65536];
    for (int i = 0; i < 65536; i++) src[i] = (byte)v_rand();
    int vw = viewwidth, vh = viewheight;
    for (int n = 0; n < 200000; n++) {
        /* ylookup/columnofs are only valid inside the current view window */
        int low = n & 1, kind = (n >> 1) % 4, wmax = (scaledviewwidth >> low) - 1, hmax = vh - 1;
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
    }
    viewwidth = vw; viewheight = vh;
    printf("VERIFY selftest done (200000 randomised calls)\n");
    v_report();
    for (int i = 0; i < 8; i++) v_calls[i] = v_bad[i] = 0;  /* demo counts start fresh */
}
#endif
