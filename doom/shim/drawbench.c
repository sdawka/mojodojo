// SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING)
// drawbench.c -- drawer-only microbenchmark (host flag --microbench).
// Linked with r_draw_mojo.o, where the upstream drawers exist renamed C_R_Draw*
// and R_Draw* are the stubs that call the Mojo kernels. Times both on the same
// synthetic inputs (320x200 screen, random texture/colormaps), so the Mojo
// numbers include the per-call stub + FFI argument marshalling.
#include <stdio.h>
#include <time.h>
#include "doomdef.h"
#include "r_local.h"

#ifndef HAVE_MOJO_DRAW
void dg_drawer_microbench(void) { printf("MICRO unavailable in this build (stock C drawers only)\n"); }
#else
void C_R_DrawColumn(void); void C_R_DrawSpan(void); void C_R_DrawFuzzColumn(void);
extern int fuzzpos;
extern byte *ylookup[]; extern int columnofs[];

static double now_ns(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static byte screen[SCREENWIDTH * SCREENHEIGHT], tex[65536], cmaps[34 * 256];
static unsigned rng = 1;
static unsigned rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }

typedef void (*drawfn)(void);
static double time_cols(drawfn f, int h, int reps) {
    double t0 = now_ns(); long px = 0;
    for (int r = 0; r < reps; r++)
        for (int x = 0; x < SCREENWIDTH; x++) {
            dc_x = x; dc_yl = (x * 7) % (SCREENHEIGHT - h); dc_yh = dc_yl + h - 1;
            dc_iscale = 0x8000 + (x << 6); dc_texturemid = x << 13;
            dc_source = tex + ((x * 128) & 0xff80); dc_colormap = cmaps + 256 * (x & 31);
            f(); px += h;
        }
    return (now_ns() - t0) / px;
}
static double time_spans(drawfn f, int w, int reps) {
    double t0 = now_ns(); long px = 0;
    for (int r = 0; r < reps; r++)
        for (int y = 0; y < SCREENHEIGHT; y++) {
            ds_y = y; ds_x1 = (y * 13) % (SCREENWIDTH - w); ds_x2 = ds_x1 + w - 1;
            ds_xfrac = y << 15; ds_yfrac = y << 14; ds_xstep = 0x4000 + y; ds_ystep = 0x2000 - y;
            ds_source = tex + ((y * 4096) & 0xf000); ds_colormap = cmaps + 256 * (y & 31);
            f(); px += w;
        }
    return (now_ns() - t0) / px;
}
void dg_drawer_microbench(void) {
    for (int i = 0; i < (int)sizeof tex; i++) tex[i] = rnd();
    for (int i = 0; i < (int)sizeof cmaps; i++) cmaps[i] = rnd();
    colormaps = cmaps; viewheight = SCREENHEIGHT; centery = SCREENHEIGHT / 2;
    for (int y = 0; y < SCREENHEIGHT; y++) ylookup[y] = screen + y * SCREENWIDTH;
    for (int x = 0; x < SCREENWIDTH; x++) columnofs[x] = x;
    struct { const char *name; int col; int len; drawfn c, m; } cases[] = {
        {"column_h168", 1, 168, C_R_DrawColumn, R_DrawColumn},
        {"column_h16", 1, 16, C_R_DrawColumn, R_DrawColumn},
        {"fuzz_h100", 1, 100, C_R_DrawFuzzColumn, R_DrawFuzzColumn},
        {"span_w320", 0, 319, C_R_DrawSpan, R_DrawSpan},
        {"span_w16", 0, 16, C_R_DrawSpan, R_DrawSpan},
    };
    for (unsigned k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        double best[2] = {1e9, 1e9};
        for (int trial = 0; trial < 7; trial++)
            for (int v = 0; v < 2; v++) {
                drawfn f = v ? cases[k].m : cases[k].c;
                int reps = cases[k].len >= 100 ? 40 : 400;
                fuzzpos = 0;
                double t = cases[k].col ? time_cols(f, cases[k].len, reps) : time_spans(f, cases[k].len, reps);
                if (t < best[v]) best[v] = t;
            }
        printf("MICRO %s c_ns_per_px=%.3f mojo_ns_per_px=%.3f mojo_over_c=%.3f\n",
               cases[k].name, best[0], best[1], best[1] / best[0]);
    }
    fflush(stdout);
}
#endif
