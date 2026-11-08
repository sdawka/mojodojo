// SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING)
// #includes upstream r_main.c (Copyright(C) 1993-1996 Id Software, Inc.; 2005-2014 Simon Howard).
// r_main_gpu.c -- tier 3 (X-DOOM-3): bracket the view render so the recorded
// column/span draws of one frame are replayed on the GPU in one dispatch,
// before the status bar, HUD and menus are drawn over the view.
// Linked as an object ahead of libdoomcore.a, so the archive's r_main.o is not used.
#define R_RenderPlayerView C_R_RenderPlayerView
#include "r_main.c"
#undef R_RenderPlayerView

void r_gpu_frame_begin(void);
void r_gpu_frame_end(void);

void R_RenderPlayerView(player_t *player)
{
    r_gpu_frame_begin();
    C_R_RenderPlayerView(player);
    r_gpu_frame_end();
}
