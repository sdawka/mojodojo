// SPDX-License-Identifier: GPL-2.0-or-later (see ../COPYING)
// dg_glue.c -- the only hand-written C on the tier-1 path (besides the engine).
//
// Mojo has no module-level globals, so the Mojo platform layer keeps its state
// in a heap struct and parks the pointer here. Everything else (DG_* callbacks,
// game loop, framebuffer, SDL, PNG/hash) is Mojo; see ../README.md.
#include <stdint.h>
#include "doomgeneric.h"
#include "i_system.h"

static void *g_host_ctx;

void  dg_glue_set_ctx(void *p) { g_host_ctx = p; }
void *dg_glue_get_ctx(void)    { return g_host_ctx; }

// Address of the engine's framebuffer pointer, so Mojo can install its own buffer.
pixel_t **dg_glue_screen_buffer_slot(void) { return &DG_ScreenBuffer; }

// The engine leaves via exit() (I_Quit / I_Error, e.g. at the end of -timedemo).
// Let Mojo flush its stats first: mojo_on_exit is a Mojo @export abi("C") function.
void mojo_on_exit(void);
void dg_glue_register_exit_hook(void) { I_AtExit(mojo_on_exit, 1); }
