/* SPDX-License-Identifier: GPL-2.0-or-later (links the doomgeneric engine)
 *
 * doomgeneric platform layer for the fpga/soc VexRiscv SoC (no OS):
 *   - DG_* callbacks: virtual clock (DG_SleepMs advances it, nothing else
 *     does) exactly like doom/src/host.mojo --headless --vclock, so frame N
 *     here is the same game state as host frame N.
 *   - DG_DrawFrame: SHA-256 of the 320x200 XRGB8888 DG_ScreenBuffer for
 *     selected frames, optional hex dump over the UART, $finish at the end.
 *   - POSIX shims for picolibc's stdio: a read-only in-memory file system
 *     holding the IWAD, which run.py preloads into main RAM at WAD_HDR_ADDR.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "doomgeneric.h"
#include "sim.h"
#include "sha256.h"

#ifndef WAD_HDR_ADDR
#define WAD_HDR_ADDR 0x42000000u
#endif
#ifndef DOOM_MAX_FRAME
#define DOOM_MAX_FRAME 1
#endif
#ifndef DOOM_DUMP_FRAME
#define DOOM_DUMP_FRAME 1
#endif
/* frames whose SHA-256 is printed (in addition to DOOM_MAX_FRAME) */
#ifndef DOOM_HASH_EVERY
#define DOOM_HASH_EVERY 1
#endif

struct wad_hdr { uint32_t magic, size, rsv0, rsv1; };  /* magic 'SWAD' */
#define WAD_MAGIC 0x44415753u

static const char *wad_name = "freedoom1.wad";

/* ---------------------------------------------------------------- files */
static struct { int used; uint32_t pos; } fds[8];

static const uint8_t *wad_data(uint32_t *size)
{
	const struct wad_hdr *h = (const struct wad_hdr *)WAD_HDR_ADDR;
	if (h->magic != WAD_MAGIC) return NULL;
	*size = h->size;
	return (const uint8_t *)(h + 1);
}

static const char *base_name(const char *p)
{
	const char *b = strrchr(p, '/');
	return b ? b + 1 : p;
}

int open(const char *path, int flags, ...)
{
	uint32_t sz;
	if ((flags & O_ACCMODE) != O_RDONLY || strcmp(base_name(path), wad_name) != 0 || !wad_data(&sz)) {
		errno = ENOENT;
		return -1;
	}
	for (int i = 3; i < 8; i++)
		if (!fds[i].used) { fds[i].used = 1; fds[i].pos = 0; return i; }
	errno = EMFILE;
	return -1;
}

ssize_t read(int fd, void *buf, size_t n)
{
	uint32_t sz;
	const uint8_t *d = wad_data(&sz);
	if (fd < 3 || fd >= 8 || !fds[fd].used || !d) { errno = EBADF; return -1; }
	if (fds[fd].pos >= sz) return 0;
	if (n > sz - fds[fd].pos) n = sz - fds[fd].pos;
	memcpy(buf, d + fds[fd].pos, n);
	fds[fd].pos += n;
	return (ssize_t)n;
}

ssize_t write(int fd, const void *buf, size_t n)
{
	if (fd == 1 || fd == 2) {
		const char *c = buf;
		for (size_t i = 0; i < n; i++) putchar(c[i]);
		return (ssize_t)n;
	}
	errno = EBADF;
	return -1;
}

off_t lseek(int fd, off_t off, int whence)
{
	uint32_t sz;
	if (fd < 3 || fd >= 8 || !fds[fd].used || !wad_data(&sz)) { errno = EBADF; return -1; }
	off_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? (off_t)fds[fd].pos : (off_t)sz;
	if (base + off < 0) { errno = EINVAL; return -1; }
	fds[fd].pos = (uint32_t)(base + off);
	return (off_t)fds[fd].pos;
}

int close(int fd)
{
	if (fd < 3 || fd >= 8 || !fds[fd].used) { errno = EBADF; return -1; }
	fds[fd].used = 0;
	return 0;
}

int fstat(int fd, struct stat *st) { (void)fd; memset(st, 0, sizeof *st); return 0; }
int isatty(int fd) { return fd < 3; }
int unlink(const char *p) { (void)p; errno = ENOENT; return -1; }
int mkdir(const char *p, mode_t m) { (void)p; (void)m; return 0; }
int rename(const char *a, const char *b) { (void)a; (void)b; errno = ENOENT; return -1; }
int system(const char *c) { (void)c; return -1; }

void _exit(int code)
{
	printf("\n[doom] exit(%d)\n", code);
	sim_finish(code == 0 ? 3 : 2);
}

/* ---------------------------------------------------------------- DG_* */
static uint32_t vclock_ms;
static uint32_t frame;
static uint64_t t_start;

void DG_Init(void) {}
void DG_SleepMs(uint32_t ms) { vclock_ms += ms; }
uint32_t DG_GetTicksMs(void) { return vclock_ms; }
int DG_GetKey(int *pressed, unsigned char *key) { (void)pressed; (void)key; return 0; }
void DG_SetWindowTitle(const char *t) { (void)t; }

static void dump_frame(void)
{
	static const char hx[] = "0123456789abcdef";
	char line[6 * DOOMGENERIC_RESX + 1];
	for (int y = 0; y < DOOMGENERIC_RESY; y++) {
		const uint32_t *row = &DG_ScreenBuffer[y * DOOMGENERIC_RESX];
		for (int x = 0; x < DOOMGENERIC_RESX; x++)
			for (int k = 0; k < 6; k++)
				line[6 * x + k] = hx[(row[x] >> (20 - 4 * k)) & 15];
		line[6 * DOOMGENERIC_RESX] = 0;
		printf("FB %03d %s\n", y, line);
	}
}

void DG_DrawFrame(void)
{
	frame++;
	uint64_t now = sim_cycles();
	if (frame == 1 || frame == DOOM_MAX_FRAME || (frame % DOOM_HASH_EVERY) == 0) {
		uint8_t d[32];
		sha256(DG_ScreenBuffer, DOOMGENERIC_RESX * DOOMGENERIC_RESY * 4, d);
		printf("FRAME %lu cycles=%llu vclock_ms=%lu sha256=", (unsigned long)frame,
		       (unsigned long long)(now - t_start), (unsigned long)vclock_ms);
		for (int i = 0; i < 32; i++) printf("%02x", d[i]);
		printf("\n");
	}
	if (frame == DOOM_DUMP_FRAME) {
		printf("FB_BEGIN %lu %dx%d xrgb\n", (unsigned long)frame, DOOMGENERIC_RESX, DOOMGENERIC_RESY);
		dump_frame();
		printf("FB_END\n");
	}
	if (frame >= DOOM_MAX_FRAME) {
		printf("DOOM_DONE frames=%lu cycles=%llu\n", (unsigned long)frame,
		       (unsigned long long)(sim_cycles() - t_start));
		sim_finish(0);
	}
}

int main(void)
{
	sim_init();
	t_start = sim_cycles();
	uint32_t sz;
	if (!wad_data(&sz)) {
		printf("[doom] no IWAD preloaded at 0x%08lx\n", (unsigned long)WAD_HDR_ADDR);
		sim_finish(4);
	}
	printf("[doom] IWAD %s: %lu bytes at 0x%08lx\n", wad_name, (unsigned long)sz,
	       (unsigned long)(WAD_HDR_ADDR + 16));
	static char *argv[] = { "doom", "-iwad", "freedoom1.wad", NULL };
	doomgeneric_Create(3, argv);
	for (;;)
		doomgeneric_Tick();
}
