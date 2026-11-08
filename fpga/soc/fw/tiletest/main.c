/* Milestone 2: random int8 matmuls on the tile peripherals (v1 wrapper and
 * tile2, both Wishbone slaves) checked against a CPU reference.
 * C[M][N] = A[M][K] * B[K][N], int32 wrapping accumulate.
 * scripts/check_tiletest.py regenerates the same matrices on the host and
 * checks the FNV-1a hash of every reference C independently. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <generated/soc.h>
#include "sim.h"
#include "tile.h"

#define MAXM 16
#define MAXK 256
#define MAXN 32
static const int shapes[][3] = { {13, 64, 9}, {8, 256, 32} };

static int8_t  A[MAXM * MAXK];
static int8_t  B[MAXK * MAXN];
static int32_t C_ref[MAXM * MAXN];
static int32_t C_tile[MAXM * MAXN];

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t xorshift(void)
{
	rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
	return (uint32_t)((rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static void ref_matmul(int M, int K, int N)
{
	for (int m = 0; m < M; m++)
		for (int n = 0; n < N; n++) {
			int32_t acc = 0;
			for (int k = 0; k < K; k++)
				acc += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
			C_ref[m * N + n] = acc;
		}
}

int main(void)
{
	sim_init();
	struct { const char *tag; const char *name; tile_matmul_fn fn; int present; } tiles[] = {
		{ "v1", tile1_name(), tile1_matmul, 1 },
		{ "v2", tile2_name(), tile2_matmul, tile2_present() },
	};
	for (unsigned t = 0; t < 2; t++)
		printf("tile %s: %s%s\n", tiles[t].tag, tiles[t].name, tiles[t].present ? "" : " (skipped)");
	int fails = 0, runs = 0;
	for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
		int M = shapes[s][0], K = shapes[s][1], N = shapes[s][2];
		for (int trial = 0; trial < 3; trial++) {
			for (int i = 0; i < M * K; i++) A[i] = (int8_t)xorshift();
			for (int i = 0; i < K * N; i++) B[i] = (int8_t)xorshift();
			if (trial == 2) {  /* extreme values: -128 * -128 accumulations */
				memset(A, 0x80, sizeof A);
				memset(B, 0x80, sizeof B);
			}
			uint64_t t0 = sim_cycles();
			ref_matmul(M, K, N);
			uint64_t cpu = sim_cycles() - t0;
			uint32_t h = 2166136261u;
			for (int i = 0; i < M * N; i++) { h ^= (uint32_t)C_ref[i]; h *= 16777619u; }
			for (unsigned t = 0; t < 2; t++) {
				if (!tiles[t].present) continue;
				struct tile_stats st;
				memset(&st, 0, sizeof st);
				memset(C_tile, 0x55, sizeof C_tile);
				uint64_t t1 = sim_cycles();
				tiles[t].fn(A, B, C_tile, M, K, N, &st);
				uint64_t total = sim_cycles() - t1;
#ifdef CORRUPT  /* negative check: the comparison must catch a one-bit error */
				C_tile[(M / 2) * N + N / 2] ^= 1;
#endif
				int bad = 0;
				for (int i = 0; i < M * N; i++)
					if (C_tile[i] != C_ref[i]) {
						if (bad < 3)
							printf("  mismatch %s C[%d][%d]: tile %ld ref %ld\n", tiles[t].tag,
							       i / N, i % N, (long)C_tile[i], (long)C_ref[i]);
						bad++;
					}
				if (st.busy_cycles >> 63) { printf("  %s: status not DONE / error\n", tiles[t].tag); bad++; }
				printf("TILE_RESULT tile=%s trial=%d M=%d K=%d N=%d mismatches=%d fnv=0x%08lx "
				       "cpu_ref_cycles=%lu tile_total_cycles=%lu tile_busy_cycles=%lu ideal_cycles=%lu "
				       "load_cycles=%lu run_cycles=%lu read_cycles=%lu\n",
				       tiles[t].tag, trial, M, K, N, bad, (unsigned long)h, (unsigned long)cpu,
				       (unsigned long)total, (unsigned long)st.busy_cycles, (unsigned long)st.ideal_cycles,
				       (unsigned long)st.load_cycles, (unsigned long)st.run_cycles,
				       (unsigned long)st.read_cycles);
				fails += bad != 0;
				runs++;
			}
		}
	}
	printf("TILETEST %s (%d runs)\n", fails ? "FAIL" : "PASS", runs);
	sim_finish(fails ? 1 : 0);
}
