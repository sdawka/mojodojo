/* Driver for rtl/tile_v1_wb.v (v1 systolic tile, N=4, behind a small
 * Wishbone wrapper). Register map: see the RTL header. */
#include <stdint.h>
#include <generated/soc.h>
#include "tile.h"

#define TN 4
#define REG(off) (*(volatile uint32_t *)(TILE_BASE + (off)))
#define T_CTRL   0x00
#define T_KLEN   0x04
#define T_A      0x08
#define T_B      0x0C
#define T_CYCLES 0x10
#define T_ID     0x14
#define T_C(r,j) (0x40 + 4*((r)*TN + (j)))

const char *tile1_name(void) { return REG(T_ID) == 0x71110004u ? "tile v1 (N=4) wb wrapper" : "unknown tile"; }

void tile1_matmul(const int8_t *A, const int8_t *B, int32_t *C,
                 int M, int K, int Nout, struct tile_stats *st)
{
	st->passes = 0; st->busy_cycles = 0; st->ideal_cycles = 0;
	REG(T_KLEN) = (uint32_t)K;
	for (int m0 = 0; m0 < M; m0 += TN) {
		for (int n0 = 0; n0 < Nout; n0 += TN) {
			REG(T_CTRL) = 1;  /* start */
			for (int k = 0; k < K; k++) {
				uint32_t a = 0, b = 0;
				for (int i = 0; i < TN; i++) {
					if (m0 + i < M) a |= (uint32_t)(uint8_t)A[(m0 + i) * K + k] << (8 * i);
					if (n0 + i < Nout) b |= (uint32_t)(uint8_t)B[k * Nout + n0 + i] << (8 * i);
				}
				REG(T_A) = a;
				REG(T_B) = b;  /* pushes the beat (bus-stalled until accepted) */
			}
			while (!(REG(T_CTRL) & 4)) ;  /* done */
			for (int r = 0; r < TN && m0 + r < M; r++)
				for (int j = 0; j < TN && n0 + j < Nout; j++)
					C[(m0 + r) * Nout + n0 + j] = (int32_t)REG(T_C(r, j));
			st->passes++;
			st->busy_cycles += REG(T_CYCLES);
			st->ideal_cycles += (uint64_t)K + 3 * TN - 1;
		}
	}
}
