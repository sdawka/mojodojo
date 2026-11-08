/* Driver for fpga/tile2 (Wishbone slave, contract v2.0, see
 * fpga/tile2/INTERFACE.md §7). Operands go into the on-chip A/B buffers,
 * the sequencer runs all output tiles, C is read back. */
#include <stdint.h>
#include <generated/soc.h>
#include <generated/mem.h>
#include "sim.h"
#include "tile.h"

#ifdef TILE2_BASE
#define T2(off)  (*(volatile uint32_t *)(TILE2_BASE + (off)))
#define T2B(off) (*(volatile uint8_t  *)(TILE2_BASE + (off)))
#define R_ID 0x00
#define R_INFO 0x04
#define R_CTRL 0x08
#define R_STATUS 0x0C
#define R_MODE 0x10
#define R_M 0x14
#define R_K 0x18
#define R_N 0x1C
#define R_ABASE 0x20
#define R_ALD 0x24
#define R_BBASE 0x28
#define R_BLD 0x2C
#define R_CBASE 0x30
#define R_CLD 0x34
#define R_CYCLES 0x38
#define BUF_A 0x08000
#define BUF_B 0x10000
#define BUF_C 0x18000

int tile2_present(void) { return T2(R_ID) == 0x54494C32u; }
const char *tile2_name(void) { return tile2_present() ? "tile2 (TIL2) wishbone" : "tile2 absent"; }

static inline int rup(int x, int n) { return (x + n - 1) / n * n; }

void tile2_matmul(const int8_t *A, const int8_t *B, int32_t *C,
                  int M, int K, int Nout, struct tile_stats *st)
{
	uint32_t info = T2(R_INFO);
	int Nt = info & 0xff;
	int Mt = rup(M, Nt), Nr = rup(Nout, Nt);
	st->passes = 1;
	while (T2(R_STATUS) & 1) ;
	uint64_t t0 = sim_cycles();
	/* Aᵀ: K rows of Mt bytes, packed into words (word stores, 4 m per store) */
	for (int k = 0; k < K; k++)
		for (int m = 0; m < Mt; m += 4) {
			uint32_t w = 0;
			for (int b = 0; b < 4; b++)
				if (m + b < M) w |= (uint32_t)(uint8_t)A[(m + b) * K + k] << (8 * b);
			T2(BUF_A + k * Mt + m) = w;
		}
	for (int k = 0; k < K; k++)
		for (int n = 0; n < Nr; n += 4) {
			uint32_t w = 0;
			for (int b = 0; b < 4; b++)
				if (n + b < Nout) w |= (uint32_t)(uint8_t)B[k * Nout + n + b] << (8 * b);
			T2(BUF_B + k * Nr + n) = w;
		}
	T2(R_M) = M; T2(R_K) = K; T2(R_N) = Nout;
	T2(R_ABASE) = 0; T2(R_ALD) = Mt;
	T2(R_BBASE) = 0; T2(R_BLD) = Nr;
	T2(R_CBASE) = 0; T2(R_CLD) = 4 * Nr;
	T2(R_MODE) = 0;
	uint64_t t1 = sim_cycles();
	T2(R_CTRL) = 3;  /* CLEAR | START */
	while (T2(R_STATUS) & 1) ;
	uint64_t t2 = sim_cycles();
	uint32_t status = T2(R_STATUS);
	for (int m = 0; m < M; m++)
		for (int n = 0; n < Nout; n++)
			C[m * Nout + n] = (int32_t)T2(BUF_C + m * 4 * Nr + 4 * n);
	uint64_t t3 = sim_cycles();
	st->busy_cycles = T2(R_CYCLES);
	st->ideal_cycles = 2 + (uint64_t)(Mt / Nt) * (Nr / Nt) * (K + 3 * Nt);
	st->load_cycles = t1 - t0; st->run_cycles = t2 - t1; st->read_cycles = t3 - t2;
	if ((status & 0xe) != 0x2) st->busy_cycles |= 1ull << 63;  /* flag: not DONE or error */
	T2(R_CTRL) = 2;  /* CLEAR */
}
#else
int tile2_present(void) { return 0; }
const char *tile2_name(void) { return "tile2 not in this SoC"; }
void tile2_matmul(const int8_t *A, const int8_t *B, int32_t *C, int M, int K, int Nout, struct tile_stats *st)
{ (void)A; (void)B; (void)C; (void)M; (void)K; (void)Nout; st->passes = 0; }
#endif
