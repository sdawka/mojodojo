/* Host-side drivers for the matmul tile peripherals. */
#ifndef FW_TILE_H
#define FW_TILE_H
#include <stdint.h>

struct tile_stats {
	uint32_t passes;        /* v1: output-tile passes; v2: runs (START writes) */
	uint64_t busy_cycles;   /* tile-reported busy cycles (v1 per pass, v2 CYCLES reg) */
	uint64_t ideal_cycles;  /* v1: sum K+3N-1 (DESIGN.md §5); v2: 2+tiles*(K+3Nt) (INTERFACE.md §6) */
	uint64_t load_cycles;   /* v2: SoC cycles spent writing A/B into the tile buffers */
	uint64_t run_cycles;    /* v2: SoC cycles from START to DONE observed */
	uint64_t read_cycles;   /* v2: SoC cycles reading C back */
};

typedef void (*tile_matmul_fn)(const int8_t *A, const int8_t *B, int32_t *C,
                               int M, int K, int Nout, struct tile_stats *st);
/* C[M][Nout] = A[M][K] (row-major int8) x B[K][Nout] (row-major int8), int32 wrap. */
const char *tile1_name(void);
void tile1_matmul(const int8_t *A, const int8_t *B, int32_t *C, int M, int K, int Nout, struct tile_stats *st);
const char *tile2_name(void);
int  tile2_present(void);
void tile2_matmul(const int8_t *A, const int8_t *B, int32_t *C, int M, int K, int Nout, struct tile_stats *st);
#endif
