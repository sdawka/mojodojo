/* Links lane-rvmojo's Mojo-compiled C-ABI object (fpga/rvmojo/fw/libtile.mojo
 * -> libtile.o) into C firmware: the Mojo drivers for tile v1 / tile2 and the
 * Mojo CPU reference are checked against a C reference. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sim.h"

uint32_t mojo_tile_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int M, int K, int N);
int32_t  mojo_tile2_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int M, int K, int N);
void     mojo_cpu_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int M, int K, int N);

#define M 12
#define K 40
#define N 10
static int8_t A[M * K], B[K * N];
static int32_t Cr[M * N], Cm[M * N];

static int check(const char *what)
{
	int bad = 0;
	for (int i = 0; i < M * N; i++) bad += Cm[i] != Cr[i];
	printf("MOJOLINK %s mismatches=%d\n", what, bad);
	return bad != 0;
}

int main(void)
{
	sim_init();
	uint32_t s = 12345;
	for (int i = 0; i < M * K; i++) { s = s * 1103515245u + 12345u; A[i] = (int8_t)(s >> 16); }
	for (int i = 0; i < K * N; i++) { s = s * 1103515245u + 12345u; B[i] = (int8_t)(s >> 16); }
	for (int m = 0; m < M; m++)
		for (int n = 0; n < N; n++) {
			int32_t acc = 0;
			for (int k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
			Cr[m * N + n] = acc;
		}
	int fails = 0;
	/* known answer from lane-rvmojo: 4x4x4, A[i][k]=i+k-3, B[k][j]=k-2j -> C[0][0] = -4 */
	int8_t a4[16], b4[16]; int32_t c4[16];
	for (int i = 0; i < 4; i++) for (int k = 0; k < 4; k++) a4[i * 4 + k] = i + k - 3;
	for (int k = 0; k < 4; k++) for (int j = 0; j < 4; j++) b4[k * 4 + j] = k - 2 * j;
	mojo_cpu_matmul_i8(a4, b4, c4, 4, 4, 4);
	printf("MOJOLINK known-answer C[0][0]=%ld (expect -4)\n", (long)c4[0]);
	fails += c4[0] != -4;
	memset(Cm, 0x55, sizeof Cm); mojo_cpu_matmul_i8(A, B, Cm, M, K, N); fails += check("mojo_cpu");
	memset(Cm, 0x55, sizeof Cm); uint32_t c1 = mojo_tile_matmul_i8(A, B, Cm, M, K, N); fails += check("mojo_tile_v1");
	memset(Cm, 0x55, sizeof Cm); int32_t c2 = mojo_tile2_matmul_i8(A, B, Cm, M, K, N); fails += check("mojo_tile2");
	printf("MOJOLINK tile_v1_cycles=%lu tile2_busy=%ld\n", (unsigned long)c1, (long)c2);
	fails += c2 < 0;
	printf("MOJOLINK %s\n", fails ? "FAIL" : "PASS");
	sim_finish(fails ? 1 : 0);
}
