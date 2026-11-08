/* Milestone 1: hello world on VexRiscv in the Verilator LiteX SoC. */
#include <stdio.h>
#include <stdint.h>
#include "sim.h"

int main(void)
{
	sim_init();
	uint64_t t0 = sim_cycles();
	printf("hello from VexRiscv (fpga/soc)\n");
	uint32_t acc = 0;
	for (uint32_t i = 1; i <= 1000; i++) acc = acc * 31u + i * i;
	uint64_t t1 = sim_cycles();
	printf("checksum 0x%08lx (expect 0x%08lx), %lu cycles\n",
	       (unsigned long)acc, 0xee85ed14UL, (unsigned long)(t1 - t0));
	sim_finish(0);
}
