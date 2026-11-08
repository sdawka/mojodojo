/* Firmware-side helpers for the fpga/soc simulation SoC. */
#ifndef FW_SIM_H
#define FW_SIM_H
#include <stdint.h>

void     sim_init(void);                 /* uart + irq setup after BIOS handoff */
uint64_t sim_cycles(void);               /* SoC cycle counter (sys_clk) */
void     sim_finish(int code) __attribute__((noreturn));  /* prints marker, $finish */
#endif
