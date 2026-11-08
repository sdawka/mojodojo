#include <stdio.h>
#include <irq.h>
#include <libbase/uart.h>
#include <generated/csr.h>
#include "sim.h"

void sim_init(void)
{
#ifdef CONFIG_CPU_HAS_INTERRUPT
	irq_setmask(0);
	irq_setie(1);
#endif
	uart_init();
}

uint64_t sim_cycles(void)
{
	simctrl_latch_write(1);
	return simctrl_cycles_read();
}

void sim_finish(int code)
{
	printf("\nSIM_FINISH %d\n", code);
#ifdef CSR_UART_BASE
	/* drain the UART TX path before stopping the clock */
	while (!uart_txempty_read()) ;
	for (volatile int i = 0; i < 2000; i++) ;
#endif
	simctrl_finish_write(code & 0xff);
	for (;;) ;
}
