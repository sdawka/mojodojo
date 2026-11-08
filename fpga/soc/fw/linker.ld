/* Firmware image linked entirely into main RAM (text, data, bss, heap, stack).
   The BIOS jumps to ORIGIN(main_ram) (ROM_BOOT_ADDRESS). LMA == VMA, so the
   crt0 data copy is a no-op. */
INCLUDE generated/output_format.ld
ENTRY(_start)
__DYNAMIC = 0;
INCLUDE generated/regions.ld

SECTIONS
{
	.text : {
		_ftext = .;
		KEEP(*crt0*(.text))
		KEEP(*(.text.isr))
		*(.text .stub .text.* .gnu.linkonce.t.*)
		_etext = .;
	} > main_ram
	.rodata : {
		. = ALIGN(8);
		_frodata = .;
		*(.rodata .rodata.* .gnu.linkonce.r.*)
		*(.rodata1)
		*(.srodata .srodata.*)
		. = ALIGN(8);
		_erodata = .;
	} > main_ram
	.init_array : {
		. = ALIGN(8);
		PROVIDE_HIDDEN (__init_array_start = .);
		KEEP(*(SORT(.init_array.*)))
		KEEP(*(.init_array))
		PROVIDE_HIDDEN (__init_array_end = .);
	} > main_ram
	.data : {
		. = ALIGN(8);
		_fdata = .;
		*(.data .data.* .gnu.linkonce.d.*)
		*(.data1)
		_gp = ALIGN(16);
		*(.sdata .sdata.* .gnu.linkonce.s.*)
		. = ALIGN(8);
		_edata = .;
	} > main_ram
	.bss (NOLOAD) : {
		. = ALIGN(8);
		_fbss = .;
		*(.dynsbss)
		*(.sbss .sbss.* .gnu.linkonce.sb.*)
		*(.scommon)
		*(.dynbss)
		*(.bss .bss.* .gnu.linkonce.b.*)
		*(COMMON)
		. = ALIGN(8);
		_ebss = .;
		_end = .;
		_heap_start = .;
	} > main_ram
}

PROVIDE(_fstack = ORIGIN(main_ram) + LENGTH(main_ram) - 16);
PROVIDE(_heap_end = ORIGIN(main_ram) + LENGTH(main_ram) - 0x40000);
PROVIDE(__heap_start = _heap_start);
PROVIDE(__heap_end = _heap_end);
PROVIDE(_fdata_rom = LOADADDR(.data));
PROVIDE(_edata_rom = LOADADDR(.data) + SIZEOF(.data));
