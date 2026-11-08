#!/usr/bin/env python3
"""LiteX VexRiscv SoC for Verilator: ROM (BIOS) + SRAM + main RAM + sim UART,
a cycle counter / finish control block, and the int8 matmul tile as a
Wishbone slave.

Build once, run many: the main RAM image is loaded by $readmemh at simulation
start from gateware/<main_ram>.init, so swapping firmware does not rebuild the
Verilator model (see run.py).
"""
import argparse
import os
from pathlib import Path

from migen import *

from litex.build.generic_platform import Pins, Subsignal
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig
from litex.gen import LiteXModule
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.builder import Builder
from litex.soc.interconnect import wishbone
from litex.soc.interconnect.csr import CSRStorage, CSRStatus, CSR
from litex.soc.cores.clock import *  # noqa: F401  (CRG helpers)
from migen.genlib.io import CRG

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

_io = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    ("serial", 0,
        Subsignal("source_valid", Pins(1)),
        Subsignal("source_ready", Pins(1)),
        Subsignal("source_data",  Pins(8)),
        Subsignal("sink_valid",   Pins(1)),
        Subsignal("sink_ready",   Pins(1)),
        Subsignal("sink_data",    Pins(8)),
    ),
]

TILE_BASE  = 0x82000000   # v1 wrapper (rtl/tile_v1_wb.v), 4 KiB
TILE2_BASE = 0x90000000   # tile2 (fpga/tile2/INTERFACE.md), 128 KiB window


class SimCtrl(LiteXModule):
    """64-bit free-running cycle counter (latched on read of the low word via
    `latch`) and a `finish` register that ends the simulation ($finish)."""
    def __init__(self):
        self.finish = CSRStorage(8, name="finish")
        self.latch  = CSR(name="latch")
        self.cycles = CSRStatus(64, name="cycles")
        counter = Signal(64)
        self.sync += [
            counter.eq(counter + 1),
            If(self.latch.re, self.cycles.status.eq(counter)),
            If(self.finish.re, Finish()),
        ]


class TileV1(LiteXModule):
    def __init__(self, platform):
        self.bus = bus = wishbone.Interface(data_width=32, address_width=32, addressing="word")
        self.specials += Instance("tile_v1_wb",
            i_clk      = ClockSignal("sys"),
            i_rst      = ResetSignal("sys"),
            i_wb_adr   = bus.adr[:30],
            i_wb_dat_w = bus.dat_w,
            o_wb_dat_r = bus.dat_r,
            i_wb_sel   = bus.sel,
            i_wb_cyc   = bus.cyc,
            i_wb_stb   = bus.stb,
            o_wb_ack   = bus.ack,
            i_wb_we    = bus.we,
        )
        platform.add_source(str(HERE / "rtl" / "tile_v1_wb.v"))
        platform.add_source(str(REPO / "fpga" / "rtl" / "systolic_n4_i8_a32.v"))


class Tile2(LiteXModule):
    """fpga/tile2 Wishbone slave (contract v2.0, fpga/tile2/INTERFACE.md)."""
    def __init__(self, platform, n=8):
        self.bus = bus = wishbone.Interface(data_width=32, address_width=32, addressing="word")
        self.irq = Signal()
        self.specials += Instance(f"tile2_n{n}",
            i_clk      = ClockSignal("sys"),
            i_rst      = ResetSignal("sys"),
            i_wb_adr   = bus.adr[:15],
            i_wb_dat_w = bus.dat_w,
            o_wb_dat_r = bus.dat_r,
            i_wb_sel   = bus.sel,
            i_wb_cyc   = bus.cyc,
            i_wb_stb   = bus.stb,
            i_wb_we    = bus.we,
            o_wb_ack   = bus.ack,
            o_irq      = self.irq,
        )
        self.comb += bus.err.eq(0)
        platform.add_source(str(REPO / "fpga" / "tile2" / "rtl" / f"tile2_n{n}.v"))


class SoC(SoCCore):
    def __init__(self, cpu_variant="standard", main_ram_size=0x0100_0000, tile="v1,v2", **kw):
        platform = SimPlatform("SIM", _io)
        self.crg = CRG(platform.request("sys_clk"))
        # A non-empty init makes Migen emit $readmemh for main RAM; run.py
        # replaces the .init file with the real image before each run.
        SoCCore.__init__(self, platform, clk_freq=int(1e6),
            ident                    = "llming fpga/soc sim",
            cpu_type                 = "vexriscv",
            cpu_variant              = cpu_variant,
            uart_name                = "sim",
            integrated_rom_size      = 0x10000,
            integrated_sram_size     = 0x10000,
            integrated_main_ram_size = main_ram_size,
            integrated_main_ram_init = [0x0000006f],  # j . (overwritten at run time)
            **kw)
        self.add_constant("ROM_BOOT_ADDRESS", self.mem_map["main_ram"])
        self.add_config("BIOS_NO_DELAYS")
        self.add_constant("SERIAL_BOOT_DISABLE")
        self.add_config("BIOS_NO_CRC")
        self.simctrl = SimCtrl()
        tiles = [t for t in tile.split(",") if t and t != "none"]
        if "v2" in tiles:
            self.tile2 = Tile2(platform)
            self.bus.add_slave("tile2", self.tile2.bus,
                SoCRegion(origin=TILE2_BASE, size=0x20000, cached=False))
            self.add_constant("TILE2_BASE", TILE2_BASE)
        if "v1" in tiles:
            self.tile = TileV1(platform)
            self.bus.add_slave("tile", self.tile.bus,
                SoCRegion(origin=TILE_BASE, size=0x1000, cached=False))
            self.add_constant("TILE_BASE", TILE_BASE)
            self.add_constant("TILE_V1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(HERE / "build" / "sim"))
    p.add_argument("--cpu-variant", default="standard")
    p.add_argument("--main-ram-mb", type=int, default=16)
    p.add_argument("--tile", default="v1,v2", help="comma list of v1, v2 (or none)")
    p.add_argument("--libc", default="minimal", choices=["minimal", "full"],
                   help="LiteX picolibc build mode (DOOM needs full)")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args()
    soc = SoC(cpu_variant=args.cpu_variant, main_ram_size=args.main_ram_mb << 20, tile=args.tile)
    cfg = SimConfig(default_clk="sys_clk")
    cfg.add_module("serial2console", "serial")
    b = Builder(soc, output_dir=args.out, build_bundle=False, libc_mode=args.libc)
    b.build(sim_config=cfg, interactive=False, run=False, jobs=args.jobs, opt_level="O3")


if __name__ == "__main__":
    main()
