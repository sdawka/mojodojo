# ECP5 board choice for the int8 tile + soft RISC-V (+ DOOM)

Research date: 2026-11-08. Web research only; nothing was bought or downloaded. Prices and stock are snapshots from that date, and every figure has a source link. **[unverified]** marks claims that could not be checked against a vendor, datasheet or source page. **[search snippet]** marks figures that came from a search-engine summary of a page I could not open directly (Crowd Supply and GroupGets blocked or rate-limited the fetcher).

## 1. What the board has to do

From [README.md](README.md) (ECP5 synthesis table) and [DESIGN.md](DESIGN.md):

| Need | Number | Consequence |
| --- | --- | --- |
| N=8 tile with DSP multipliers | 64 MULT18X18D, about 2.3k LUT4, 3.4k FF; 67.5 MHz on a 45F (DSP 64/72) | Needs a **45F or 85F**. The 12F and 25F have only 28 multipliers. |
| N=16 tile | 256 MULT18X18D (does not fit any ECP5), or 68k LUT4 at 96% of an 85F with `-nodsp` | N=16 is out of reach next to a CPU on any ECP5 |
| VexRiscv/LiteX SoC next to the tile | A few thousand LUTs, plus some DSPs if the CPU has a hardware multiplier **[unverified: exact MULT18X18D count for VexRiscv's MulPlugin on ECP5]** | On a 45F the tile plus the CPU leaves almost no DSP headroom (64 of 72 used by the tile alone). An **85F (156 DSP)** leaves room for the CPU, a v2 registered-product DSP mapping, and a second tile. |
| DOOM | Shareware WAD about 4 MB ([LiteX notes](https://enjoy-digital.github.io/posts/chromatix-doom/)), plus heap and framebuffers. Hazard3-Doom's 32 MiB profile reserves a 12 MiB heap and a 12 MiB IWAD region ([Hazard3-Doom README](https://github.com/ulx3s/Hazard3-Doom)). The ChromatiX port fits in 7.5 MB of PSRAM. | You want **32 MB or more of external DRAM**. 8 MB is a proven but tight floor. |
| Fully open flow from macOS | Yosys + nextpnr-ecp5 + openFPGALoader. openFPGALoader installs with `brew install openfpgaloader` ([install guide](https://github.com/trabucayre/openFPGALoader/blob/master/doc/guide/install.rst)). | Prefer boards with a built-in USB programmer that openFPGALoader knows ([board.hpp](https://github.com/trabucayre/openFPGALoader/blob/master/src/board.hpp)). |

ECP5 device resources, from Lattice's family table ([ECP5 product page](https://www.latticesemi.com/Products/FPGAandCPLD/ECP5)):

| Device | LUTs | 18 Kb EBR blocks (Kbit) | Dist. RAM Kbit | 18x18 mult | SERDES ch |
| --- | --- | --- | --- | --- | --- |
| LFE5U-12 | 12k | 32 (576) | 97 | 28 | 0 |
| LFE5U-25 / UM(5G)-25 | 24k | 56 (1008) | 194 | 28 | 0 / 2 |
| LFE5U-45 / UM(5G)-45 | 44k | 108 (1944) | 351 | 72 | 0 / 4 |
| LFE5U-85 / UM(5G)-85 | 84k | 208 (3744) | 669 | 156 | 0 / 4 |

## 2. ECP5 boards

LiteX target names are the file names in [litex-boards/targets](https://github.com/litex-hub/litex-boards/tree/master/litex_boards/targets). "Linux" means the board is in the `./make.py --help` list of [linux-on-litex-vexriscv](https://github.com/litex-hub/linux-on-litex-vexriscv). That project requires at least 32 MB of RAM, or 8 MB when the rootfs is on SD or NFS (same README).

| Board | FPGA (DSP) | External RAM | Video out | Programming / macOS | LiteX target; Linux | Price / where (2026-11-08) | DOOM / Linux evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ULX3S 85F** (Radiona / Intergalaktik), v3.1.8 | LFE5U-85F-6BG381C (156) | 32 MB SDR SDRAM, 166 MHz ([Lectronz listing](https://lectronz.com/products/ulx3s-85f-v3-1-8), [radiona.org](https://radiona.org/ulx3s/)). The ULX3S repository lists 8/16/32/64 MB build options ([emard/ulx3s](https://github.com/emard/ulx3s)). Hazard3-Doom uses a 64 MiB profile on its 85F, so some 85F batches have 64 MB. | GPDI, a DVI/HDMI-compatible differential connector. It is "mostly TMDS tolerant", not licensed HDMI ([emard/ulx3s](https://github.com/emard/ulx3s)). LiteX has `--with-video-terminal/framebuffer`. | FTDI FT231XS: bit-banged JTAG (about 500 kbit/s) plus USB serial. Supported by openFPGALoader (`-b ulx3s`, cable ft231X); DFU and ESP32 variants also exist ([board.hpp](https://github.com/trabucayre/openFPGALoader/blob/master/src/board.hpp)). | `radiona_ulx3s` (`--device LFE5U-85F`, `--sdram-module MT48LC16M16` default, `AS4C32M16` for 64 MB); Linux: yes (`ulx3s`) | **Out of stock** at Lectronz ($243.40) and Envox ([€210](https://www.envox.eu/product/ulx3s/)). Crowd Supply page not fetchable **[unverified]**. | **DOOM runs**: [ulx3s/Hazard3-Doom](https://github.com/ulx3s/Hazard3-Doom) (DoomGeneric on a Hazard3 RV32 core, HDMI 320x200 indexed, 50 MHz on 85F, last push 2026-11-04). It is listed on [ulx3s.github.io](https://ulx3s.github.io/). This is **not LiteX/VexRiscv**. |
| ULX3S 12F | LFE5U-12F (28) | 32 MB SDRAM ([Lectronz 12F](https://lectronz.com/products/ulx3s-12f-v3-1-8)) | GPDI | as above | `radiona_ulx3s --device LFE5U-12F` | **In stock**: $149.61, 10 units (Lectronz) | Hazard3-Doom also runs on the 12F (32 MiB profile, 40 MHz). **Too few DSPs for the N=8 tile.** |
| OrangeCrab 25F/85F (Good Stuff Dept.) | LFE5U-25F/85F-8MG285C (28/156) ([platform](https://github.com/litex-hub/litex-boards/blob/master/litex_boards/platforms/gsd_orangecrab.py)) | DDR3L. LiteX default `MT41K64M16` (128 MB); the target also accepts 128M16/256M16/512M16. Farnell's 85F listing title says 512 MiB ([element14](https://nz.element14.com/groupgets-llc/orangecrab-r0d2-85/dev-board-ecp5-85f-512-mib-fpga/dp/3651572)) **[search snippet]**. | None (Feather form factor) | USB DFU bootloader; openFPGALoader `-b orangeCrab` (DFU 1209:5af0). Native USB, so there is no FTDI to drive. | `gsd_orangecrab`; Linux: yes (`orange_crab`) | **Discontinued**: [1BitSquared](https://1bitsquared.com/products/orangecrab) page titled "(Discontinued)". DigiKey marks ORANGECRAB-R0D2-85 obsolete at about $158.75; one broker claims stock at $172.66 **[search snippet]**. | Linux-on-LiteX supported. No DOOM port found. |
| Colorlight i5 / i9 (SODIMM modules + extension board) | i5 v7.0: LFE5U-25F (28); i9 v7.2: LFE5U-45F (72) ([platform](https://github.com/litex-hub/litex-boards/blob/master/litex_boards/platforms/colorlight_i5.py)) | 8 MB SDR SDRAM (LiteX uses `EM638325`, 2M x 32) ([target](https://github.com/litex-hub/litex-boards/blob/master/litex_boards/targets/colorlight_i5.py)) | HDMI connector on the extension board. The LiteX target has `--with-video-terminal/framebuffer`. | The extension board's CMSIS-DAP (DAPLink) programmer; openFPGALoader `-b colorlight-i5` / `colorlight-i9` (cable cmsisdap) | `colorlight_i5 --board i9`; Linux: yes (`colorlight_i5`, SD card Pmod) | About **$24** per module on [AliExpress](https://www.aliexpress.com/item/1005007788475037.html), with an i9 + extension-board bundle option **[search snippet]**. Amazon has i9 + ext bundles. | Linux-on-LiteX supported. No DOOM port found. 8 MB matches the ChromatiX DOOM's 7.5 MB budget. |
| Colorlight i9**plus** | **Artix-7 XC7A50T**, not ECP5 (`colorlight-i9+`, xc7a50tfgg484, in [board.hpp](https://github.com/trabucayre/openFPGALoader/blob/master/src/board.hpp)) | 8 MB SDRAM (`M12L64322A`) | | | `colorlight_i9plus` | | Don't confuse with the i9 |
| **ButterStick r1.0** (Good Stuff Dept.) | LFE5UM5G-25F/85F-8BG381C (28/156) | 2 Gbit (256 MB) or 8 Gbit (1 GB) DDR3L, as 2x MT41K64M16 or 2x MT41K256M16 ([butterstick-hardware](https://github.com/butterstick-fpga/butterstick-hardware)) | None on board (3x SYZYGY, GigE, HS USB) | USB DFU bootloader ([butterstick-bootloader](https://github.com/butterstick-fpga/butterstick-bootloader)) plus a 0.1" JTAG header. Not a named openFPGALoader board **[unverified: generic DFU mode works]**. | `gsd_butterstick`; Linux: yes (`butter_stick`) | GroupGets store (README link). The 2021 group buy was $179.99 **[search snippet]**. Current price and stock **[unverified]** (GroupGets returned HTTP 429). | Linux-on-LiteX supported. No DOOM port found. |
| **LambdaConcept ECPIX-5** | LFE5UM5G-45F / 85F (72/156) | 512 MB DDR3L ([shop](https://shop.lambdaconcept.com/home/46-ecpix-5.html)) | HDMI 1.4b transmitter; LiteX `--with-video-terminal/framebuffer` | Built-in JTAG + serial over USB-C; openFPGALoader `-b ecpix5` (cable ecpix5-debug) / `ecpix5_r03` (ft4232) | `lambdaconcept_ecpix5` (default 85F); Linux: yes (`ecpix5`) | **€151 (ex VAT), out of stock** (shop) | Linux-on-LiteX supported. This is Enjoy-Digital's ECP5 bench board ([2026-11-03 post](https://enjoy-digital.github.io/posts/litex-agents-bench/)). No DOOM port found. |
| Lattice ECP5-5G Eval (LFE5UM5G-85F-EVN) | LFE5UM5G-85F-8BG381 (156) | **None** (the LiteX target has no DRAM) | None | On-board FT2232H; openFPGALoader `-b ecp5_evn` | `lattice_ecp5_evn` (`lattice_versa_ecp5` is the older Versa board with DDR3) | **In stock**: DigiKey 12 at $158.75, Mouser about $165, Newark $101.99 **[search snippet]** | No DRAM, so **no DOOM** without a hand-built memory add-on |
| iCESugar-Pro (Muse Lab) | LFE5U-25F-6BG256C (28) | 32 MB SDRAM (IS42S16160B) ([README](https://github.com/wuxx/icesugar-pro)) | HDMI (LiteX `gpdi`) | iCELink (DAPLink): drag-and-drop or `icesprog` | `muselab_icesugar_pro`; Linux: yes | About $67 on Amazon, May 2026 **[search snippet]** | Too few DSPs for N=8 |
| Icepi Zero | LFE5U-25F (45F option in the platform file) (28) | 32 MB SDRAM (W9825G6KH6) ([target](https://github.com/litex-hub/litex-boards/blob/master/litex_boards/targets/icepi_zero.py)) | mini GPDI | FT231X; openFPGALoader `-b icepi-zero` | `icepi_zero`; Linux: yes | Crowd Supply campaign from $69 (2025) **[search snippet]**; current stock **[unverified]** | Too few DSPs for N=8 |
| ULX4M-LD v2 (Radiona, CM4 form factor) | LFE5UM5G-85F (156) | DDR3. LiteX accepts MT41K64M16 to MT41K512M16; Hazard3-Doom qualifies an MT41K512M16HA part. | GPDI (on the carrier) | DFU (1d50:614b) or JTAG | `radiona_ulx4m_ld_v2`; Linux: yes (`ulx4m_ld_v2`) | No shop listing found **[unverified]** | **DOOM runs** (Hazard3-Doom, 40 MHz, LiteDRAM). Needs a CM4 carrier. |

Other ECP5 targets in litex-boards that I did not price: `logicbone`, `trellisboard`, `lattice_versa_ecp5`, `colorlight_5a_75x`, `hackaday_hadbadge`, `machdyne_*` (several), `intergalaktik_ulx5m_gs`.

## 3. Non-ECP5 sanity check

| Board | Why not |
| --- | --- |
| iCEBreaker (iCE40UP5K) | 5,280 LUTs, 8 MAC16 DSPs, 120 Kbit EBR, 1 Mbit SPRAM ([Lattice iCE40 UltraPlus](https://www.latticesemi.com/Products/FPGAandCPLD/iCE40UltraPlus)). There aren't enough DSPs or LUTs for an N=8 tile plus a CPU. DOOM on it is a heroic, single-purpose build (below). |
| Sipeed Tang Nano 20K (Gowin GW2AR-18) | About 20.7k LUT4, 48 multipliers, 8 MB in-package SDRAM, HDMI, about $25 at launch ([Sipeed wiki](https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html), [CNX](https://www.cnx-software.com/2023/05/22/25-sipeed-tang-nano-20k-fpga-board-can-simulate-a-risc-v-core-run-linux-retro-games/)). Cheap, but it has fewer DSPs than a 45F and the Gowin open flow (Apicula) is less mature. |
| Sipeed Tang Primer 20K (Gowin GW2A-18) | It has 128 MB DDR3, but the LiteX target asserts `not (toolchain == "apicula" and with_dram)` ([target](https://github.com/litex-hub/litex-boards/blob/master/litex_boards/targets/sipeed_tang_primer_20k.py)). **DDR3 needs the vendor Gowin IDE**, so it is not fully open. |
| Digilent Arty A7-35T/100T (Artix-7) | The LiteX target notes that with `--toolchain=yosys+nextpnr`, "DDR3 should be disabled" and the clock lowered ([target](https://github.com/litex-hub/litex-boards/blob/master/litex_boards/targets/digilent_arty.py)). The fully open Xilinx flow therefore **can't use the 256 MB DDR3 in LiteX today**, so DOOM would need Vivado (no macOS build). |

ECP5 is the family where the open flow drives DDR3/SDRAM, video and DSPs on macOS without vendor tools. That is the main reason to stay on ECP5.

## 4. From-memory claims, checked

1. **"ULX3S has 32 MB SDRAM and GPDI/HDMI out."** **Mostly true, with two caveats.**
   - Current v3.1.8 listings say "32MB SDRAM 166 MHz" and "GPDI General-Purpose Differential Interface" ([Lectronz 85F](https://lectronz.com/products/ulx3s-85f-v3-1-8), [radiona.org](https://radiona.org/ulx3s/)).
   - SDRAM size is a build option (8/16/32/64 MB per [emard/ulx3s](https://github.com/emard/ulx3s)), and Hazard3-Doom assumes 64 MiB on the 85F.
   - GPDI is DVI/HDMI-compatible electrically ("mostly TMDS tolerant"), but it is not a licensed HDMI port.
2. **"smunaut ran DOOM on an iCE40 UP5K with a RISC-V core."** **True.**
   - [smunaut/ice40-playground `projects/riscv_doom`](https://github.com/smunaut/ice40-playground/tree/master/projects/riscv_doom) has README title "RISC-V Doom for iCE40UP5k". Its Makefile sets `DEVICE = up5k`, `BOARD ?= icebreaker`.
   - `rtl/` contains `VexRiscv.v`. So the core is **VexRiscv** (not LiteX).
   - It uses a `qpi_memctrl` QSPI memory controller with two chip selects and a 12-bit parallel HDMI Pmod output (`rtl/top.v`).
   - The Doom source port is [smunaut/doom_riscv](https://github.com/smunaut/doom_riscv). Coverage: [Hackaday, 2021-02-07](https://hackaday.com/2021/02/07/ice40-runs-doom/).
   - Exact PSRAM part **[unverified]**. Press says "iCEBreaker with added SPI RAM".
3. **"LiteX VexRiscv has run DOOM on ECP5 boards."** **Not verified.**
   - I found no LiteX + VexRiscv DOOM on any ECP5 board.
   - What does exist:
     - LiteX + VexRiscv + DoomGeneric on a **Gowin GW5A-25** (ModRetro Chromatic). It runs at 67 MHz from 7.5 MB PSRAM at 31 fps ([LiteX notes, 2026-10-05](https://enjoy-digital.github.io/posts/chromatix-doom/)).
     - DOOM on **ECP5** with a **Hazard3** core and a custom (non-LiteX) SoC on ULX3S/ULX4M ([Hazard3-Doom](https://github.com/ulx3s/Hazard3-Doom)).
     - smunaut's VexRiscv DOOM on iCE40.
   - The LiteX/VexRiscv-on-ECP5 combination is very likely to work (same core and SoC as Chromatic, with more RAM), but it has not been demonstrated in any source I found.

## 5. Recommendation

**First choice: ULX3S 85F** (LiteX `radiona_ulx3s --device LFE5U-85F`).
- 156 DSPs: room for the N=8 tile (64), the CPU, and v2 experiments.
- 32 MB (or 64 MB) SDRAM.
- GPDI video.
- Single-USB FT231X programming that openFPGALoader supports on macOS.
- The only ECP5 board with a maintained, current DOOM port (Hazard3-Doom, updated 2026-11-04).
- Supported by Linux-on-LiteX-VexRiscv.
- Caveat: **the 85F is out of stock at both shops I could check** (Lectronz, Envox). Check Crowd Supply and the Lectronz restock notice before falling back. The in-stock ULX3S 12F is **not** a substitute (28 DSPs).

**If no ULX3S 85F is available**, buy an **ECPIX-5 85F** when it is back in stock (€151, 512 MB DDR3, HDMI, Enjoy-Digital's own reference board). Otherwise buy remaining **OrangeCrab 85F** stock (DDR3, DFU, no video: dump frames over USB/UART). The Lattice ECP5-5G EVN is in stock but has no DRAM, so it only serves the tile goal, not DOOM.

**Budget alternative: Colorlight i9 (LFE5U-45F) + extension board**, about $25 to $60 total.
- 72 DSPs fit the N=8 tile (64/72 measured). Give the CPU a LUT multiplier or accept a very tight DSP budget.
- 8 MB SDRAM: DOOM is feasible at the ChromatiX budget, but with no slack.
- The extension board provides HDMI and the CMSIS-DAP programmer.
- Supported by `colorlight_i5 --board i9` and Linux-on-LiteX.
- Buy the **i9**, not the i5 (25F, 28 DSPs) and not the "i9plus" (Artix-7).

## 6. What to buy (macOS)

- ULX3S **85F** (v3.1.8). Confirm the SDRAM size on the listing.
- USB cable for the board's US1 port (FT231X). On v3.1.x this is micro-USB **[unverified on v3.1.8: connector type]**; get a USB-C/micro-B cable or adapter to match.
- Second USB cable if you want the US2 (FPGA-native USB) port for HID keyboard or data. Optional.
- microSD card (WAD and Linux rootfs).
- HDMI cable plus a DVI/HDMI monitor input for GPDI. Optional: a USB HDMI capture dongle so the Mac can see the output.
- Software (no purchase): `brew install openfpgaloader`; OSS CAD Suite or the existing pixi Yosys/nextpnr; an xPack RISC-V GCC (Hazard3-Doom notes that the bare-metal GCC must be installed separately on macOS).
- No separate JTAG probe is needed. The FT231X on the board is the programmer. Budget fallback: Colorlight i9 + extension board needs only a USB cable, because DAPLink is on the extension board.
