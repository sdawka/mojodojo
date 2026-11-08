# Build DOOM (doomgeneric, GPL-2.0+) for the fpga/soc VexRiscv SoC.
#   pixi run make -f fw/doom/doom.mk SOC_BUILD=build/doom [MAX_FRAME=1 DUMP_FRAME=1 HASH_EVERY=1]
# Engine sources are compiled read-only from doom/third_party/doomgeneric
# (pinned by doom/scripts/fetch.sh); nothing upstream is modified.
SOC_BUILD ?= build/doom
FW_DIR    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BUILD_DIR := $(abspath $(SOC_BUILD))
DG        ?= $(abspath $(FW_DIR)/../../../doom/third_party/doomgeneric/doomgeneric)
OUT       ?= $(BUILD_DIR)/fw/doom
MAX_FRAME  ?= 1
DUMP_FRAME ?= 1
HASH_EVERY ?= 1
WAD_HDR_ADDR ?= 0x42000000
HEAP_END     ?= 0x41ff0000

include $(BUILD_DIR)/software/include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak

# Same object list as doom/scripts/build_c.sh (engine minus platform backend), plus r_draw.
CORE = dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net \
  f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound i_system \
  i_timer memio m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng \
  p_doors p_enemy p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup \
  p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_main r_plane r_segs r_sky r_things \
  sha1 sounds statdump st_lib st_stuff s_sound tables v_video wi_stuff w_checksum w_file w_main \
  w_wad z_zone w_file_stdc i_input i_video doomgeneric r_draw

# Host build flags (doom/scripts/build_c.sh) + -fsigned-char: the host is
# arm64 macOS (signed char); RISC-V char is unsigned by default.
DOOM_FLAGS = -O2 -g0 -w -fsigned-char -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE \
  -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 -I$(DG)
ENGINE_CFLAGS = $(filter-out -Os -fno-builtin -Wall,$(COMMONFLAGS)) $(DOOM_FLAGS)
PLAT_CFLAGS = $(filter-out -Os -fno-builtin,$(COMMONFLAGS)) -O2 -fsigned-char -I$(DG) -I$(FW_DIR) \
  -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 -DWAD_HDR_ADDR=$(WAD_HDR_ADDR) \
  -DDOOM_MAX_FRAME=$(MAX_FRAME) -DDOOM_DUMP_FRAME=$(DUMP_FRAME) -DDOOM_HASH_EVERY=$(HASH_EVERY)

ENGINE_OBJS = $(CORE:%=$(OUT)/engine/%.o)
PLAT_OBJS = $(OUT)/crt0.o $(OUT)/sim.o $(OUT)/dg_soc.o $(OUT)/sha256.o

all: $(OUT)/doom.bin

$(OUT)/engine:
	mkdir -p $@

$(OUT)/engine/%.o: $(DG)/%.c | $(OUT)/engine
	$(CC) -c $(ENGINE_CFLAGS) $< -o $@

$(OUT)/libdoomengine.a: $(ENGINE_OBJS)
	@rm -f $@; echo " AR       $@"; $(AR_normal) crs $@ $^

$(OUT)/crt0.o: $(CPU_DIRECTORY)/crt0.S | $(OUT)/engine
	$(CC) -c $(CFLAGS) -o $@ $<
$(OUT)/sim.o: $(FW_DIR)/sim.c | $(OUT)/engine
	$(CC) -c $(PLAT_CFLAGS) $< -o $@
# Rebuild the platform objects whenever the frame/address parameters change.
PARAMS := MAX_FRAME=$(MAX_FRAME) DUMP_FRAME=$(DUMP_FRAME) HASH_EVERY=$(HASH_EVERY) WAD_HDR_ADDR=$(WAD_HDR_ADDR) HEAP_END=$(HEAP_END)
$(OUT)/params.stamp: FORCE | $(OUT)/engine
	@echo '$(PARAMS)' | cmp -s - $@ || echo '$(PARAMS)' > $@
$(OUT)/%.o: $(FW_DIR)/doom/%.c $(MAKEFILE_LIST) $(OUT)/params.stamp | $(OUT)/engine
	$(CC) -c $(PLAT_CFLAGS) $< -o $@

$(OUT)/doom.elf: $(PLAT_OBJS) $(OUT)/libdoomengine.a $(FW_DIR)/linker.ld $(OUT)/params.stamp
	$(CC) $(LDFLAGS) -T $(FW_DIR)/linker.ld -N -o $@ $(PLAT_OBJS) $(OUT)/libdoomengine.a \
		$(addprefix -L$(BUILD_DIR)/software/,libc libcompiler_rt libbase) \
		-Wl,--start-group -lc -lcompiler_rt -lbase -Wl,--end-group \
		-Wl,--defsym=_heap_end=$(HEAP_END) -Wl,--defsym=__heap_end=$(HEAP_END) \
		-Wl,--gc-sections -Wl,-Map,$@.map

$(OUT)/doom.bin: $(OUT)/doom.elf
	$(OBJCOPY) -O binary $< $@

.PHONY: all FORCE
FORCE:
