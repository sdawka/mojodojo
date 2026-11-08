# Shared firmware rules. Usage from fpga/soc:
#   make -f fw/fw.mk APP=hello SOC_BUILD=build/sim
# Produces $(OUT)/$(APP).elf / .bin. Sources: fw/$(APP)/*.c + fw/sim.c + crt0.
SOC_BUILD ?= build/sim
APP       ?= hello
FW_DIR    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(abspath $(SOC_BUILD))
OUT       ?= $(BUILD_DIR)/fw/$(APP)

include $(BUILD_DIR)/software/include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak

APP_SRCS  ?= $(wildcard $(FW_DIR)/$(APP)/*.c)
EXTRA_OBJS ?=
APP_COMMON ?=
OBJS := $(OUT)/crt0.o $(patsubst %.c,$(OUT)/common_%.o,sim.c $(APP_COMMON)) $(patsubst $(FW_DIR)/$(APP)/%.c,$(OUT)/%.o,$(APP_SRCS))
CFLAGS += -O2 -I$(FW_DIR) $(APP_CFLAGS)

all: $(OUT)/$(APP).bin

$(OUT):
	mkdir -p $@

$(OUT)/crt0.o: $(CPU_DIRECTORY)/crt0.S | $(OUT)
	$(assemble)
$(OUT)/common_%.o: $(FW_DIR)/%.c | $(OUT)
	$(compile)
$(OUT)/%.o: $(FW_DIR)/$(APP)/%.c | $(OUT)
	$(compile)

$(OUT)/$(APP).elf: $(OBJS) $(EXTRA_OBJS) $(FW_DIR)/linker.ld
	$(CC) $(LDFLAGS) -T $(FW_DIR)/linker.ld -N -o $@ $(OBJS) $(EXTRA_OBJS) \
		$(addprefix -L$(BUILD_DIR)/software/,libc libcompiler_rt libbase) \
		-Wl,--start-group -lc -lcompiler_rt -lbase $(APP_LIBS) -Wl,--end-group \
		-Wl,--gc-sections -Wl,-Map,$@.map

$(OUT)/$(APP).bin: $(OUT)/$(APP).elf
	$(OBJCOPY) -O binary $< $@

-include $(OBJS:.o=.d)
.PHONY: all
