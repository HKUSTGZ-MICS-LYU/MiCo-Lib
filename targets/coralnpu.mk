CORALNPU_PATH = $(MICO_DIR)/targets/coralnpu
RVV_PATH = $(MICO_DIR)/targets/rvv

# Set this to any CoralNPU checkout. There is intentionally no repository-relative default.
CORALNPU_ROOT ?=
CORALNPU_BAZEL ?= bazel
CORALNPU_EXEC_ROOT ?= $(if $(strip $(CORALNPU_ROOT)),$(shell cd "$(CORALNPU_ROOT)" && "$(CORALNPU_BAZEL)" info execution_root 2>/dev/null),)
CORALNPU_EXEC_TOOLCHAIN = $(CORALNPU_EXEC_ROOT)/external/toolchain_coralnpu_v2
CORALNPU_CACHE_ROOT ?= $(HOME)/.cache/bazel/_bazel_$(USER)
CORALNPU_CACHED_TOOLCHAIN = $(firstword $(wildcard $(CORALNPU_CACHE_ROOT)/*/external/toolchain_coralnpu_v2))
CORALNPU_TOOLCHAIN ?= $(if $(wildcard $(CORALNPU_EXEC_TOOLCHAIN)/bin/riscv32-unknown-elf-gcc),$(CORALNPU_EXEC_TOOLCHAIN),$(CORALNPU_CACHED_TOOLCHAIN))
CORALNPU_CRT_DIR ?= $(if $(strip $(CORALNPU_ROOT)),$(CORALNPU_ROOT)/toolchain/crt,)
CORALNPU_LINKER ?= $(if $(strip $(CORALNPU_ROOT)),$(CORALNPU_ROOT)/toolchain/coralnpu_tcm.ld,)
CORALNPU_SIM ?= $(if $(strip $(CORALNPU_ROOT)),$(CORALNPU_ROOT)/bazel-bin/tests/verilator_sim/rvv_core_mini_axi_sim,)
CORALNPU_SIM_TARGET ?= //tests/verilator_sim:rvv_core_mini_axi_sim
CORALNPU_CYCLES ?= 100000000
CORALNPU_SIM_FLAGS ?=

# Set to 0 to use the scalar MiCo baseline for comparisons.
CORALNPU_RVV ?= 1
MABI ?= ilp32
ifeq ($(CORALNPU_RVV),1)
MARCH ?= rv32imf_zve32x_zicsr_zifencei_zbb
else
MARCH ?= rv32imf_zicsr_zifencei_zbb
endif

CC = $(CORALNPU_TOOLCHAIN)/bin/riscv32-unknown-elf-gcc
OBJDUMP = $(CORALNPU_TOOLCHAIN)/bin/riscv32-unknown-elf-objdump

CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
CFLAGS += -ffreestanding -fno-common -fno-builtin-printf
CFLAGS += -DRISCV_CORALNPU -DTEST_NUM=$(TEST_NUM)
ifeq ($(CORALNPU_RVV),1)
CFLAGS += -DMICO_RVV
endif
CFLAGS += -I$(RVV_PATH) -I$(CORALNPU_PATH)

LDFLAGS += -static --specs=nano.specs -lm -lc -lgcc -nostartfiles
LDFLAGS += -Wl,-T$(CORALNPU_LINKER)

RISCV_SOURCE = $(CORALNPU_PATH)/coralnpu_runtime.c
ifeq ($(CORALNPU_RVV),1)
RISCV_SOURCE += $(wildcard $(RVV_PATH)/*.c)
endif
RISCV_SOURCE += $(if $(strip $(CORALNPU_CRT_DIR)),$(CORALNPU_CRT_DIR)/crt.S $(CORALNPU_CRT_DIR)/coralnpu_start.S,)

.PHONY: coralnpu-check coralnpu-sim run-coralnpu

coralnpu-check:
	@test -n "$(strip $(CORALNPU_ROOT))" || { echo "Set CORALNPU_ROOT to the CoralNPU checkout directory."; exit 1; }
	@test -x "$(CC)" || { echo "Missing CoralNPU compiler: $(CC)"; echo "Set CORALNPU_TOOLCHAIN or build the CoralNPU Bazel workspace first."; exit 1; }
	@test -f "$(CORALNPU_LINKER)" || { echo "Missing CoralNPU linker script: $(CORALNPU_LINKER)"; exit 1; }
	@test -f "$(CORALNPU_CRT_DIR)/crt.S" -a -f "$(CORALNPU_CRT_DIR)/coralnpu_start.S" || { echo "Missing CoralNPU CRT sources under $(CORALNPU_CRT_DIR)"; exit 1; }

$(MAIN).elf: coralnpu-check

coralnpu-sim: coralnpu-check
	@cd "$(CORALNPU_ROOT)" && "$(CORALNPU_BAZEL)" build "$(CORALNPU_SIM_TARGET)"
	@test -x "$(CORALNPU_SIM)" || { echo "Missing CoralNPU simulator: $(CORALNPU_SIM)"; exit 1; }

run-coralnpu: coralnpu-sim $(MAIN).elf
	@"$(CORALNPU_SIM)" --binary "$(MAIN).elf" --cycles=$(CORALNPU_CYCLES) $(CORALNPU_SIM_FLAGS)
