RVV_PATH = $(MICO_DIR)/targets/rvv

SPIKE ?= spike
SPIKE_ISA ?= rv64gcv_zicntr_zihpm
SPIKE_FLAGS ?=

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump

CFLAGS += -DUSE_CHIPYARD -DMICO_RVV -DTEST_NUM=$(TEST_NUM)
CFLAGS += -fno-common -fno-builtin-printf -specs=htif_nano.specs
LDFLAGS += -static

# Keep RVV Spike tests compatible with larger generated tensors.
HTIF_HEAP_SIZE ?= 0x100000
LDFLAGS += -Wl,--defsym=__heap_size=$(HTIF_HEAP_SIZE)

MABI ?= lp64d
MARCH ?= rv64gcv_zicntr_zihpm
CFLAGS += -march=$(MARCH) -mabi=$(MABI)

CFLAGS += -I$(RVV_PATH)
RISCV_SOURCE = $(wildcard $(RVV_PATH)/*.c) $(wildcard $(RVV_PATH)/*.S)

run-rvv: $(MAIN).elf
	$(SPIKE) --isa=$(SPIKE_ISA) $(SPIKE_FLAGS) $<
