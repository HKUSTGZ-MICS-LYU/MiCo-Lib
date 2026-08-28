SPIKE_PATH = $(MICO_DIR)/targets/spike

SPIKE = spike

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump
CFLAGS += -DUSE_CHIPYARD -DTEST_NUM=$(TEST_NUM)
CFLAGS += -fno-common -fno-builtin-printf -specs=htif_nano.specs
LDFLAGS += -static 

# htif_nano.ld defaults to a 128K heap, which is too small for larger tests.
HTIF_HEAP_SIZE ?= 0x100000
LDFLAGS += -Wl,--defsym=__heap_size=$(HTIF_HEAP_SIZE)

MABI?=lp64d
MARCH?=rv64imafdc_zicntr_zihpm

CFLAGS += -march=$(MARCH) -mabi=$(MABI)

RISCV_SOURCE = $(wildcard $(SPIKE_PATH)/*.c) $(wildcard $(SPIKE_PATH)/*.S)

run-spike: $(MAIN).elf
	$(SPIKE) $<