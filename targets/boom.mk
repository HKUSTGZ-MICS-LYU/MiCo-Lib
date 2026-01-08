BOOM_PATH = $(MICO_DIR)/targets/boom

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump
CFLAGS += -DUSE_CHIPYARD -DTEST_NUM=$(TEST_NUM)
CFLAGS += -fno-common -fno-builtin-printf -specs=htif_nano.specs
LDFLAGS += -static

MABI?=lp64d
MARCH?=rv64imafdc_zicntr_zihpm

CFLAGS += -march=$(MARCH) -mabi=$(MABI)

RISCV_SOURCE = $(wildcard $(BOOM_PATH)/*.c) $(wildcard $(BOOM_PATH)/*.S)