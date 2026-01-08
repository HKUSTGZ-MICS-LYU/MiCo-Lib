GEMMINI_PATH = $(MICO_DIR)/targets/gemmini

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump
CFLAGS += -DUSE_CHIPYARD -DTEST_NUM=$(TEST_NUM)
CFLAGS += -fno-common -fno-builtin-printf -specs=htif_nano.specs
LDFLAGS += -static

MABI?=lp64d
MARCH?=rv64imafdc_zicntr_zihpm

CFLAGS += -march=$(MARCH) -mabi=$(MABI)
CFLAGS += -I$(GEMMINI_PATH)/

RISCV_SOURCE = $(wildcard $(GEMMINI_PATH)/*.c) $(wildcard $(GEMMINI_PATH)/*.S)