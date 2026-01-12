CHIPYARD_DIR ?=/home/jzj/work/chipyard
GEMMINI_PATH = $(MICO_DIR)/targets/gemmini

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump
CFLAGS += -DUSE_CHIPYARD -DTEST_NUM=$(TEST_NUM) -DBAREMETAL -DUSE_ALT_LAYOUT -DUSE_GEMMINI
CFLAGS += -fno-common -fno-builtin-printf -specs=htif_nano.specs
LDFLAGS += -static

MABI?=lp64d
MARCH?=rv64imafdc_zicntr_zihpm

CFLAGS += -march=$(MARCH) -mabi=$(MABI)

GEMMINI_INCLUDES = $(CHIPYARD_DIR)/generators/gemmini/software/gemmini-rocc-tests/
CFLAGS += -I$(GEMMINI_INCLUDES) -I$(GEMMINI_INCLUDES)/include

RISCV_SOURCE = $(wildcard $(GEMMINI_PATH)/*.c) $(wildcard $(GEMMINI_PATH)/*.S)