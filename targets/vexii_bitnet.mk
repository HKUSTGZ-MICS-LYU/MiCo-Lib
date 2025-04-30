VEXII_PATH = $(MICO_DIR)/targets/vexii_bitnet
VEXII_LD = $(MICO_DIR)/targets/vexii_bitnet/vexii.ld

MABI?=ilp32
MARCH?=rv32imc

LARGE_RAM?=
HEAP_SIZE=4096*1024

BITNET_QUANT ?= 3
USE_SIMD ?= 32

ifeq ($(LARGE_RAM), 1)
	VEXII_LD = $(MICO_DIR)/targets/vexii_bitnet/vexii_64mb.ld
	HEAP_SIZE=32*1024*1024
endif

ifeq ($(TARGET), vexii_bitnet)
	CC = $(RISCV_PREFIX)-gcc
	OBJDUMP = $(RISCV_PREFIX)-objdump

	CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
	CFLAGS += -DRISCV_VEXII -DTEST_NUM=$(TEST_NUM) -DMAX_HEAP_SIZE=$(HEAP_SIZE)
	CFLAGS += -DBITNET_QUANT=$(BITNET_QUANT) -DUSE_SIMD=$(USE_SIMD)
	CFLAGS += -fno-common -fno-inline
	CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
	CFLAGS += -I${VEXII_PATH}/ -I${VEXII_PATH}/driver

	LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
	LDFLAGS += -nostdlib -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(MAIN).map,--print-memory-usage
	LDFLAGS += -L./ -nolibc -lm -lc

	ifeq ($(MARCH), rv32imc)
		LDFLAGS += -L$(MICO_DIR)/lib/ -lrvfp
	else
		LDFLAGS += -lgcc
	endif
	RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)
endif