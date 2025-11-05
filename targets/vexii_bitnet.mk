VEXII_BN_PATH = $(MICO_DIR)/targets/vexii_bitnet
VEXII_BN_LD = $(MICO_DIR)/targets/vexii_bitnet/vexii.ld

MABI?=ilp32
MARCH?=rv32imc
ifeq ($(findstring rv64, $(MARCH)), rv64)
	CFLAGS += -DMICO_ALIGN=8
	ifneq ($(findstring f, $(MARCH)),)
		MABI = lp64f
	else
		MABI = lp64
	endif
else
	ifneq ($(findstring f, $(MARCH)),)
		MABI = ilp32f
	else
		MABI = ilp32
	endif
endif

LARGE_RAM?=
HEAP_SIZE=4096*1024

BITNET_QUANT ?= 3
USE_SIMD ?= 32

ifeq ($(LARGE_RAM), 1)
	VEXII_BN_LD = $(MICO_DIR)/targets/vexii_bitnet/vexii_64mb.ld
	HEAP_SIZE=32*1024*1024
endif

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump

CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
CFLAGS += -DRISCV_VEXII -DTEST_NUM=$(TEST_NUM) -DMAX_HEAP_SIZE=$(HEAP_SIZE)
CFLAGS += -DBITNET_QUANT=$(BITNET_QUANT) -DUSE_SIMD=$(USE_SIMD)
CFLAGS += -fno-common -fno-inline
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -I${VEXII_BN_PATH}/ -I${VEXII_BN_PATH}/driver

LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
LDFLAGS += -nostdlib -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_BN_LD),-Map,$(BUILD)/$(MAIN).map,--print-memory-usage
LDFLAGS += -L./ -nolibc -lm -lc

ifneq ($(findstring f, $(MARCH)),)
    LDFLAGS += -lgcc
	CFLAGS += -DUSE_RVF
else ifneq ($(findstring rv64, $(MARCH)),)
	LDFLAGS += -lgcc
else ifneq ($(findstring m, $(MARCH)),)
    LDFLAGS += -L$(MICO_DIR)/lib/$(MARCH)/ -lrvfp
else
	LDFLAGS += -lgcc
endif
RISCV_SOURCE = $(wildcard $(VEXII_BN_PATH)/*.c) $(wildcard $(VEXII_BN_PATH)/*.S)
