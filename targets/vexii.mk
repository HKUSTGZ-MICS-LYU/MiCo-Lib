VEXII_PATH = $(MICO_DIR)/targets/vexii
VEXII_LD = $(MICO_DIR)/targets/vexii/vexii.ld

MABI?=ilp32
MARCH?=rv32imc
MICO_SIMD_DIR?=mico32
ifeq ($(findstring rv64, $(MARCH)), rv64)
	MICO_SIMD_DIR = mico64
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

ifeq ($(LARGE_RAM), 1)
	VEXII_LD = $(MICO_DIR)/targets/vexii/vexii_64mb.ld
	HEAP_SIZE=32*1024*1024
endif

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump

CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
CFLAGS += -DRISCV_VEXII -DTEST_NUM=$(TEST_NUM) -DMAX_HEAP_SIZE=$(HEAP_SIZE)
CFLAGS += -fno-common -fno-inline
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -I${VEXII_PATH}/ -I${VEXII_PATH}/driver

LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
LDFLAGS += -nostdlib -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(MAIN).map,--print-memory-usage
LDFLAGS += -L./ -nolibc -lm -lc

ifneq ($(findstring f, $(MARCH)),)
    LDFLAGS += -lgcc
	CFLAGS += -DUSE_RVF
else
    LDFLAGS += -L$(MICO_DIR)/lib/ -lrvfp
endif

RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)
ifneq ($(filter simd, $(OPT)),)
	MICO_SOURCES += $(wildcard $(VEXII_PATH)/$(MICO_SIMD_DIR)/*.c)
	RISCV_SOURCE += $(wildcard $(VEXII_PATH)/$(MICO_SIMD_DIR)/*.S)
endif
