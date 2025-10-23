VEXII_PATH = $(MICO_DIR)/targets/vexii_soc
VEXII_LD = $(MICO_DIR)/targets/vexii_soc/soc.ld

MABI?=ilp32
MARCH?=rv32imc

# ABI Correction based on MARCH
MICO_SIMD_DIR = $(MICO_DIR)/targets/vexii/mico32
ifeq ($(findstring rv64, $(MARCH)), rv64)
	MICO_SIMD_DIR = $(MICO_DIR)/targets/vexii/mico64
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

REPAET?=0

RAM_SIZE?=256K
HEAP_SIZE?=32K
STACK_SIZE?=$(HEAP_SIZE)

VLEN ?= 128

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump

CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
CFLAGS += -DRISCV_VEXII -DSOC -DTEST_NUM=$(TEST_NUM) -DMAX_HEAP_SIZE=$(HEAP_SIZE)

ifeq ($(REPAET), 1)
CFLAGS += -DREPEAT
endif

CFLAGS += -fno-common -fno-inline
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -I${VEXII_PATH}/ -I${VEXII_PATH}/driver

LDFLAGS += -Wl,--defsym=RAM_SIZE=$(RAM_SIZE)
LDFLAGS += -Wl,--defsym=HEAP_SIZE=$(HEAP_SIZE)
LDFLAGS += -Wl,--defsym=STACK_SIZE=$(STACK_SIZE)

LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
LDFLAGS += -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(MAIN).map,--print-memory-usage
LDFLAGS += -L./ -lm -lc

ifneq ($(findstring f, $(MARCH)),)
    LDFLAGS += -lgcc
	CFLAGS += -DUSE_RVF
else ifneq ($(findstring rv64, $(MARCH)),)
    LDFLAGS += -lgcc
else
    LDFLAGS += -L$(MICO_DIR)/lib/ -lrvfp
endif

RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)
ifneq ($(filter cfu, $(OPT)),)
	RISCV_SOURCE += $(wildcard $(VEXII_PATH)/cfu/*.c) $(wildcard $(VEXII_PATH)/cfu/*.S)
	RISCV_SOURCE += $(wildcard $(VEXII_PATH)/cfu/v$(VLEN)/*.S)
	CFLAGS += -DMICO_ALIGN=$$(($(VLEN)/8))
endif

ifneq ($(filter simd, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_SIMD_DIR)/*.c)
	RISCV_SOURCE += $(wildcard $(MICO_SIMD_DIR)/*.S)
endif