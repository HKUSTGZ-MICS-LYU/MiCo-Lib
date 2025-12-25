VEXII_PATH = $(MICO_DIR)/targets/vexii_soc
VEXII_LD = $(MICO_DIR)/targets/vexii_soc/soc.ld

MABI?=ilp32
MARCH?=rv32imc

# Parse base ISA once (splits off any z* extensions after an underscore)
BASE_ISA := $(firstword $(subst _, ,$(MARCH)))
HAS_RV64 := $(findstring rv64,$(BASE_ISA))
HAS_D    := $(findstring d,$(BASE_ISA))
HAS_F    := $(findstring f,$(BASE_ISA))

# ABI Correction based on MARCH
MICO_SIMD_DIR = $(MICO_DIR)/targets/vexii/mico32
ifeq ($(HAS_RV64),rv64)
	MICO_SIMD_DIR = $(MICO_DIR)/targets/vexii/mico64
	CFLAGS += -DMICO_ALIGN=8
	ifneq ($(HAS_D),)
		MABI = lp64d
	else ifneq ($(HAS_F),)
		MABI = lp64f
	else
		MABI = lp64
	endif
else
	ifneq ($(HAS_F),)
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
LDFLAGS += -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(notdir $(MAIN)).map,--print-memory-usage
LDFLAGS += -L./ -lm -lc

ifneq ($(HAS_F)$(HAS_D),)
    LDFLAGS += -lgcc
	CFLAGS += -DUSE_RVF
else ifneq ($(HAS_RV64),)
	LDFLAGS += -lgcc
else ifneq ($(findstring m, $(MARCH)),)
    LDFLAGS += -L$(MICO_DIR)/lib/$(MARCH)/ -lrvfp
else
	LDFLAGS += -lgcc
endif

RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)
ifneq ($(filter cfu, $(OPT)),)
	RISCV_SOURCE += $(wildcard $(VEXII_PATH)/cfu/*.c) $(wildcard $(VEXII_PATH)/cfu/*.S)
	RISCV_SOURCE += $(wildcard $(VEXII_PATH)/cfu/v$(VLEN)/*.S)
	CFLAGS += -DMICO_ALIGN=$$(($(VLEN)/8))
endif

ifneq ($(filter legacy, $(OPT)),)
	MICO_SIMD_DIR := $(MICO_SIMD_DIR)/legacy
endif

ifneq ($(filter simd, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_SIMD_DIR)/*.c)
	RISCV_SOURCE += $(wildcard $(MICO_SIMD_DIR)/*.S)
endif
