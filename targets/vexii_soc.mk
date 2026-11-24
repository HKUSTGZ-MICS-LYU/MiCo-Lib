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
MICO_BNRV_DIR = $(MICO_DIR)/targets/vexii_bitnet

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
SPARSE_RAM_SIZE?=256M
HEAP_SIZE?=32K
STACK_SIZE?=$(HEAP_SIZE)
SPRAM?=

ifeq ($(SPRAM), 1)
	VEXII_LD = $(MICO_DIR)/targets/vexii_soc/soc_sp.ld
	HEAP_SIZE = 32M
	STACK_SIZE = 32M
	CFLAGS += -DSPRAM
endif

# For MiCo CFU VPU
VLEN ?= 128

# For MiCo BNRV
USE_SIMD ?= 32
BITNET_QUANT ?= 3
BNCFU_REG_DEPTH ?= 2
# Opt in only with matching offline-packed W8 artifacts. Default keeps the
# original raw-INT8 kernel for paper reproduction, independently of Q8 quant.
BNCFU_INT8_MATMUL ?= 0
ifneq ($(BNCFU_INT8_MATMUL),0)
ifneq ($(BNCFU_INT8_MATMUL),1)
$(error BNCFU_INT8_MATMUL must be 0 or 1)
endif
endif
# Prepacked Q8: 1 = radix4 on hardware --bitnet-cfu-with-q2;
# default 0 = balanced ternary3 on Q1.58-only hardware. Match Python --qtype.
BNCFU_WITH_Q2 ?= 0
BNCFU_Q2T ?= 0
BNCFU_Q8 ?= 0

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
LDFLAGS += -Wl,--defsym=SPARSE_RAM_SIZE=$(SPARSE_RAM_SIZE)
LDFLAGS += -Wl,--defsym=HEAP_SIZE=$(HEAP_SIZE)
LDFLAGS += -Wl,--defsym=STACK_SIZE=$(STACK_SIZE)

LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
LDFLAGS += -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(notdir $(MAIN)).map,--print-memory-usage
LDFLAGS += -L./ -lm -lc

MARCH_BASE = $(shell echo $(MARCH) | sed 's/_.*$$//')

ifneq ($(HAS_F)$(HAS_D),)
    LDFLAGS += -lgcc
	CFLAGS += -DUSE_RVF
else ifneq ($(HAS_RV64),)
	LDFLAGS += -lgcc
else ifneq ($(findstring m, $(MARCH_BASE)),)
    LDFLAGS += -L$(MICO_DIR)/lib/$(MARCH_BASE)/ -lrvfp
else
	LDFLAGS += -lgcc
endif

RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)

# Integer RVV matmul on Vexii: retain the Vexii CRT and scalar FP support.
# Compiler ISA feature macros describe instruction encodings, not conformance
# of an experimental accelerator integration. Never undefine __riscv_zve32x.
ifneq ($(filter rvv, $(OPT)),)
RVV_BACKEND ?= coral
RVV_Q8_EEW_EQ_SEW ?= 1
ifeq ($(RVV_BACKEND),coral)
	CFLAGS += -DMICO_RVV_BACKEND_CORAL=1
else ifeq ($(RVV_BACKEND),vicuna2)
	CFLAGS += -DMICO_RVV_BACKEND_VICUNA2=1
ifneq ($(filter bncfu cfu bnrv simd, $(OPT)),)
$(error RVV_BACKEND=vicuna2 cannot be combined with another MatMul backend)
endif
ifeq ($(RVV_CACHE_MAINTENANCE),1)
$(error Vicuna2 software cache maintenance is not validated; use cacheless memory)
endif
else
$(error RVV_BACKEND must be coral or vicuna2)
endif
# Both integrations start with the proven conservative Q8 load helper. A future
# opt-out must be paired with a real mixed-EEW correctness gate, not an unchecked
# "verified" command-line assertion. No such gate is provided by this version.
ifneq ($(RVV_Q8_EEW_EQ_SEW),1)
$(error RVV_Q8_EEW_EQ_SEW must remain 1 until a mixed-EEW gate is implemented)
endif
	MICO_SOURCES += $(MICO_DIR)/targets/rvv/qmatmul.c
	MICO_SOURCES += $(MICO_DIR)/targets/rvv/qmatmul_lowbit.c
	CFLAGS += -DMICO_RVV -DMICO_RVV_HAS_REDUCTION=1 -DMICO_RVV_Q8_EEW_EQ_SEW=$(RVV_Q8_EEW_EQ_SEW)
ifeq ($(RVV_CACHE_MAINTENANCE),1)
	CFLAGS += -DMICO_RVV_EXPLICIT_CACHE
endif
	# Only explicit RVV kernels: keep scalar support code scalar. GCC can expand
	# a recognized strlen loop into vle8ff even with tree vectorization disabled;
	# Coral's bridge does not support fault-only-first memory instructions.
	CFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize -fno-builtin
	# Restore normal O3 inline optimization only for the new RVV backend.
	CFLAGS += -finline
	# Keep compiler-generated support/structure copies out of the RVV subset.
	CFLAGS += -mstringop-strategy=scalar
endif

ifneq ($(filter cfu, $(OPT)),)
	ifneq ($(filter legacy, $(OPT)),)
		RISCV_SOURCE += $(wildcard $(VEXII_PATH)/cfu/v$(VLEN)/*.S)
		RISCV_SOURCE += $(wildcard $(VEXII_PATH)/cfu/legacy/*.c) $(wildcard $(VEXII_PATH)/cfu/legacy/*.S)
	else
		RISCV_SOURCE += $(wildcard $(VEXII_PATH)/cfu/*.c) $(wildcard $(VEXII_PATH)/cfu/*.S)
	endif
	CFLAGS += -DMICO_ALIGN=$$(($(VLEN)/8)) -DVLEN=$(VLEN)
endif

ifneq ($(filter simd, $(OPT)),)
	ifneq ($(filter legacy, $(OPT)),)
		MICO_SIMD_DIR := $(MICO_SIMD_DIR)/legacy
	endif
	MICO_SOURCES += $(wildcard $(MICO_SIMD_DIR)/*.c)
	RISCV_SOURCE += $(wildcard $(MICO_SIMD_DIR)/*.S)
endif

ifneq ($(filter bnrv, $(OPT)),)
	MICO_SOURCES += $(MICO_BNRV_DIR)/bitnet_matmul.c
	RISCV_SOURCE += $(MICO_BNRV_DIR)/bnrv.S $(MICO_BNRV_DIR)/dotp.S
	CFLAGS += -DUSE_SIMD=$(USE_SIMD) -DBITNET_QUANT=$(BITNET_QUANT)
endif

ifneq ($(filter bncfu, $(OPT)),)
	MICO_SOURCES += $(filter-out $(VEXII_PATH)/bncfu/bitnet_cfu_q8_matmul.c,$(wildcard $(VEXII_PATH)/bncfu/*.c))
	CFLAGS += -DMICO_ALIGN=$$(($(VLEN)/8)) -DVLEN=$(VLEN) -DBITNET_QUANT=$(BITNET_QUANT) -DBNCFU_REG_DEPTH=$(BNCFU_REG_DEPTH) -DBNCFU_WITH_Q2=$(BNCFU_WITH_Q2) -DBNCFU_INT8_MATMUL=$(BNCFU_INT8_MATMUL) -DKIVI_BNCFU_INT8_BDOT
ifeq ($(BNCFU_INT8_MATMUL),1)
	MICO_SOURCES += $(VEXII_PATH)/bncfu/bitnet_cfu_q8_matmul.c
	CFLAGS += -DMICO_BNCFU_Q8_OVERRIDE
endif
ifeq ($(BNCFU_Q2T),1)
	CFLAGS += -DBNCFU_Q2T
endif
ifeq ($(BNCFU_Q8),1)
	CFLAGS += -DBNCFU_Q8
endif
ifneq ($(BNCFU_QUANT_WIDTH),)
	CFLAGS += -DBNCFU_QUANT_WIDTH=$(BNCFU_QUANT_WIDTH)
endif
endif
