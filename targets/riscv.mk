RISCV_PATH = $(MICO_DIR)/targets/riscv

MABI?=ilp32
MARCH?=rv32im

# ABI Correction based on MARCH
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

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump

CFLAGS += -march=$(MARCH) -mabi=$(MABI)
CFLAGS += -DRISCV -DTEST_NUM=$(TEST_NUM)
CFLAGS += -fno-common
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -I${RISCV_PATH}/

LDFLAGS += -march=$(MARCH) -mabi=$(MABI)
LDFLAGS += -static
LDFLAGS += -lm -lc

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

MICO_SOURCES += $(wildcard $(RISCV_PATH)/*.c)
