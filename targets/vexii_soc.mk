VEXII_PATH = $(MICO_DIR)/targets/vexii_soc
VEXII_LD = $(MICO_DIR)/targets/vexii_soc/soc.ld

MABI?=ilp32
MARCH?=rv32imc

RAM_SIZE?=1024K
HEAP_SIZE?=32K
STACK_SIZE?=$(HEAP_SIZE)

CC = $(RISCV_PREFIX)-gcc
OBJDUMP = $(RISCV_PREFIX)-objdump

CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
CFLAGS += -DRISCV_VEXII -DTEST_NUM=$(TEST_NUM) -DMAX_HEAP_SIZE=$(HEAP_SIZE)
CFLAGS += -fno-common -fno-inline
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -I${VEXII_PATH}/ -I${VEXII_PATH}/driver

LDFLAGS += -Wl,--defsym=RAM_SIZE=$(RAM_SIZE)
LDFLAGS += -Wl,--defsym=HEAP_SIZE=$(HEAP_SIZE)
LDFLAGS += -Wl,--defsym=STACK_SIZE=$(STACK_SIZE)

LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
LDFLAGS += -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(MAIN).map,--print-memory-usage
LDFLAGS += -L./ -lm -lc

ifeq ($(MARCH), rv32imc)
	LDFLAGS += -L$(MICO_DIR)/lib/ -lrvfp
else
	LDFLAGS += -lgcc
endif

RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)
