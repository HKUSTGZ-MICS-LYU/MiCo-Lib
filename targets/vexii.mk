VEXII_PATH = $(MICO_DIR)/targets/vexii
VEXII_LD = $(MICO_DIR)/targets/vexii/vexii.ld

MABI?=ilp32
MARCH?=rv32imc

ifeq ($(TARGET), vexii)
	CC = $(RISCV_PREFIX)-gcc
	OBJDUMP = $(RISCV_PREFIX)-objdump

	CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
	CFLAGS += -DRISCV_VEXII -DTEST_NUM=1
	CFLAGS += -fno-common -fno-inline
	CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
	CFLAGS += -I${VEXII_PATH}/ -I${VEXII_PATH}/driver

	LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
	LDFLAGS += -nostdlib -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(MAIN).map,--print-memory-usage
	LDFLAGS += -L./ -nolibc -lm -lc -lgcc

	RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)
	ifneq ($(filter simd, $(OPT)),)
		MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico_simd/*.c)
		RISCV_SOURCE += $(wildcard ${VEXII_PATH}/mico/*.S)
	endif
endif