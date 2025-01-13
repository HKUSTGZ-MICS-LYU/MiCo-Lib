ROCKET_PATH = $(MICO_DIR)/targets/rocket

ifeq ($(TARGET), rocket)
	CC = $(RISCV_PREFIX)-gcc
	OBJDUMP = $(RISCV_PREFIX)-objdump

	CFLAGS += -DRISCV_ROCKET -DTEST_NUM=$(TEST_NUM)
	CFLAGS += -fno-common -fno-builtin-printf -specs=htif_nano.specs

	LDFLAGS += -static

	RISCV_SOURCE = $(wildcard $(ROCKET_PATH)/*.c) $(wildcard $(ROCKET_PATH)/*.S)
endif