
# Software Optimization
ifneq ($(filter unroll, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico_unrolled/*.c)
endif

ifneq ($(filter gcc-unroll, $(OPT)),)
	CFLAGS += -funroll-all-loops
endif

ifneq ($(filter im2col, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/im2col_conv2d/*.c)
endif

ifneq ($(filter riscv, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/riscv/*.c)
endif

ifneq ($(filter ref, $(OPT)),)
	CFLAGS += -DREF
endif

ifneq ($(filter quant_reuse, $(OPT)),)
	CFLAGS += -DQUANT_REUSE
endif