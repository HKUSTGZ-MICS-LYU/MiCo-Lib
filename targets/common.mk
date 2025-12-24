
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

ifneq ($(filter opt, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/optimized/*.c)
endif

ifneq ($(filter lut, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico_lut/*.c)
endif

ifneq ($(filter ref, $(OPT)),)
	CFLAGS += -DREF
endif

ifneq ($(filter quant_reuse, $(OPT)),)
	CFLAGS += -DQUANT_REUSE
endif

HAS_UNROLL := $(filter unroll, $(OPT))
HAS_LUT := $(filter lut, $(OPT))

ifneq ($(HAS_UNROLL),)
	CFLAGS += -DMICO_HAS_UNROLL
endif

ifneq ($(HAS_LUT),)
	CFLAGS += -DMICO_HAS_LUT
endif

ifneq ($(HAS_UNROLL),)
	ifneq ($(HAS_LUT),)
		# Both available, default is explicit selection
	else
		CFLAGS += -DMICO_DEFAULT_MATMUL_OPT=1
	endif
else ifneq ($(HAS_LUT),)
	CFLAGS += -DMICO_DEFAULT_MATMUL_OPT=2
endif
