
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

ifneq ($(filter outer, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico_outer/*.c)
endif

ifneq ($(filter tiled, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico_tiled/*.c)
endif

ifneq ($(filter ref, $(OPT)),)
	CFLAGS += -DREF
endif

ifneq ($(filter alt-layout, $(OPT)),)
	CFLAGS += -DUSE_ALT_LAYOUT
endif