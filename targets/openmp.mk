OMP_PATH = $(MICO_DIR)/targets/openmp

MICO_SOURCES += $(MICO_DIR)/targets/openmp/qmatmul.c
CFLAGS += -DUSE_HOST