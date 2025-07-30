X86_PATH = $(MICO_DIR)/targets/x86

MICO_SOURCES += $(wildcard $(X86_PATH)/*.c)
CFLAGS += -mavx -mavx2 -DUSE_HOST -DUSE_X86
