CC = nvcc

CUDA_PATH = $(MICO_DIR)/targets/cuda
CUDA_SOURCE = $(wildcard $(CUDA_PATH)/*.cu)
OBJS += $(CUDA_SOURCE:.cu=.o)
CFLAGS += -DUSE_CUDA -DUSE_GPU

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	@echo "Compiling Source File ($<)..."
	@$(CC) $(CFLAGS) -c -o $@ $< $(addprefix -I,$(INCLUDES))
