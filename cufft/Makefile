# Check CUDA Version
CUDA_DIR = /usr/local/cuda

# 32-bit or 64-bit  
OS_TYPE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

# Compilers
GCC ?= g++
NVC = $(CUDA_DIR)/bin/nvcc  -ccbin $(GCC)

NVCC_FLAGS   := 
NVCC_FLAGS   += -O2 
NVCC_FLAGS   += -m${OS_TYPE}

# Directories for Header Files 
NV_INC  = -I$(CUDA_DIR)/include/ -I$(CUDA_DIR)/samples/common/inc
NV_INC +=  -I../

# Directories for Libraries
ifeq ($(OS_TYPE), 64)
  NV_LIB = -L$(CUDA_DIR)/lib64
else
  NV_LIB = -L$(CUDA_DIR)/lib
endif

LIB_LINKER = -lcudart -lcufft

SMS ?= 30 35 50 52 60 61 70 75

ifeq ($(GENCODE_FLAGS),)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

#-----------------------------------------------------------------------------#
# Build Targets
#-----------------------------------------------------------------------------#
EXE = cuFFT_1d cuFFT_2d cuFFT_3d  cuFFT_1d_solution cuFFT_2d_solution cuFFT_3d_solution test1d test1d_many

all: $(EXE)

cuFFT_1d: cuFFT_1d.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

cuFFT_1d_solution: cuFFT_1d_solution.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

test1d: test1d.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

test1d_many: test1d_many.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

cuFFT_2d: cuFFT_2d.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

cuFFT_2d_solution: cuFFT_2d_solution.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

cuFFT_3d: cuFFT_3d.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

cuFFT_3d_solution: cuFFT_3d_solution.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

.PHONY: clean
clean:
	rm -rf $(EXE) 
