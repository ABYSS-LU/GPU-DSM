cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(DSM_Demo CXX CUDA)

# Set CUDA architecture (A800 is SM80)
set(CMAKE_CUDA_ARCHITECTURES 80)
cmake_policy(SET CMP0104 NEW)

# Enable separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 查找 MPI
find_package(MPI REQUIRED)

# 查找 NVSHMEM
set(NVSHMEM_DIR /usr/local/nvshmem/lib/cmake/nvshmem)
find_package(NVSHMEM REQUIRED)

# 添加可执行文件
add_executable(dsm_demo dsm_demo.cu dummy.cu)

# Ensure device symbols are resolved for NVSHMEM kernels
set_target_properties(dsm_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dsm_demo PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# 包含 CUDA 头文件路径
target_include_directories(dsm_demo PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# 包含 MPI 头文件路径
target_include_directories(dsm_demo PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# 链接 NVSHMEM 库
target_link_libraries(dsm_demo PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device CUDA::cudart MPI::MPI_CXX)

# 如果需要，可以添加 CUDA
# find_package(CUDA REQUIRED)
# target_sources(dsm_demo PRIVATE your_cuda_file.cu)
# target_link_libraries(dsm_demo PRIVATE ${CUDA_LIBRARIES})
