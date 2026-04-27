cmake_minimum_required(VERSION 3.13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(DSM_Demo CXX CUDA)

# Provide compatibility find modules (e.g. FindCUDAToolkit) for older CMake.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set CUDA architecture (A800 is SM80)
# NOTE: For CMake < 3.18, use explicit -gencode flags.
set(DSM_CUDA_SM "80")
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  set(CMAKE_CUDA_ARCHITECTURES ${DSM_CUDA_SM})
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${DSM_CUDA_SM},code=sm_${DSM_CUDA_SM}")
endif()

# Enable separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA Toolkit
# - CMake >= 3.17: CUDAToolkit package provides CUDA::cudart target
# - Older CMake: fallback to legacy FindCUDA
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
  find_package(CUDAToolkit REQUIRED)
  set(DSM_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
  set(DSM_CUDART_LIB CUDA::cudart)
else()
  find_package(CUDA REQUIRED)
  set(DSM_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  set(DSM_CUDART_LIB ${CUDA_CUDART_LIBRARY})
endif()

# 查找 MPI
find_package(MPI REQUIRED)

# 查找 NVSHMEM
set(NVSHMEM_DIR /usr/local/nvshmem/lib/cmake/nvshmem)
find_package(NVSHMEM REQUIRED)

function(dsm_add_demo target_name src_file)
  add_executable(${target_name}
    ${src_file}
    src/dummy.cu
  )

  # Ensure device symbols are resolved for NVSHMEM kernels
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(${target_name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(${target_name} PRIVATE
    ${DSM_CUDA_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

  target_link_libraries(${target_name} PRIVATE
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device
    ${DSM_CUDART_LIB}
    MPI::MPI_CXX
  )
endfunction()

# Examples
dsm_add_demo(dsm_legacy_demo examples/legacy_demo.cu)
dsm_add_demo(dsm_global_array_demo examples/global_array_demo.cu)

# 如果需要，可以添加 CUDA
# find_package(CUDA REQUIRED)
# target_sources(dsm_demo PRIVATE your_cuda_file.cu)
# target_link_libraries(dsm_demo PRIVATE ${CUDA_LIBRARIES})
