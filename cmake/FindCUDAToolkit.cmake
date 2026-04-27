# Minimal FindCUDAToolkit.cmake for older CMake versions.
#
# NVSHMEM's CMake package requires: find_dependency(CUDAToolkit REQUIRED)
# but CMake 3.13 doesn't ship with FindCUDAToolkit.cmake.
#
# This module provides:
# - CUDAToolkit_FOUND
# - CUDAToolkit_INCLUDE_DIRS
# - CUDAToolkit_LIBRARY_DIR
# - CUDAToolkit_VERSION_MAJOR
# - Imported targets: CUDA::cudart, CUDA::cudart_static (needed by NVSHMEMDeviceTargets)

include(FindPackageHandleStandardArgs)

# Prefer values from FindCUDA if already executed.
set(_cuda_root "")
if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND CUDA_TOOLKIT_ROOT_DIR)
  set(_cuda_root "${CUDA_TOOLKIT_ROOT_DIR}")
elseif(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
  set(_cuda_root "$ENV{CUDA_HOME}")
elseif(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
  set(_cuda_root "$ENV{CUDA_PATH}")
else()
  set(_cuda_root "/usr/local/cuda")
endif()

find_path(CUDAToolkit_INCLUDE_DIR
  NAMES cuda_runtime.h
  PATHS "${_cuda_root}/include"
  NO_DEFAULT_PATH
)

find_library(CUDAToolkit_CUDART_LIBRARY
  NAMES cudart
  PATHS "${_cuda_root}/lib64" "${_cuda_root}/lib"
  NO_DEFAULT_PATH
)

find_library(CUDAToolkit_CUDART_STATIC_LIBRARY
  NAMES cudart_static
  PATHS "${_cuda_root}/lib64" "${_cuda_root}/lib"
  NO_DEFAULT_PATH
)

set(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIR}")
if(EXISTS "${_cuda_root}/lib64")
  set(CUDAToolkit_LIBRARY_DIR "${_cuda_root}/lib64")
else()
  set(CUDAToolkit_LIBRARY_DIR "${_cuda_root}/lib")
endif()

# Try to reuse CUDA_VERSION from FindCUDA.
if(DEFINED CUDA_VERSION AND NOT "${CUDA_VERSION}" STREQUAL "")
  string(REPLACE "." ";" _cuda_ver_list "${CUDA_VERSION}")
  list(GET _cuda_ver_list 0 CUDAToolkit_VERSION_MAJOR)
else()
  # Best-effort default for modern CUDA installs.
  set(CUDAToolkit_VERSION_MAJOR 12)
endif()

find_package_handle_standard_args(CUDAToolkit
  REQUIRED_VARS CUDAToolkit_INCLUDE_DIR CUDAToolkit_CUDART_LIBRARY CUDAToolkit_CUDART_STATIC_LIBRARY
)

if(CUDAToolkit_FOUND)
  if(NOT TARGET CUDA::cudart)
    add_library(CUDA::cudart SHARED IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES
      IMPORTED_LOCATION "${CUDAToolkit_CUDART_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
    )
  endif()

  if(NOT TARGET CUDA::cudart_static)
    add_library(CUDA::cudart_static STATIC IMPORTED)
    set_target_properties(CUDA::cudart_static PROPERTIES
      IMPORTED_LOCATION "${CUDAToolkit_CUDART_STATIC_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
    )
  endif()
endif()

