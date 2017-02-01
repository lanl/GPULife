cmake_minimum_required (VERSION 2.8.11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
# --- custom targets: ---
INCLUDE( cmake/Modules/TargetDistclean.cmake OPTIONAL)

project (gameOfLife)

# The version number.
set (gameOfLife_VERSION_MAJOR 2)
set (gameOfLife_VERSION_MINOR 0.7)
set (gameOfLife_VERSION "${gameOfLife_VERSION_MAJOR}.${gameOfLife_VERSION_MINOR}")

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

# uncomment to build debug
#set(CMAKE_BUILD_TYPE Debug)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
   # using Clang
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -O0 -Wextra")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
   # using GCC
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -O0 -Wextra")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
   # using Intel C++
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -O0 -Wextra")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
   # using pgi
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "pgCC")
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -O0 -Wextra")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
endif()

find_package(MPI)

find_package( OpenACC)

find_package( OpenMP)

# This checks to see if the compiler supports the OpenMP4 standard and
# sets an OpenMP4_FOUND flag
if (NOT OpenMP OR NOT EXISTS "${OpenMP_C_SPEC_DATE}" OR "${OpenMP_C_SPEC_DATE}" LESS 201307)
  set (OpenMP4_FOUND false)
  message(STATUS "OpenMP4 not found: " ${OpenMP_C_SPEC_DATE})
else ()
  set (OpenMP4_FOUND true)
  message(STATUS "Found OpenMP4: " ${OpenMP_C_SPEC_DATE})
endif ()


# Cuda does not work on OSX with GCC compiler
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND APPLE)
  message(STATUS "CUDA disabled: cannot use GCC compiler on Mac OSx -- try using Clang instead")
else ()
  find_package( CUDA)
  if (CUDA_VERSION_STRING VERSION_EQUAL "7.5")
    if (("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU") AND
        (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.0") ) # or with GCC greater than 4.9 with CUDA 7.5`
      message(STATUS "CUDA disabled: cannot use GCC compiler greater than 4.9 with CUDA 7.5")
      set(CUDA_FOUND false)
    endif()
    if (("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel") AND
        (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "15.1.0") ) # or with Intel greater than 15.1.0 with CUDA 7.5`
      message(STATUS "CUDA disabled: cannot use Intel compiler greater than 15.1.0 with CUDA 7.5")
      set(CUDA_FOUND false)
    endif()
  endif (CUDA_VERSION_STRING VERSION_EQUAL "7.5")
endif ()

find_package(OpenCL)
#  OPENCL_FOUND        : TRUE if found, FALSE otherwise
#  OPENCL_INCLUDE_DIRS : Include directories for OpenCL
#  OPENCL_LIBRARIES    : The libraries to link against
if (OPENCL_FOUND)
   include_directories(${OPENCL_INCLUDE_DIRS})
endif (OPENCL_FOUND)

#set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_22,code=sm_22)
#set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode code=sm_22)

set(gameOfLife_SOURCES main.c applyRules_OpenMP4.c applyRules_OpenACC.c) 
if (CUDA_FOUND)
  set(gameOfLife_SOURCES ${gameOfLife_SOURCES} applyRules_CUDA.cu)
  cuda_add_executable (gameOfLife ${gameOfLife_SOURCES}) 
else (CUDA_FOUND)
   add_executable (gameOfLife ${gameOfLife_SOURCES}) 
endif (CUDA_FOUND)


#set_source_files_properties(applyRules_OpenMP4.c PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})

if (MPI_FOUND)
   include_directories(${MPI_INCLUDE_PATH})
   set(gameOfLife_DEFINITIONS "${gameOfLife_DEFINITIONS};HAVE_MPI")
   target_link_libraries(gameOfLife ${MPI_LIBRARIES})
endif (MPI_FOUND)

if (OpenACC_FOUND)
   set(gameOfLife_DEFINITIONS "${gameOfLife_DEFINITIONS};HAVE_ACC;_OPENACC")
   set_source_files_properties(gameOfLife PROPERTIES COMPILE_FLAGS ${OpenACC_C_FLAGS})
   target_link_libraries(gameOfLife ${OpenACC_LIBRARIES})
endif (OpenACC_FOUND)

if (OpenMP4_FOUND)
   set(gameOfLife_DEFINITIONS "${gameOfLife_DEFINITIONS};HAVE_OPENMP4")
   target_link_libraries(gameOfLife ${OpenMP_LIBRARIES})
endif (OpenMP4_FOUND)

if (CUDA_FOUND)
   set(gameOfLife_DEFINITIONS "${gameOfLife_DEFINITIONS};HAVE_CUDA")
   target_link_libraries(gameOfLife ${CUDA_LIBRARIES})
endif (CUDA_FOUND)

set_target_properties(gameOfLife PROPERTIES COMPILE_DEFINITIONS "${gameOfLife_DEFINITIONS}")

#if (OpenCL_FOUND)
#   set_target_properties(gameOfLife PROPERTIES COMPILE_DEFINITIONS HAVE_OPENCL)
#   set_source_files_properties(applyRules_OpenCL.c PROPERTIES COMPILE_FLAGS ${OpenACC_C_FLAGS})
#   target_link_libraries(gameOfLife ${OpenCL_LIBRARIES})
#endif (OpenMP_FOUND)


# we need to add a version check
#if(GCC_VERSION VERSION_LESS "4.9")
	#then cuda is allowed and openmp4 is not
#else
	# then openmp4 is allowed and cuda is not

