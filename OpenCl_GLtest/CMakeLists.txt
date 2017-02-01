cmake_minimum_required (VERSION 2.8.11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
# --- custom targets: ---
INCLUDE( cmake/Modules/TargetDistclean.cmake OPTIONAL)

project (OpenCl_GLtest)

include_directories ( /System/Library/Frameworks )

# The version number.
set (OpenCL_GLtest_VERSION_MAJOR 2)
set (OpenCL_GLtest_VERSION_MINOR 0.7)
set (OpenCL_GLtest_VERSION "${OpenCL_GLtest_VERSION_MAJOR}.${OpenCL_GLtest_VERSION_MINOR}")

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

# uncomment to build debug
#set(CMAKE_BUILD_TYPE Debug)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
   # using Clang
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -g -O0 -Wextra")
   set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g -O3")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
   # using GCC
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
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

find_package(OpenCL)
#  OPENCL_FOUND        : TRUE if found, FALSE otherwise
#  OPENCL_INCLUDE_DIRS : Include directories for OpenCL
#  OPENCL_LIBRARIES    : The libraries to link against
if (OPENCL_FOUND)
   include_directories(${OPENCL_INCLUDE_DIRS})
endif (OPENCL_FOUND)

find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)

find_package(GLEW REQUIRED)

set(OpenCL_GLtest_SOURCES main.c applyRules_OpenCL.c) 

add_executable (OpenCL_GLtest ${OpenCL_GLtest_SOURCES}) 

if (OpenCL_FOUND)
   set(OpenCL_GLtest_DEFINITIONS "${OpenCL_GLtest_DEFINITIONS};HAVE_OPENCL")
   #set_target_properties(OpenCL_GLtest PROPERTIES COMPILE_FLAGS "${OPENCL_C_FLAGS}")
   target_link_libraries(OpenCL_GLtest ${OpenCL_LIBRARIES})
endif (OpenCL_FOUND)

if (OpenGL_FOUND)
   set(OpenCL_GLtest_DEFINITIONS "${OpenCL_GLtest_DEFINITIONS};HAVE_OPENGL")
   set_target_properties(OpenCL_GLtest PROPERTIES COMPILE_FLAGS "-I${OPENGL_INCLUDE_DIR}")
   # Need both GL and GLUT libraries
   target_link_libraries(OpenCL_GLtest ${OPENGL_LIBRARIES})
endif (OpenGL_FOUND)

if (GLUT_FOUND)
   set(OpenCL_GLtest_DEFINITIONS "${OpenCL_GLtest_DEFINITIONS};HAVE_GLUT")
   set_target_properties(OpenCL_GLtest PROPERTIES COMPILE_FLAGS "-I${GLUT_INCLUDE_DIR}")
   target_link_libraries(OpenCL_GLtest ${GLUT_LIBRARIES})
endif (GLUT_FOUND)

if (GLEW_FOUND)
   set(OpenCL_GLtest_DEFINITIONS "${OpenCL_GLtest_DEFINITIONS};HAVE_GLEW")
   set_target_properties(OpenCL_GLtest PROPERTIES COMPILE_FLAGS "-I${GLEW_INCLUDE_DIRS}")
   target_link_libraries(OpenCL_GLtest ${GLEW_LIBRARIES})
endif (GLEW_FOUND)

set_target_properties(OpenCL_GLtest PROPERTIES COMPILE_DEFINITIONS "${OpenCL_GLtest_DEFINITIONS}")

