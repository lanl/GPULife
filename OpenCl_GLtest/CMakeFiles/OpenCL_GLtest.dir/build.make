# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.6.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.6.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/brobey/Programs/GPULife/OpenCl_GLtest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/brobey/Programs/GPULife/OpenCl_GLtest

# Include any dependencies generated for this target.
include CMakeFiles/OpenCL_GLtest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/OpenCL_GLtest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/OpenCL_GLtest.dir/flags.make

CMakeFiles/OpenCL_GLtest.dir/main.c.o: CMakeFiles/OpenCL_GLtest.dir/flags.make
CMakeFiles/OpenCL_GLtest.dir/main.c.o: main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/brobey/Programs/GPULife/OpenCl_GLtest/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/OpenCL_GLtest.dir/main.c.o"
	/usr/local/bin/gcc-6  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/OpenCL_GLtest.dir/main.c.o   -c /Users/brobey/Programs/GPULife/OpenCl_GLtest/main.c

CMakeFiles/OpenCL_GLtest.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/OpenCL_GLtest.dir/main.c.i"
	/usr/local/bin/gcc-6  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/brobey/Programs/GPULife/OpenCl_GLtest/main.c > CMakeFiles/OpenCL_GLtest.dir/main.c.i

CMakeFiles/OpenCL_GLtest.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/OpenCL_GLtest.dir/main.c.s"
	/usr/local/bin/gcc-6  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/brobey/Programs/GPULife/OpenCl_GLtest/main.c -o CMakeFiles/OpenCL_GLtest.dir/main.c.s

CMakeFiles/OpenCL_GLtest.dir/main.c.o.requires:

.PHONY : CMakeFiles/OpenCL_GLtest.dir/main.c.o.requires

CMakeFiles/OpenCL_GLtest.dir/main.c.o.provides: CMakeFiles/OpenCL_GLtest.dir/main.c.o.requires
	$(MAKE) -f CMakeFiles/OpenCL_GLtest.dir/build.make CMakeFiles/OpenCL_GLtest.dir/main.c.o.provides.build
.PHONY : CMakeFiles/OpenCL_GLtest.dir/main.c.o.provides

CMakeFiles/OpenCL_GLtest.dir/main.c.o.provides.build: CMakeFiles/OpenCL_GLtest.dir/main.c.o


CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o: CMakeFiles/OpenCL_GLtest.dir/flags.make
CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o: applyRules_OpenCL.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/brobey/Programs/GPULife/OpenCl_GLtest/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o"
	/usr/local/bin/gcc-6  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o   -c /Users/brobey/Programs/GPULife/OpenCl_GLtest/applyRules_OpenCL.c

CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.i"
	/usr/local/bin/gcc-6  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/brobey/Programs/GPULife/OpenCl_GLtest/applyRules_OpenCL.c > CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.i

CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.s"
	/usr/local/bin/gcc-6  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/brobey/Programs/GPULife/OpenCl_GLtest/applyRules_OpenCL.c -o CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.s

CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.requires:

.PHONY : CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.requires

CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.provides: CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.requires
	$(MAKE) -f CMakeFiles/OpenCL_GLtest.dir/build.make CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.provides.build
.PHONY : CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.provides

CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.provides.build: CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o


# Object files for target OpenCL_GLtest
OpenCL_GLtest_OBJECTS = \
"CMakeFiles/OpenCL_GLtest.dir/main.c.o" \
"CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o"

# External object files for target OpenCL_GLtest
OpenCL_GLtest_EXTERNAL_OBJECTS =

OpenCL_GLtest: CMakeFiles/OpenCL_GLtest.dir/main.c.o
OpenCL_GLtest: CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o
OpenCL_GLtest: CMakeFiles/OpenCL_GLtest.dir/build.make
OpenCL_GLtest: /usr/local/lib/libGLEW.dylib
OpenCL_GLtest: CMakeFiles/OpenCL_GLtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/brobey/Programs/GPULife/OpenCl_GLtest/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable OpenCL_GLtest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OpenCL_GLtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/OpenCL_GLtest.dir/build: OpenCL_GLtest

.PHONY : CMakeFiles/OpenCL_GLtest.dir/build

CMakeFiles/OpenCL_GLtest.dir/requires: CMakeFiles/OpenCL_GLtest.dir/main.c.o.requires
CMakeFiles/OpenCL_GLtest.dir/requires: CMakeFiles/OpenCL_GLtest.dir/applyRules_OpenCL.c.o.requires

.PHONY : CMakeFiles/OpenCL_GLtest.dir/requires

CMakeFiles/OpenCL_GLtest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/OpenCL_GLtest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/OpenCL_GLtest.dir/clean

CMakeFiles/OpenCL_GLtest.dir/depend:
	cd /Users/brobey/Programs/GPULife/OpenCl_GLtest && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest/CMakeFiles/OpenCL_GLtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/OpenCL_GLtest.dir/depend
