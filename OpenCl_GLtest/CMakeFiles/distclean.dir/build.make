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

# Utility rule file for distclean.

# Include the progress variables for this target.
include CMakeFiles/distclean.dir/progress.make

CMakeFiles/distclean:
	@echo cleaning for source distribution

distclean: CMakeFiles/distclean
distclean: CMakeFiles/distclean.dir/build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "distribution clean"
	rm -Rf CMakeTmp cmake.depends cmake.check_depends CMakeCache.txt */CMakeCache.txt cmake.check_cache *.cmake *.a */*.a Makefile */tests/Makefile core core.* gmon.out *~ CMakeFiles */CMakeFiles */*/CMakeFiles */CTestTestfile.cmake */*/CTestTestfile.cmake */Testing */*/Testing cmake_install.cmake */cmake_install.cmake */*/cmake_install.cmake install_manifest.txt */install_manifest.txt *.dSYM */*.dSYM */*/*.dSYM
.PHONY : distclean

# Rule to build all files generated by this target.
CMakeFiles/distclean.dir/build: distclean

.PHONY : CMakeFiles/distclean.dir/build

CMakeFiles/distclean.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/distclean.dir/cmake_clean.cmake
.PHONY : CMakeFiles/distclean.dir/clean

CMakeFiles/distclean.dir/depend:
	cd /Users/brobey/Programs/GPULife/OpenCl_GLtest && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest /Users/brobey/Programs/GPULife/OpenCl_GLtest/CMakeFiles/distclean.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/distclean.dir/depend

