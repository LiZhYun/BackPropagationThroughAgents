# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /share/apps/spack/envs/fgci-centos7-generic/software/cmake/3.15.3/qvh6hn6/bin/cmake

# The command to remove a file.
RM = /share/apps/spack/envs/fgci-centos7-generic/software/cmake/3.15.3/qvh6hn6/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build

# Include any dependencies generated for this target.
include hanabi_lib/CMakeFiles/hanabi.dir/depend.make

# Include the progress variables for this target.
include hanabi_lib/CMakeFiles/hanabi.dir/progress.make

# Include the compile flags for this target's objects.
include hanabi_lib/CMakeFiles/hanabi.dir/flags.make

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_card.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/hanabi_card.cc.o: ../hanabi_lib/hanabi_card.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/hanabi_card.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/hanabi_card.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_card.cc

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_card.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/hanabi_card.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_card.cc > CMakeFiles/hanabi.dir/hanabi_card.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_card.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/hanabi_card.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_card.cc -o CMakeFiles/hanabi.dir/hanabi_card.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_game.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/hanabi_game.cc.o: ../hanabi_lib/hanabi_game.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/hanabi_game.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/hanabi_game.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_game.cc

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_game.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/hanabi_game.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_game.cc > CMakeFiles/hanabi.dir/hanabi_game.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_game.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/hanabi_game.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_game.cc -o CMakeFiles/hanabi.dir/hanabi_game.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_hand.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/hanabi_hand.cc.o: ../hanabi_lib/hanabi_hand.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/hanabi_hand.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/hanabi_hand.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_hand.cc

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_hand.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/hanabi_hand.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_hand.cc > CMakeFiles/hanabi.dir/hanabi_hand.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_hand.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/hanabi_hand.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_hand.cc -o CMakeFiles/hanabi.dir/hanabi_hand.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_history_item.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/hanabi_history_item.cc.o: ../hanabi_lib/hanabi_history_item.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/hanabi_history_item.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/hanabi_history_item.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_history_item.cc

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_history_item.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/hanabi_history_item.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_history_item.cc > CMakeFiles/hanabi.dir/hanabi_history_item.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_history_item.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/hanabi_history_item.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_history_item.cc -o CMakeFiles/hanabi.dir/hanabi_history_item.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_move.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/hanabi_move.cc.o: ../hanabi_lib/hanabi_move.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/hanabi_move.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/hanabi_move.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_move.cc

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_move.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/hanabi_move.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_move.cc > CMakeFiles/hanabi.dir/hanabi_move.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_move.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/hanabi_move.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_move.cc -o CMakeFiles/hanabi.dir/hanabi_move.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_observation.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/hanabi_observation.cc.o: ../hanabi_lib/hanabi_observation.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/hanabi_observation.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/hanabi_observation.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_observation.cc

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_observation.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/hanabi_observation.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_observation.cc > CMakeFiles/hanabi.dir/hanabi_observation.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_observation.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/hanabi_observation.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_observation.cc -o CMakeFiles/hanabi.dir/hanabi_observation.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_state.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/hanabi_state.cc.o: ../hanabi_lib/hanabi_state.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/hanabi_state.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/hanabi_state.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_state.cc

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_state.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/hanabi_state.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_state.cc > CMakeFiles/hanabi.dir/hanabi_state.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/hanabi_state.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/hanabi_state.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/hanabi_state.cc -o CMakeFiles/hanabi.dir/hanabi_state.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/util.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/util.cc.o: ../hanabi_lib/util.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/util.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/util.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/util.cc

hanabi_lib/CMakeFiles/hanabi.dir/util.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/util.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/util.cc > CMakeFiles/hanabi.dir/util.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/util.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/util.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/util.cc -o CMakeFiles/hanabi.dir/util.cc.s

hanabi_lib/CMakeFiles/hanabi.dir/canonical_encoders.cc.o: hanabi_lib/CMakeFiles/hanabi.dir/flags.make
hanabi_lib/CMakeFiles/hanabi.dir/canonical_encoders.cc.o: ../hanabi_lib/canonical_encoders.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object hanabi_lib/CMakeFiles/hanabi.dir/canonical_encoders.cc.o"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hanabi.dir/canonical_encoders.cc.o -c /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/canonical_encoders.cc

hanabi_lib/CMakeFiles/hanabi.dir/canonical_encoders.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hanabi.dir/canonical_encoders.cc.i"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/canonical_encoders.cc > CMakeFiles/hanabi.dir/canonical_encoders.cc.i

hanabi_lib/CMakeFiles/hanabi.dir/canonical_encoders.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hanabi.dir/canonical_encoders.cc.s"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && /share/apps/spack/envs/fgci-centos7-generic/software/gcc/9.2.0/dnrscms/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib/canonical_encoders.cc -o CMakeFiles/hanabi.dir/canonical_encoders.cc.s

# Object files for target hanabi
hanabi_OBJECTS = \
"CMakeFiles/hanabi.dir/hanabi_card.cc.o" \
"CMakeFiles/hanabi.dir/hanabi_game.cc.o" \
"CMakeFiles/hanabi.dir/hanabi_hand.cc.o" \
"CMakeFiles/hanabi.dir/hanabi_history_item.cc.o" \
"CMakeFiles/hanabi.dir/hanabi_move.cc.o" \
"CMakeFiles/hanabi.dir/hanabi_observation.cc.o" \
"CMakeFiles/hanabi.dir/hanabi_state.cc.o" \
"CMakeFiles/hanabi.dir/util.cc.o" \
"CMakeFiles/hanabi.dir/canonical_encoders.cc.o"

# External object files for target hanabi
hanabi_EXTERNAL_OBJECTS =

hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/hanabi_card.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/hanabi_game.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/hanabi_hand.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/hanabi_history_item.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/hanabi_move.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/hanabi_observation.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/hanabi_state.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/util.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/canonical_encoders.cc.o
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/build.make
hanabi_lib/libhanabi.a: hanabi_lib/CMakeFiles/hanabi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX static library libhanabi.a"
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && $(CMAKE_COMMAND) -P CMakeFiles/hanabi.dir/cmake_clean_target.cmake
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hanabi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
hanabi_lib/CMakeFiles/hanabi.dir/build: hanabi_lib/libhanabi.a

.PHONY : hanabi_lib/CMakeFiles/hanabi.dir/build

hanabi_lib/CMakeFiles/hanabi.dir/clean:
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib && $(CMAKE_COMMAND) -P CMakeFiles/hanabi.dir/cmake_clean.cmake
.PHONY : hanabi_lib/CMakeFiles/hanabi.dir/clean

hanabi_lib/CMakeFiles/hanabi.dir/depend:
	cd /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/hanabi_lib /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib /scratch/work/liz23/BackPropagationThroughAgents/bta/envs/hanabi/build/hanabi_lib/CMakeFiles/hanabi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : hanabi_lib/CMakeFiles/hanabi.dir/depend
