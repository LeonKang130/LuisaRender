cmake_minimum_required(VERSION 3.18...3.21)
cmake_policy(VERSION 3.18)

cmake_policy(SET CMP0069 NEW)
cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0091 NEW)

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS ON)

project(LuisaRender LANGUAGES C CXX VERSION 0.1)

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "LuisaRender only supports 64-bit platforms")
endif ()

if (NOT DEFINED LUISA_RENDER_MASTER_PROJECT)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(LUISA_RENDER_MASTER_PROJECT ON)
    else ()
        set(LUISA_RENDER_MASTER_PROJECT OFF)
    endif ()
endif ()

option(LUISA_RENDER_BUILD_TESTS "Build tests for LuisaRender" ${LUISA_RENDER_MASTER_PROJECT})
option(LUISA_RENDER_ENABLE_UNITY_BUILD "Enable unity build to speed up compilation" OFF)

include(src/compute/scripts/setup_output_dirs.cmake)

# rpath
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../bin;@loader_path/../lib")
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../bin;$ORIGIN/../lib")
endif ()

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "dist")
endif ()

# set up install directories
include(GNUInstallDirs)
if (NOT WIN32)
    # DLLs are installed in the same directory as executables
    set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_BINDIR})
endif ()

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

add_subdirectory(src)
