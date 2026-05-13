cmake_minimum_required(VERSION 3.15)
project(mod_vm_detect
        VERSION 1.0.0
        DESCRIPTION "wav2vec2 voicemail detection module for FreeSWITCH"
        HOMEPAGE_URL "https://github.com/eschmidbauer/mod_vm_detect"
        LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
include(GNUInstallDirs)

option(ENABLE_LOCAL            "Enable local compile/debug (adds /usr/local FreeSWITCH paths)" OFF)
option(BUILD_WAV2VEC_TESTS     "Build vendored wav2vec test/bench binaries" OFF)
option(BUILD_WAV2VEC_CLI       "Build the standalone wav2vec classifier CLI (vm_detect_cli)" OFF)

if(ENABLE_LOCAL)
    set(ENV{PKG_CONFIG_PATH} "/usr/local/freeswitch/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(FreeSWITCH REQUIRED IMPORTED_TARGET freeswitch)
pkg_get_variable(FS_MOD_DIR freeswitch modulesdir)
message(STATUS "FreeSWITCH modules dir: ${FS_MOD_DIR}")

# ── SIMD feature detection (mirrors wav2vec-vm-finetune-c/c/Makefile) ───────

set(WAV2VEC_SIMD_FLAGS "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_arm_arch "armv8.2-a+fp16+dotprod")
    set(_has_i8mm FALSE)

    if(APPLE)
        execute_process(
            COMMAND sysctl -n hw.optional.arm.FEAT_I8MM
            OUTPUT_VARIABLE _i8mm_out
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_i8mm_out STREQUAL "1")
            set(_has_i8mm TRUE)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        execute_process(
            COMMAND grep -m1 i8mm /proc/cpuinfo
            OUTPUT_QUIET ERROR_QUIET
            RESULT_VARIABLE _grep_rc)
        if(_grep_rc EQUAL 0)
            set(_has_i8mm TRUE)
        endif()
    endif()

    if(_has_i8mm)
        set(_arm_arch "${_arm_arch}+i8mm")
        message(STATUS "wav2vec: ARM i8mm (SMMLA) detected")
    else()
        message(STATUS "wav2vec: ARM dotprod baseline (no i8mm)")
    endif()

    list(APPEND WAV2VEC_SIMD_FLAGS "-march=${_arm_arch}")

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    if(APPLE)
        # macOS x86 hosts are Haswell+ (AVX2/FMA). VNNI not available.
        list(APPEND WAV2VEC_SIMD_FLAGS "-mavx2" "-mfma")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        execute_process(
            COMMAND grep -m1 -E " (avx2|fma)" /proc/cpuinfo
            OUTPUT_QUIET ERROR_QUIET
            RESULT_VARIABLE _avx2_rc)
        if(_avx2_rc EQUAL 0)
            list(APPEND WAV2VEC_SIMD_FLAGS "-mavx2" "-mfma")
            message(STATUS "wav2vec: x86 AVX2+FMA detected")
        endif()

        execute_process(
            COMMAND grep -m1 -E " (avx_vnni|avx512_vnni)" /proc/cpuinfo
            OUTPUT_QUIET ERROR_QUIET
            RESULT_VARIABLE _vnni_rc)
        if(_vnni_rc EQUAL 0)
            list(APPEND WAV2VEC_SIMD_FLAGS "-mavxvnni")
            message(STATUS "wav2vec: x86 AVX-VNNI detected")
        endif()
    endif()
endif()

# ── Common settings ────────────────────────────────────────────────────────

set(WAV2VEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/wav2vec)
set(WAV2VEC_COMPILE_OPTS -O3 -ffast-math -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter ${WAV2VEC_SIMD_FLAGS})

# ── mod_vm_detect.so (the FreeSWITCH module) ───────────────────────────────

add_library(mod_vm_detect SHARED
    mod_vm_detect.c
    mod_vm_detect.h
    fireredvad.c
    fireredvad.h
    ${WAV2VEC_DIR}/wav2vec_vm.c
    ${WAV2VEC_DIR}/wav2vec_vm.h
    ${WAV2VEC_DIR}/sgemm.c
    ${WAV2VEC_DIR}/sgemm.h)

target_include_directories(mod_vm_detect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${WAV2VEC_DIR})

target_compile_options(mod_vm_detect PRIVATE ${WAV2VEC_COMPILE_OPTS})

target_link_libraries(mod_vm_detect PRIVATE
    PkgConfig::FreeSWITCH
    speexdsp
    pthread
    m)

set_target_properties(mod_vm_detect PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON)

install(TARGETS mod_vm_detect
        COMPONENT mod_vm_detect
        DESTINATION ${FS_MOD_DIR})

pkg_get_variable(FS_CONF_DIR freeswitch confdir)
if(FS_CONF_DIR)
    install(FILES vm_detect.conf.xml
            DESTINATION ${FS_CONF_DIR}/autoload_configs
            COMPONENT mod_vm_detect)
endif()

# ── Optional: vendored test/bench binaries ─────────────────────────────────

if(BUILD_WAV2VEC_TESTS)
    add_executable(test_concurrent
        ${WAV2VEC_DIR}/test_concurrent.c
        ${WAV2VEC_DIR}/wav2vec_vm.c
        ${WAV2VEC_DIR}/sgemm.c)
    target_include_directories(test_concurrent PRIVATE ${WAV2VEC_DIR})
    target_compile_options(test_concurrent PRIVATE ${WAV2VEC_COMPILE_OPTS})
    target_link_libraries(test_concurrent PRIVATE pthread m)

    add_executable(bench_concurrent
        ${WAV2VEC_DIR}/bench_concurrent.c
        ${WAV2VEC_DIR}/wav2vec_vm.c
        ${WAV2VEC_DIR}/sgemm.c)
    target_include_directories(bench_concurrent PRIVATE ${WAV2VEC_DIR})
    target_compile_options(bench_concurrent PRIVATE ${WAV2VEC_COMPILE_OPTS})
    target_link_libraries(bench_concurrent PRIVATE pthread m)
endif()

if(BUILD_WAV2VEC_CLI)
    add_executable(vm_detect_cli
        ${WAV2VEC_DIR}/main.c
        ${WAV2VEC_DIR}/wav2vec_vm.c
        ${WAV2VEC_DIR}/sgemm.c)
    target_include_directories(vm_detect_cli PRIVATE ${WAV2VEC_DIR})
    target_compile_options(vm_detect_cli PRIVATE ${WAV2VEC_COMPILE_OPTS})
    target_link_libraries(vm_detect_cli PRIVATE pthread m)
endif()

# ── Packaging ──────────────────────────────────────────────────────────────

set(CPACK_COMPONENTS_ALL mod_vm_detect)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libspeexdsp1, libfreeswitch1")
set(CPACK_PACKAGE_NAME "mod-vm-detect")
message(STATUS "Components to pack: ${CPACK_COMPONENTS_ALL}")
