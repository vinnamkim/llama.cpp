cmake_minimum_required(VERSION 3.22.2)
project(ggml-htp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

set(HTP_LIB ggml-htp-${DSP_VERSION})

add_library(${HTP_LIB} SHARED
    main.c
    htp_iface_skel.c
    work-queue.c
    dma-queue.c
    hmx-queue.c
    htp-tensor.c
    matmul-ops.c
    hmx-mm-cpp.cc
    flash-attn-ops.c
    gated-delta-net-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    act-ops.c
    rope-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
    gather-rows.c
    repeat-ops.c
    ssm-conv.c
    cumsum-ops.c
    fill-ops.c
    concat-ops.c
    diag-ops.c
    solve-tri-ops.c
    pad-ops.c
    argsort-ops.c
    im2col-ops.c
    roll-ops.c
    pool-ops.c
    allreduce-ops.c
)

# C++ sources use the .cc suffix: hexagon_fun.cmake globs *.cpp and then ships libc++ files this SDK lacks.
# The C++ kernels are freestanding (no exceptions, RTTI or libc++) so the plain C link line still works.
target_compile_options(${HTP_LIB} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++20 -fno-exceptions -fno-rtti -fno-threadsafe-statics>)

target_compile_definitions(${HTP_LIB} PRIVATE
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>)


build_idl(htp_iface.idl ${HTP_LIB})

set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

install(TARGETS ${HTP_LIB})

# per-op cycle breakdown of the C++ HMX matmul pipeline, printed with FARF (needs a <process>.farf file on the target)
option(GGML_HEXAGON_HMX_MM_STATS "ggml-hexagon: print hmx-mm pipeline statistics" OFF)
if (GGML_HEXAGON_HMX_MM_STATS)
    target_compile_definitions(${HTP_LIB} PRIVATE HMX_MM_STATS=1)
endif()
