cmake_minimum_required(VERSION 3.6)

project(int32_addition_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Linker flags to allow undefined shared libs
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/header/)
set(RKNN_RT_LIB ${RKNN_API_PATH}/librknnrt.so)
include_directories(${RKNN_API_PATH})

# Add libdrm include path
include_directories(/usr/include/libdrm)
include_directories(/usr/include)

set(CMAKE_INSTALL_RPATH "lib")

# Build 'int32add' demo
add_executable(int32add int32add.cpp)
target_link_libraries(int32add ${RKNN_RT_LIB})

# Install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/int32_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS int32add DESTINATION ./)

# Install libraries
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)