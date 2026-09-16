cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(linalg_xor_example LANGUAGES CXX)
endif()

include(../cmake/LinalgSample.cmake)

linalg_add_sample(xor-network xor-network.cpp
  xor-network.hlsl xor-network-wave.hlsl)
