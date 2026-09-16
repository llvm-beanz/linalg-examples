cmake_minimum_required(VERSION 3.24)
project(linalg_sin_example LANGUAGES CXX)

set(DXC_VERSION "1.10.2605.37-preview" CACHE STRING
  "Microsoft.Direct3D.DXC package version")
set(AGILITY_SDK_VERSION "1.721.3-preview" CACHE STRING
  "Microsoft.Direct3D.D3D12 package version")
set(WARP_VERSION "1.65535.20-preview" CACHE STRING
  "Microsoft.Direct3D.WARP package version")
option(LINALG_USE_PREVIEW_HEADERS
       "Build the D3D12 linear algebra capability query with preview headers" OFF)

set(NUGET_SOURCE "https://packagefeedproxy.microsoft.io/nuget/v3/index.json"
    CACHE STRING "NuGet package source")
set(NUGET_PACKAGES_DIR "${CMAKE_BINARY_DIR}/packages" CACHE PATH
    "NuGet package installation directory")
set(DXC_ROOT "${NUGET_PACKAGES_DIR}/Microsoft.Direct3D.DXC.${DXC_VERSION}")
set(AGILITY_SDK_ROOT
    "${NUGET_PACKAGES_DIR}/Microsoft.Direct3D.D3D12.${AGILITY_SDK_VERSION}")
set(WARP_ROOT
    "${NUGET_PACKAGES_DIR}/Microsoft.Direct3D.WARP.${WARP_VERSION}")
find_program(NUGET_EXE nuget.exe HINTS "${CMAKE_BINARY_DIR}/nuget")
if(NOT NUGET_EXE)
  message(STATUS "Downloading nuget.exe")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/nuget")
  file(DOWNLOAD
    "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
    "${CMAKE_BINARY_DIR}/nuget/nuget.exe"
    TLS_VERIFY ON)
  find_program(NUGET_EXE nuget.exe REQUIRED
    HINTS "${CMAKE_BINARY_DIR}/nuget")
endif()

function(install_nuget_package package version marker)
  set(package_root "${NUGET_PACKAGES_DIR}/${package}.${version}")
  if(NOT EXISTS "${package_root}/${marker}")
    message(STATUS "Installing ${package} ${version} from ${NUGET_SOURCE}")
    execute_process(
      COMMAND "${NUGET_EXE}" install -ForceEnglishOutput -NonInteractive
              "${package}" -Version "${version}"
              -Source "${NUGET_SOURCE}"
              -OutputDirectory "${NUGET_PACKAGES_DIR}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE nuget_output
      ERROR_VARIABLE nuget_output)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR
        "nuget install ${package} ${version} failed with exit code ${result}:\n${nuget_output}")
    endif()
    if(NOT EXISTS "${package_root}/${marker}")
      message(FATAL_ERROR
        "nuget installed ${package} ${version}, but ${marker} was not found")
    endif()
  endif()
endfunction()

install_nuget_package(Microsoft.Direct3D.DXC "${DXC_VERSION}"
  "build/native/Microsoft.Direct3D.DXC.targets")
install_nuget_package(Microsoft.Direct3D.D3D12 "${AGILITY_SDK_VERSION}"
  "build/native/Microsoft.Direct3D.D3D12.targets")
install_nuget_package(Microsoft.Direct3D.WARP "${WARP_VERSION}"
  "build/native/Microsoft.Direct3D.WARP.targets")

if(NOT EXISTS "${DXC_ROOT}/build/native/Microsoft.Direct3D.DXC.targets")
  message(FATAL_ERROR "DXC_ROOT does not contain the requested DXC package")
endif()
if(NOT EXISTS "${AGILITY_SDK_ROOT}/build/native/Microsoft.Direct3D.D3D12.targets")
  message(FATAL_ERROR "AGILITY_SDK_ROOT does not contain the requested Agility SDK package")
endif()
if(NOT EXISTS "${WARP_ROOT}/build/native/Microsoft.Direct3D.WARP.targets")
  message(FATAL_ERROR "WARP_ROOT does not contain the requested WARP package")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
  set(WARP_PLATFORM arm64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(WARP_PLATFORM x64)
else()
  message(FATAL_ERROR "The preview DXC package supports only ARM64 and x64")
endif()
set(WARP_DLL "${WARP_ROOT}/build/native/bin/${WARP_PLATFORM}/d3d10warp.dll")
set(AGILITY_SDK_BIN "${AGILITY_SDK_ROOT}/build/native/bin/${WARP_PLATFORM}")
set(DXC_NATIVE_ROOT "${DXC_ROOT}/build/native")

add_executable(sin-network sin-network.cpp)
target_compile_features(sin-network PRIVATE cxx_std_20)
target_compile_definitions(sin-network PRIVATE
  LINALG_USE_PREVIEW_HEADERS=$<BOOL:${LINALG_USE_PREVIEW_HEADERS}>)
target_include_directories(sin-network BEFORE PRIVATE
  "${AGILITY_SDK_ROOT}/build/native/include")
target_include_directories(sin-network PRIVATE "${DXC_NATIVE_ROOT}/include")
target_link_libraries(sin-network PRIVATE
  d3d12 dxgi dxguid
  "${DXC_NATIVE_ROOT}/lib/${WARP_PLATFORM}/dxcompiler.lib")

add_custom_command(TARGET sin-network POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
    "$<TARGET_FILE_DIR:sin-network>/D3D12"
  COMMAND ${CMAKE_COMMAND} -E make_directory
    "$<TARGET_FILE_DIR:sin-network>/dx"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${AGILITY_SDK_BIN}/D3D12Core.dll"
    "$<TARGET_FILE_DIR:sin-network>/D3D12"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${AGILITY_SDK_BIN}/D3D12SDKLayers.dll"
    "$<TARGET_FILE_DIR:sin-network>/D3D12"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${WARP_DLL}" "$<TARGET_FILE_DIR:sin-network>/D3D12"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DXC_NATIVE_ROOT}/bin/${WARP_PLATFORM}/dxcompiler.dll"
      "$<TARGET_FILE_DIR:sin-network>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DXC_NATIVE_ROOT}/bin/${WARP_PLATFORM}/dxil.dll"
      "$<TARGET_FILE_DIR:sin-network>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${WARP_DLL}" "$<TARGET_FILE_DIR:sin-network>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/sin-network.hlsl"
      "$<TARGET_FILE_DIR:sin-network>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DXC_NATIVE_ROOT}/include/hlsl/dx/linalg.h"
      "$<TARGET_FILE_DIR:sin-network>/dx")