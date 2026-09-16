#define NOMINMAX
#include <windows.h>

#include <d3d12.h>
#include <d3d12shader.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LINALG_USE_PREVIEW_HEADERS
#define LINALG_USE_PREVIEW_HEADERS 0
#endif

extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion =
    D3D12_PREVIEW_SDK_VERSION;
__declspec(dllexport) extern const char *D3D12SDKPath = ".\\D3D12\\";
}

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT kInputCount = 4;
constexpr UINT kMatrixDimension = 4;
constexpr UINT64 kHalfMatrixSize =
  kMatrixDimension * kMatrixDimension * sizeof(uint16_t);
constexpr UINT64 kOutputBufferSize = kHalfMatrixSize;

struct Input {
  float x;
  float y;
};

constexpr std::array<Input, kInputCount> kInputs = {
    Input{0.0f, 0.0f}, Input{0.0f, 1.0f}, Input{1.0f, 0.0f},
    Input{1.0f, 1.0f}};
constexpr std::array<UINT, kInputCount> kExpected = {0, 1, 1, 0};
constexpr std::array<std::array<float, 4>, kMatrixDimension> kHiddenWeights = {
    std::array<float, 4>{20.0f, 20.0f, 0.0f, 0.0f},
    std::array<float, 4>{20.0f, 20.0f, 0.0f, 0.0f},
    std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f},
  std::array<float, 4>{-10.0f, -30.0f, 0.0f, 0.0f}};
constexpr std::array<std::array<float, 4>, kMatrixDimension> kOutputWeights = {
  std::array<float, 4>{20.0f, 0.0f, 0.0f, 0.0f},
  std::array<float, 4>{-20.0f, 0.0f, 0.0f, 0.0f},
  std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f},
  std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}};

void Check(HRESULT result, const char *operation) {
  if (FAILED(result)) {
    std::ostringstream message;
    message << operation << " failed: 0x" << std::hex
            << static_cast<unsigned>(result);
    throw std::runtime_error(message.str());
  }
}

uint16_t FloatToHalf(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  uint32_t sign = (bits >> 16) & 0x8000;
  int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000) >> 13));
  }
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00);
  return static_cast<uint16_t>(sign | (exponent << 10) |
                               ((mantissa + 0x1000) >> 13));
}

float HalfToFloat(uint16_t value) {
  uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x03ff;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | (exponent << 23) | ((mantissa & 0x03ff) << 13);
    }
  } else if (exponent == 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

ComPtr<IDxcBlob> CompileShader(const std::filesystem::path &path) {
  ComPtr<IDxcUtils> utils;
  ComPtr<IDxcCompiler3> compiler;
  Check(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils)),
        "Create DXC utils");
  Check(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)),
        "Create DXC compiler");

  ComPtr<IDxcBlobEncoding> source;
  Check(utils->LoadFile(path.c_str(), nullptr, &source), "Load shader");
  DxcBuffer sourceBuffer{source->GetBufferPointer(), source->GetBufferSize(),
                         DXC_CP_ACP};
  std::array<wchar_t, MAX_PATH> executablePath{};
  DWORD pathLength = GetModuleFileNameW(
      nullptr, executablePath.data(), static_cast<DWORD>(executablePath.size()));
  if (pathLength == 0 || pathLength == executablePath.size())
    throw std::runtime_error("Get executable path failed");
  std::wstring includePath =
      std::filesystem::path(executablePath.data()).parent_path().wstring();
  const wchar_t *arguments[] = {L"-E", L"main", L"-T", L"cs_6_10",
                                L"-enable-16bit-types", L"-I",
                                includePath.c_str()};
  ComPtr<IDxcIncludeHandler> includes;
  Check(utils->CreateDefaultIncludeHandler(&includes), "Create include handler");
  ComPtr<IDxcResult> result;
  Check(compiler->Compile(&sourceBuffer, arguments, std::size(arguments),
                          includes.Get(), IID_PPV_ARGS(&result)),
        "Compile shader");

  ComPtr<IDxcBlobUtf8> errors;
  result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
  if (errors && errors->GetStringLength())
    std::cerr << errors->GetStringPointer();
  HRESULT status;
  Check(result->GetStatus(&status), "Get shader status");
  Check(status, "Shader compilation");

  ComPtr<IDxcBlob> shader;
  Check(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader), nullptr),
        "Get shader object");
  return shader;
}

ComPtr<ID3D12Resource> CreateBuffer(
    ID3D12Device *device, UINT64 size, D3D12_HEAP_TYPE heapType,
    D3D12_RESOURCE_STATES state,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = heapType;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = flags;
  ComPtr<ID3D12Resource> resource;
  Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                        state, nullptr,
                                        IID_PPV_ARGS(&resource)),
        "Create buffer");
  return resource;
}

template <typename T, size_t Size>
ComPtr<ID3D12Resource> CreateUploadBuffer(ID3D12Device *device,
                                         const std::array<T, Size> &data) {
  auto buffer = CreateBuffer(device, sizeof(data), D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void *mapped = nullptr;
  Check(buffer->Map(0, nullptr, &mapped), "Map upload buffer");
  memcpy(mapped, data.data(), sizeof(data));
  buffer->Unmap(0, nullptr);
  return buffer;
}

ComPtr<ID3D12Device> CreateWarpDevice() {
  Check(D3D12EnableExperimentalFeatures(1, &D3D12ExperimentalShaderModels,
                                        nullptr, nullptr),
        "Enable experimental shader models");
  ComPtr<ID3D12Debug> debug;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
    debug->EnableDebugLayer();

  ComPtr<IDXGIFactory6> factory;
  Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Create DXGI factory");
  ComPtr<IDXGIAdapter> warp;
  Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "Find WARP adapter");
  ComPtr<ID3D12Device> device;
  Check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_12_0,
                          IID_PPV_ARGS(&device)),
        "Create WARP device");
  std::cout << "Using WARP for deterministic threadgroup matrix support.\n";
  return device;
}

void ReportLinearAlgebraSupport(ID3D12Device *device) {
#if LINALG_USE_PREVIEW_HEADERS
  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT support{};
  Check(device->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT,
                                    &support, sizeof(support)),
        "Query linear algebra support");
  if (support.LinearAlgebraTier < D3D12_LINEAR_ALGEBRA_TIER_1_0)
    throw std::runtime_error("The adapter does not support linear algebra tier 1");

  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT operation{};
  operation.OperationType =
      D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREADGROUP_MATRIX_MULTIPLY;
    auto &threadGroup = operation.ThreadGroupMatrixMultiply;
    threadGroup.WaveInputs.WaveSize = 4;
    threadGroup.WaveInputs.MatrixAComponentType =
      D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
    threadGroup.WaveInputs.MatrixBComponentType =
      D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
    threadGroup.WaveInputs.AccumulatorComponentType =
      D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
    threadGroup.Shape = {kMatrixDimension, kMatrixDimension, kMatrixDimension};
  Check(device->CheckFeatureSupport(
            D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
            &operation, sizeof(operation)),
      "Query FP16 threadgroup matrix-matrix support");
    if ((threadGroup.SupportFlags &
       D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) == 0)
    throw std::runtime_error(
      "FP16 threadgroup matrix-matrix Multiply is not supported");
    std::cout << "Linear algebra tier 1 and FP16 threadgroup Multiply are supported.\n";
#else
  (void)device;
  std::cout << "Built without D3D12 preview headers; capability query skipped.\n";
#endif
}

ComPtr<ID3D12RootSignature> CreateRootSignature(ID3D12Device *device) {
  D3D12_DESCRIPTOR_RANGE ranges[2]{};
  ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
  ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 3};
  D3D12_ROOT_PARAMETER parameters[1]{};
  parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  parameters[0].DescriptorTable = {2, ranges};
  parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC desc{1, parameters, 0, nullptr,
                                 D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ComPtr<ID3DBlob> blob;
  ComPtr<ID3DBlob> errors;
  Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob,
                                    &errors),
        "Serialize root signature");
  ComPtr<ID3D12RootSignature> rootSignature;
  Check(device->CreateRootSignature(0, blob->GetBufferPointer(),
                                    blob->GetBufferSize(),
                                    IID_PPV_ARGS(&rootSignature)),
        "Create root signature");
  return rootSignature;
}

void WaitForGpu(ID3D12CommandQueue *queue, ID3D12Fence *fence, HANDLE event) {
  Check(queue->Signal(fence, 1), "Signal queue");
  if (fence->GetCompletedValue() < 1) {
    Check(fence->SetEventOnCompletion(1, event), "Set fence event");
    WaitForSingleObject(event, INFINITE);
  }
}
} // namespace

int wmain(int argc, wchar_t **argv) try {
  const std::filesystem::path shaderPath =
      argc > 1 ? argv[1] : std::filesystem::path(L"xor-network.hlsl");
  auto shader = CompileShader(shaderPath);
  auto device = CreateWarpDevice();
  ReportLinearAlgebraSupport(device.Get());

  std::array<uint16_t, kMatrixDimension * kMatrixDimension> inputs{};
  std::array<uint16_t, kMatrixDimension * kMatrixDimension> hiddenWeights{};
  std::array<uint16_t, kMatrixDimension * kMatrixDimension> outputWeights{};
  for (UINT row = 0; row < kMatrixDimension; ++row) {
    inputs[row * kMatrixDimension] = FloatToHalf(kInputs[row].x);
    inputs[row * kMatrixDimension + 1] = FloatToHalf(kInputs[row].y);
    inputs[row * kMatrixDimension + 3] = FloatToHalf(1.0f);
    for (UINT column = 0; column < kMatrixDimension; ++column) {
      hiddenWeights[row * kMatrixDimension + column] =
          FloatToHalf(kHiddenWeights[row][column]);
      outputWeights[row * kMatrixDimension + column] =
          FloatToHalf(kOutputWeights[row][column]);
    }
  }

  auto inputBuffer = CreateUploadBuffer(device.Get(), inputs);
  auto hiddenBuffer = CreateUploadBuffer(device.Get(), hiddenWeights);
  auto outputWeightBuffer = CreateUploadBuffer(device.Get(), outputWeights);
  auto outputBuffer = CreateBuffer(
      device.Get(), kOutputBufferSize, D3D12_HEAP_TYPE_DEFAULT,
      D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto readback = CreateBuffer(device.Get(), kOutputBufferSize,
                               D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.NumDescriptors = 4;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ComPtr<ID3D12DescriptorHeap> descriptors;
  Check(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descriptors)),
        "Create descriptor heap");
  UINT descriptorSize = device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto cpu = descriptors->GetCPUDescriptorHandleForHeapStart();

  D3D12_SHADER_RESOURCE_VIEW_DESC rawSrv{};
  rawSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  rawSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  rawSrv.Format = DXGI_FORMAT_R32_TYPELESS;
  rawSrv.Buffer.NumElements = static_cast<UINT>(kHalfMatrixSize / 4);
  rawSrv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
  device->CreateShaderResourceView(inputBuffer.Get(), &rawSrv, cpu);
  cpu.ptr += descriptorSize;
  device->CreateShaderResourceView(hiddenBuffer.Get(), &rawSrv, cpu);
  cpu.ptr += descriptorSize;
  device->CreateShaderResourceView(outputWeightBuffer.Get(), &rawSrv, cpu);
  cpu.ptr += descriptorSize;

  D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
  uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uav.Format = DXGI_FORMAT_R32_TYPELESS;
  uav.Buffer.NumElements = static_cast<UINT>(kOutputBufferSize / 4);
  uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
  device->CreateUnorderedAccessView(outputBuffer.Get(), nullptr, &uav, cpu);

  auto rootSignature = CreateRootSignature(device.Get());
  D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
  psoDesc.pRootSignature = rootSignature.Get();
  psoDesc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
  ComPtr<ID3D12PipelineState> pipeline;
  Check(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline)),
        "Create compute pipeline");

  D3D12_COMMAND_QUEUE_DESC queueDesc{};
  ComPtr<ID3D12CommandQueue> queue;
  Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)),
        "Create command queue");
  ComPtr<ID3D12CommandAllocator> allocator;
  Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&allocator)),
        "Create command allocator");
  ComPtr<ID3D12GraphicsCommandList> commands;
  Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  allocator.Get(), pipeline.Get(),
                                  IID_PPV_ARGS(&commands)),
        "Create command list");

  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = outputBuffer.Get();
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  commands->ResourceBarrier(1, &barrier);
  ID3D12DescriptorHeap *heaps[] = {descriptors.Get()};
  commands->SetDescriptorHeaps(1, heaps);
  commands->SetComputeRootSignature(rootSignature.Get());
  commands->SetComputeRootDescriptorTable(
      0, descriptors->GetGPUDescriptorHandleForHeapStart());
  commands->Dispatch(1, 1, 1);

  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  commands->ResourceBarrier(1, &barrier);
  commands->CopyResource(readback.Get(), outputBuffer.Get());
  Check(commands->Close(), "Close command list");
  ID3D12CommandList *lists[] = {commands.Get()};
  queue->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
        "Create fence");
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!event)
    throw std::runtime_error("CreateEvent failed");
  WaitForGpu(queue.Get(), fence.Get(), event);
  CloseHandle(event);

  uint16_t *results = nullptr;
  D3D12_RANGE readRange{0, kOutputBufferSize};
  Check(readback->Map(0, &readRange, reinterpret_cast<void **>(&results)),
        "Map readback buffer");
  bool passed = true;
  std::cout << std::fixed << std::setprecision(6);
  for (UINT index = 0; index < kInputCount; ++index) {
    float probability = HalfToFloat(results[index * kMatrixDimension]);
    UINT predicted = probability >= 0.5f ? 1 : 0;
    passed &= predicted == kExpected[index];
    std::cout << static_cast<UINT>(kInputs[index].x) << " XOR "
              << static_cast<UINT>(kInputs[index].y) << " = " << predicted
              << " (probability " << probability << ")\n";
  }
  readback->Unmap(0, nullptr);
  if (!passed)
    throw std::runtime_error("XOR classification failed");
  std::cout << "All XOR cases classified correctly.\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
