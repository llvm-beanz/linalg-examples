#define NOMINMAX
#include <windows.h>

#include <d3d12.h>
#include <d3d12shader.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
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
constexpr UINT kSampleCount = 257;
constexpr UINT kHiddenCount = 16;
constexpr UINT64 kHiddenBufferSize = 288;
constexpr float kPi = 3.14159265358979323846f;

constexpr std::array<float, kHiddenCount> kSlopes = {
    0.25f,         0.294921875f, 0.34814453125f, 0.410888671875f,
    0.48486328125f, 0.572265625f, 0.67529296875f, 0.79736328125f,
    0.94091796875f, 1.1103515625f, 1.310546875f, 1.546875f,
    1.8251953125f, 2.154296875f,  2.54296875f,   3.0f,
};

constexpr std::array<float, kHiddenCount + 1> kOutputWeights = {
    -8.81840503277f, -5.18517281839f, -0.85500787976f, 3.06444520451f,
    5.20010106434f,  4.66269561419f,  1.9101763575f,   -1.17763229372f,
    -2.4721684571f,  -1.40283863383f, 0.578348160398f, 1.407665367f,
    0.503075230007f, -0.704164785806f, -0.522221657915f, 0.372038452716f,
    0.0f,
};

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
  DWORD executablePathLength = GetModuleFileNameW(
      nullptr, executablePath.data(), static_cast<DWORD>(executablePath.size()));
  if (executablePathLength == 0 || executablePathLength == executablePath.size())
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

ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device *device, UINT64 size,
                                    D3D12_HEAP_TYPE heapType,
                                    D3D12_RESOURCE_STATES state,
                                    D3D12_RESOURCE_FLAGS flags =
                                        D3D12_RESOURCE_FLAG_NONE) {
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

template <typename T>
ComPtr<ID3D12Resource> CreateUploadBuffer(ID3D12Device *device,
                                         const std::vector<T> &data) {
  auto buffer = CreateBuffer(device, data.size() * sizeof(T),
                             D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
  void *mapped = nullptr;
  Check(buffer->Map(0, nullptr, &mapped), "Map upload buffer");
  memcpy(mapped, data.data(), data.size() * sizeof(T));
  buffer->Unmap(0, nullptr);
  return buffer;
}

ComPtr<ID3D12Device> CreateDevice() {
  std::cerr << "Enabling D3D12 experimental shader models...\n";
  Check(D3D12EnableExperimentalFeatures(1, &D3D12ExperimentalShaderModels,
                                        nullptr, nullptr),
        "Enable experimental shader models");
  std::cerr << "D3D12 experimental shader models enabled.\n";
  ComPtr<ID3D12Debug> debug;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
    debug->EnableDebugLayer();

  ComPtr<IDXGIFactory6> factory;
  std::cerr << "Creating DXGI factory...\n";
  Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Create DXGI factory");
  std::cerr << "DXGI factory created.\n";
  for (UINT index = 0;; ++index) {
    ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapterByGpuPreference(
            index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND)
      break;
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
      std::wcerr << L"Trying hardware adapter: " << desc.Description << L"\n";
      ComPtr<ID3D12Device> device;
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                      IID_PPV_ARGS(&device))))
        return device;
    }
  }
  ComPtr<IDXGIAdapter> warp;
  Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "Find WARP adapter");
  std::cerr << "Trying WARP adapter...\n";
  ComPtr<ID3D12Device> device;
  Check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_12_0,
                          IID_PPV_ARGS(&device)),
        "Create WARP device");
  return device;
}

void ReportLinearAlgebraSupport(ID3D12Device *device) {
#if LINALG_USE_PREVIEW_HEADERS
  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT support{};
  std::cerr << "Querying linear algebra tier...\n";
  Check(device->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT,
                                    &support, sizeof(support)),
        "Query linear algebra support");
  std::cerr << "Linear algebra tier query completed.\n";
  if (support.LinearAlgebraTier < D3D12_LINEAR_ALGEBRA_TIER_1_0)
    throw std::runtime_error("The adapter does not support linear algebra tier 1");

  D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT operation{};
  operation.OperationType =
      D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY;
  auto &inputs = operation.ThreadVectorMatrixMultiply;
  inputs.VectorInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
  inputs.MatrixInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
  inputs.BiasInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
  inputs.VectorResultType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
  std::cerr << "Querying FP16 vector-matrix support...\n";
  Check(device->CheckFeatureSupport(
        D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
        &operation, sizeof(operation)),
        "Query FP16 vector-matrix support");
  std::cerr << "FP16 vector-matrix support query completed.\n";
  if ((inputs.SupportFlags &
       D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) == 0)
    throw std::runtime_error("FP16 vector-matrix MultiplyAdd is not supported");
  std::cout << "Linear algebra tier 1 and FP16 MultiplyAdd are supported.\n";
#else
  (void)device;
  std::cout << "Built without D3D12 preview headers; capability query skipped.\n";
#endif
}

ComPtr<ID3D12RootSignature> CreateRootSignature(ID3D12Device *device) {
  D3D12_DESCRIPTOR_RANGE ranges[2]{};
  ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
  ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 3};
  D3D12_ROOT_PARAMETER parameters[2]{};
  parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  parameters[0].DescriptorTable = {2, ranges};
  parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  parameters[1].Constants = {0, 0, 1};
  parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC desc{2, parameters, 0, nullptr,
                                 D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ComPtr<ID3DBlob> blob;
  ComPtr<ID3DBlob> errors;
  Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                    &blob, &errors),
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
      argc > 1 ? argv[1] : std::filesystem::path(L"sin-network.hlsl");
  auto shader = CompileShader(shaderPath);
  auto device = CreateDevice();
  ReportLinearAlgebraSupport(device.Get());
  std::cerr << "Creating network resources...\n";

  std::vector<uint16_t> hidden(kHiddenBufferSize / sizeof(uint16_t), 0);
  for (UINT row = 0; row < kHiddenCount; ++row) {
    hidden[row * 8] = FloatToHalf(kSlopes[row]);
    hidden[row * 8 + 3] = FloatToHalf(0.0f);
  }
  std::vector<float> inputs(kSampleCount);
  for (UINT i = 0; i < kSampleCount; ++i)
    inputs[i] = -kPi + 2.0f * kPi * i / (kSampleCount - 1);
  std::vector<float> outputWeights(kOutputWeights.begin(), kOutputWeights.end());

  auto hiddenBuffer = CreateUploadBuffer(device.Get(), hidden);
  auto inputBuffer = CreateUploadBuffer(device.Get(), inputs);
  auto outputWeightBuffer = CreateUploadBuffer(device.Get(), outputWeights);
  auto outputBuffer = CreateBuffer(
      device.Get(), kSampleCount * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
      D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto readback = CreateBuffer(device.Get(), kSampleCount * sizeof(float),
                               D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_STATE_COPY_DEST);
  std::cerr << "Network resources created.\n";

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
  rawSrv.Buffer.NumElements = static_cast<UINT>(kHiddenBufferSize / 4);
  rawSrv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
  device->CreateShaderResourceView(hiddenBuffer.Get(), &rawSrv, cpu);
  cpu.ptr += descriptorSize;

  D3D12_SHADER_RESOURCE_VIEW_DESC structuredSrv{};
  structuredSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  structuredSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  structuredSrv.Buffer.StructureByteStride = sizeof(float);
  structuredSrv.Buffer.NumElements = kSampleCount;
  device->CreateShaderResourceView(inputBuffer.Get(), &structuredSrv, cpu);
  cpu.ptr += descriptorSize;
  structuredSrv.Buffer.NumElements = static_cast<UINT>(outputWeights.size());
  device->CreateShaderResourceView(outputWeightBuffer.Get(), &structuredSrv,
                                   cpu);
  cpu.ptr += descriptorSize;

  D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
  uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uav.Buffer.NumElements = kSampleCount;
  uav.Buffer.StructureByteStride = sizeof(float);
  device->CreateUnorderedAccessView(outputBuffer.Get(), nullptr, &uav, cpu);

  std::cerr << "Creating root signature...\n";
  auto rootSignature = CreateRootSignature(device.Get());
  std::cerr << "Root signature created.\nCreating compute pipeline...\n";
  D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
  psoDesc.pRootSignature = rootSignature.Get();
  psoDesc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
  ComPtr<ID3D12PipelineState> pipeline;
  Check(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline)),
      "Create compute pipeline (the runtime/driver may not support SM 6.10 "
      "linear algebra yet)");
  std::cerr << "Compute pipeline created.\n";

  std::cerr << "Recording compute commands...\n";
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

  D3D12_RESOURCE_BARRIER toUav{};
  toUav.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  toUav.Transition.pResource = outputBuffer.Get();
  toUav.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  toUav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  toUav.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  commands->ResourceBarrier(1, &toUav);
  ID3D12DescriptorHeap *heaps[] = {descriptors.Get()};
  commands->SetDescriptorHeaps(1, heaps);
  commands->SetComputeRootSignature(rootSignature.Get());
  commands->SetComputeRootDescriptorTable(
      0, descriptors->GetGPUDescriptorHandleForHeapStart());
  commands->SetComputeRoot32BitConstant(1, kSampleCount, 0);
  commands->Dispatch((kSampleCount + 63) / 64, 1, 1);

  std::swap(toUav.Transition.StateBefore, toUav.Transition.StateAfter);
  toUav.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  commands->ResourceBarrier(1, &toUav);
  commands->CopyResource(readback.Get(), outputBuffer.Get());
  Check(commands->Close(), "Close command list");
  std::cerr << "Submitting compute commands...\n";
  ID3D12CommandList *lists[] = {commands.Get()};
  queue->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
        "Create fence");
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!event)
    throw std::runtime_error("CreateEvent failed");
  std::cerr << "Waiting for compute completion...\n";
  WaitForGpu(queue.Get(), fence.Get(), event);
  std::cerr << "Compute completed.\n";
  CloseHandle(event);

  std::cerr << "Reading results...\n";
  float *results = nullptr;
  D3D12_RANGE readRange{0, kSampleCount * sizeof(float)};
  Check(readback->Map(0, &readRange, reinterpret_cast<void **>(&results)),
        "Map readback buffer");
  double squaredError = 0.0;
  float maxError = 0.0f;
  for (UINT i = 0; i < kSampleCount; ++i) {
    float error = results[i] - std::sin(inputs[i]);
    squaredError += error * error;
    maxError = std::max(maxError, std::abs(error));
  }
  std::cout << "Evaluated " << kSampleCount << " samples on [-pi, pi]\n"
            << "max error: " << maxError << "\n"
            << "RMS error: " << std::sqrt(squaredError / kSampleCount) << "\n";
  readback->Unmap(0, nullptr);
  return 0;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}