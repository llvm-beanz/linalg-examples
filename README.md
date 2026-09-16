# HLSL Linear Algebra Examples

This repository contains standalone HLSL matrix multiplication shaders and
native D3D12 linear algebra samples.

## Implementation Examples

This repository demonstrates different approaches based on feature availability:

- **[Standalone shaders](standalone-shaders/README.md)**: three GEMM
  implementations using groupshared memory, wave-scope matrix operations, and
  threadgroup-scope matrix operations.
- **[sin-network](sin-network/README.md)**: a complete native D3D12 sample that
  approximates `sin(x)` with a 16-neuron network and thread-scope matrix-vector
  multiplication. Its README contains build instructions and a walkthrough of
  the C++ host, HLSL shader, resource layout, and dispatch flow.
- **[xor-network](xor-network/README.md)**: a complete native D3D12 sample that
  classifies the four XOR inputs as one batch with a `2 -> 4 -> 1` network,
  compares FP16 threadgroup matrix multiplication with a wave-tiled GEMM
  implementation, and verifies both output matrices agree.
