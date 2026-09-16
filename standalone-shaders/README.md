# Standalone HLSL Linear Algebra Shaders

These shaders demonstrate different approaches to general matrix
multiplication (GEMM).

## What is GEMM?

GEMM performs the operation `C = alpha * A * B + beta * C`, where:

- `A` is an M x K matrix.
- `B` is a K x N matrix.
- `C` is an M x N matrix.
- `alpha` and `beta` are scalars.

Each element in the result is the dot product of a row from `A` and a column
from `B`:

```text
C[i,j] = sum(k=0 to K-1) A[i,k] * B[k,j]
```

The operation has O(M x N x K) computational complexity. The inner dimension
`K` must match between `A` and `B`.

## Implementations

- **[gemm.hlsl](gemm.hlsl)**: uses groupshared-memory tiles to improve data
  reuse.
- **[linalg-wave.hlsl](linalg-wave.hlsl)**: uses manually tiled, wave-scope
  `linalg::Matrix` objects.
- **[linalg-threadgroup.hlsl](linalg-threadgroup.hlsl)**: uses threadgroup-scope
  `linalg::Matrix` objects and lets the driver tile operations for the target
  hardware.

## Matrix Dimensions

The shaders define configurable `M`, `N`, and `K` dimensions:

- `M`: rows in `A` and `C`.
- `N`: columns in `B` and `C`.
- `K`: columns in `A` and rows in `B`.

Edit the definitions near the top of a shader to change its matrix sizes:

```hlsl
#define M 2048    // Rows in A and C
#define N 1024    // Columns in B and C
#define K 512     // Columns in A, rows in B
```