# HLSL Linear Algebra Examples

This repository contains HLSL implementations of matrix operations, specifically focusing on GEMM (General Matrix Multiply) operations with various implementation approaches.

## What is GEMM?

A **GEMM (General Matrix Multiply)** operation is one of the most fundamental and computationally intensive operations in linear algebra and machine learning. Here's a comprehensive description:

### Mathematical Operation

GEMM performs the operation: **C = α × A × B + β × C**

Where:
- **A** is an M×K matrix (M rows, K columns)
- **B** is a K×N matrix (K rows, N columns) 
- **C** is an M×N matrix (M rows, N columns)
- **α** (alpha) and **β** (beta) are scalars

### Core Matrix Multiplication

The fundamental computation for each element C[i,j] is:
```
C[i,j] = Σ(k=0 to K-1) A[i,k] × B[k,j]
```

This means each element in the result matrix C is the dot product of a row from matrix A and a column from matrix B.

### Key Characteristics

**Dimensions**:
- Input A: M × K
- Input B: K × N  
- Output C: M × N
- The inner dimension K must match between A and B

**Computational Complexity**: O(M × N × K) operations

**Memory Access Patterns**: 
- A is accessed row-wise
- B is accessed column-wise
- C is accessed in various patterns depending on algorithm

## Implementation Examples

This repository demonstrates different approaches based on feature availability:

- **[gemm.hlsl](gemm.hlsl)**: Basic tiling with shared memory tiles to improve cache locality
- **[linalg-wave.hlsl](linalg-wave.hlsl)**: Hardware-accelerated manually tiled using `Wave`-scope `linalg::Matrix` objects
- **[linalg-threadgroup.hlsl](linalg-threadgroup.hlsl)**: Hardware-accelerated driver tiled using `ThreadGroup`-scope `linalg::Matrix` objects

## Matrix Dimensions

All implementations support configurable M, N, and K dimensions:
- **M**: Number of rows in matrix A and C
- **N**: Number of columns in matrix B and C  
- **K**: Number of columns in matrix A and rows in matrix B

To modify matrix sizes, edit the `#define` statements at the top of each shader file:

```hlsl
#define M 2048    // Rows in A and C
#define N 1024    // Columns in B and C  
#define K 512     // Columns in A, rows in B
```
