// This shader performs matrix multiplication C = α*A*B + β*C
// where A, B, and C are matrices of dimensions MxK, KxN, and MxN respectively.
// The shader uses threadgroup-level parallelism to compute tiles of the output
// matrix C. The GPU driver will generate code to split the matrix into optimal
// tiles based on the hardware capabilities.

#include <dx/linalg.h>

using namespace dx::linalg;

// GEMM constants
cbuffer GemmConstants : register(b0)
{
    float alpha;    // Scalar multiplier for A*B
    float beta;     // Scalar multiplier for existing C
}

ByteAddressBuffer MatrixA;
ByteAddressBuffer MatrixB;
RWByteAddressBuffer MatrixC;

// Matrix dimensions - can be configured as needed
#define M 1024    // Rows in A and C
#define N 1024    // Columns in B and C  
#define K 1024    // Columns in A, rows in B

// Optimized GEMM using threadgroup-level parallelism
[numthreads(1024, 1, 1)]
void main()
{
    // Matrix type definitions for threadgroup scope
    using MatrixATy = Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::ThreadGroup>;
    using MatrixBTy = Matrix<ComponentType::F16, N, K, MatrixUse::B, MatrixScope::ThreadGroup>;
    using MatrixResultTy = Matrix<ComponentType::F32, M, N, MatrixUse::Accumulator, MatrixScope::ThreadGroup>;

    MatrixATy a_matrix = MatrixATy::Load(MatrixA, 0, K * sizeof(half), MatrixLayout::RowMajor);
            
    MatrixBTy b_matrix = MatrixBTy::Load(MatrixB, 0, N * sizeof(half), MatrixLayout::RowMajor);
    
    // Load existing C matrix for GEMM equation: C = α*A*B + β*C  
    MatrixResultTy c_existing = MatrixResultTy::Load(MatrixC, 0, N * sizeof(float), MatrixLayout::RowMajor);
    
    // Compute A*B
    MatrixResultTy ab_result = MatrixResultTy::Multiply(a_matrix, b_matrix);
    
    // Apply GEMM scaling element-wise: α*A*B + β*C
    for (uint i = 0; i < ab_result.Length(); i++) {
        float ab_val = ab_result.Get(i);
        float c_val = c_existing.Get(i);
        float result = alpha * ab_val + beta * c_val;
        ab_result.Set(i, result);
    }
    
    ab_result.Store(MatrixC, 0, N * sizeof(float), MatrixLayout::RowMajor);
}