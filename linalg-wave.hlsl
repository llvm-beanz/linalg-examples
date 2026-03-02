// This shader performs matrix multiplication C = α*A*B + β*C
// where A, B, and C are matrices of dimensions MxK, KxN, and MxN respectively.
// The shader uses wave-level parallelism to compute tiles of the output matrix
// C. Each wave computes a TILE_SIZExTILE_SIZE tile of C. The dispatch must
// allocate waves for each tile of the MxN output matrix.
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
#define TILE_SIZE 16

// Optimized GEMM using wave-level parallelism
[numthreads(TILE_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID)
{
    // Matrix type definitions for wave scope
    using MatrixATy = Matrix<ComponentType::F16, TILE_SIZE, TILE_SIZE, MatrixUse::A, MatrixScope::Wave>;
    using MatrixBTy = Matrix<ComponentType::F16, TILE_SIZE, TILE_SIZE, MatrixUse::B, MatrixScope::Wave>;
    using MatrixResultTy = Matrix<ComponentType::F32, TILE_SIZE, TILE_SIZE, MatrixUse::Accumulator, MatrixScope::Wave>;
    
    // Calculate tile coordinates for this thread group
    uint tile_row = group_id.y;
    uint tile_col = group_id.x;
    
    // Initialize accumulator
    MatrixResultTy c_tile = MatrixResultTy::Splat(0.0f);
    
    // Perform tiled matrix multiplication across K dimension
    for (uint k = 0; k < K; k += TILE_SIZE)
    {
        // Calculate byte offsets for A and B tiles
        uint a_offset = ((tile_row * TILE_SIZE) * K + k) * sizeof(half);
        uint b_offset = (k * N + (tile_col * TILE_SIZE)) * sizeof(half);
        
        // Load A and B tiles for this K iteration using ByteAddressBuffer
        MatrixATy a_k_tile = MatrixATy::Load(
            MatrixA, a_offset, K * sizeof(half), MatrixLayout::RowMajor);
            
        MatrixBTy b_k_tile = MatrixBTy::Load(
            MatrixB, b_offset, N * sizeof(half), MatrixLayout::RowMajor);
        
        // Multiply and accumulate with mixed precision (half inputs -> float accumulation)
        c_tile.MultiplyAccumulate(a_k_tile, b_k_tile);
    }
    
    // Calculate output offset for GEMM equation: C = α*A*B + β*C
    uint c_offset = ((tile_row * TILE_SIZE) * N + (tile_col * TILE_SIZE)) * sizeof(float);
    
    // Load existing C tile
    MatrixResultTy c_existing = MatrixResultTy::Load(MatrixC, c_offset, N * sizeof(float), MatrixLayout::RowMajor);
    
    // Apply GEMM scaling element-wise: α*A*B + β*C
    for (uint i = 0; i < c_tile.Length(); i++) {
        float ab_val = c_tile.Get(i);
        float c_val = c_existing.Get(i);
        float result = alpha * ab_val + beta * c_val;
        c_tile.Set(i, result);
    }
    
    c_tile.Store(MatrixC, c_offset, N * sizeof(float), MatrixLayout::RowMajor);
}