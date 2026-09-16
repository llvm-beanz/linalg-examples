// This shader performs matrix multiplication C = α*A*B + β*C
// where A, B, and C are matrices of dimensions MxK, KxN, and MxN respectively.
// The shader uses groupshared memory to store tiles of A and B. Each thread
// computes one element of the output matrix C. The dispatch dimensions must
// allocate threads for each element of the MxN output matrix.

// GEMM constants
cbuffer GemmConstants : register(b0)
{
    float alpha;    // Scalar multiplier for A*B
    float beta;     // Scalar multiplier for existing C
}

// Input/Output buffers
StructuredBuffer<float> MatrixA : register(t0);
StructuredBuffer<float> MatrixB : register(t1);
RWStructuredBuffer<float> MatrixC : register(u0);

// Matrix dimensions - can be configured as needed
#define M 1024    // Rows in A and C
#define N 1024    // Columns in B and C  
#define K 1024    // Columns in A, rows in B

// Shared memory for tile-based computation
#define TILE_SIZE 16
groupshared float tileA[TILE_SIZE][TILE_SIZE];
groupshared float tileB[TILE_SIZE][TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
           uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.y;
    uint col = dispatchThreadID.x;
    
    float sum = 0.0f;
    
    // Number of tiles needed to cover the K dimension
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Process tiles along the K dimension
    for (uint tileIdx = 0; tileIdx < numTiles; tileIdx++)
    {
        // Load tile from matrix A into shared memory
        uint aRow = row;
        uint aCol = tileIdx * TILE_SIZE + groupThreadID.x;
        
        if (aRow < M && aCol < K)
        {
            tileA[groupThreadID.y][groupThreadID.x] = MatrixA[aRow * sizeof(float) * K + aCol];
        }
        else
        {
            tileA[groupThreadID.y][groupThreadID.x] = 0.0f;
        }
        
        // Load tile from matrix B into shared memory
        uint bRow = tileIdx * TILE_SIZE + groupThreadID.y;
        uint bCol = col;
        
        if (bRow < K && bCol < N)
        {
            tileB[groupThreadID.y][groupThreadID.x] = MatrixB[bRow * sizeof(float) * N + bCol];
        }
        else
        {
            tileB[groupThreadID.y][groupThreadID.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        GroupMemoryBarrierWithGroupSync();
        
        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[groupThreadID.y][k] * tileB[k][groupThreadID.x];
        }
        
        // Synchronize before loading next tile
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Write result to output matrix C using GEMM equation: C = α*A*B + β*C
    if (row < M && col < N)
    {
        uint cIndex = row * N + col;
        float existingC = MatrixC[cIndex];
        MatrixC[cIndex] = alpha * sum + beta * existingC;
    }
}