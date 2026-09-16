#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer Inputs : register(t0);
ByteAddressBuffer HiddenWeights : register(t1);
ByteAddressBuffer OutputWeights : register(t2);
RWByteAddressBuffer Outputs : register(u0);

static const uint MatrixDimension = 8;
static const uint TileSize = 4;
static const uint MatrixStride = MatrixDimension * sizeof(half);

using InputTile =
    Matrix<ComponentType::F16, TileSize, TileSize, MatrixUse::A, MatrixScope::Wave>;
using WeightTile =
    Matrix<ComponentType::F16, TileSize, TileSize, MatrixUse::B, MatrixScope::Wave>;
using ActivationTile =
    Matrix<ComponentType::F16, TileSize, TileSize, MatrixUse::A, MatrixScope::Wave>;
using AccumulatorTile =
    Matrix<ComponentType::F16, TileSize, TileSize, MatrixUse::Accumulator, MatrixScope::Wave>;

float Sigmoid(float value)
{
    return 1.0f / (1.0f + exp(-value));
}

[numthreads(4, 1, 1)]
void main(uint2 tile : SV_GroupID)
{
    AccumulatorTile output = AccumulatorTile::Splat(0.0h);

    for (uint hiddenColumn = 0; hiddenColumn < MatrixDimension;
         hiddenColumn += TileSize)
    {
        AccumulatorTile hidden = AccumulatorTile::Splat(0.0h);

        for (uint inputColumn = 0; inputColumn < MatrixDimension;
             inputColumn += TileSize)
        {
            uint inputOffset =
                ((tile.y * TileSize) * MatrixDimension + inputColumn) * sizeof(half);
            uint hiddenWeightOffset =
                (inputColumn * MatrixDimension + hiddenColumn) * sizeof(half);

            InputTile inputTile = InputTile::Load(
                Inputs, inputOffset, MatrixStride, MatrixLayout::RowMajor);
            WeightTile hiddenWeightTile = WeightTile::Load(
                HiddenWeights, hiddenWeightOffset, MatrixStride,
                MatrixLayout::RowMajor);
            hidden.MultiplyAccumulate(inputTile, hiddenWeightTile);
        }

        for (uint element = 0; element < hidden.Length(); ++element)
            hidden.Set(element, (half)Sigmoid(hidden.Get(element)));

        ActivationTile activations =
            hidden.Cast<ComponentType::F16, MatrixUse::A>();
        uint outputWeightOffset =
            (hiddenColumn * MatrixDimension + tile.x * TileSize) * sizeof(half);
        WeightTile outputWeightTile = WeightTile::Load(
            OutputWeights, outputWeightOffset, MatrixStride,
            MatrixLayout::RowMajor);
        output.MultiplyAccumulate(activations, outputWeightTile);
    }

    for (uint element = 0; element < output.Length(); ++element)
    {
        uint2 coordinate = output.GetCoordinate(element);
        uint outputColumn = tile.x * TileSize + coordinate.y;
        float bias = outputColumn == 0 ? -10.0f : 0.0f;
        output.Set(element, (half)Sigmoid(output.Get(element) + bias));
    }

    uint outputOffset =
        ((tile.y * TileSize) * MatrixDimension + tile.x * TileSize) * sizeof(half);
    output.Store(Outputs, outputOffset, MatrixStride, MatrixLayout::RowMajor);
}
