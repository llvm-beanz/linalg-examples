#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer Inputs : register(t0);
ByteAddressBuffer HiddenWeights : register(t1);
ByteAddressBuffer OutputWeights : register(t2);
RWByteAddressBuffer Outputs : register(u0);

using InputMatrix =
    Matrix<ComponentType::F16, 8, 8, MatrixUse::A, MatrixScope::ThreadGroup>;
using HiddenWeightMatrix =
    Matrix<ComponentType::F16, 8, 8, MatrixUse::B, MatrixScope::ThreadGroup>;
using HiddenActivationMatrix =
    Matrix<ComponentType::F16, 8, 8, MatrixUse::A, MatrixScope::ThreadGroup>;
using OutputWeightMatrix =
    Matrix<ComponentType::F16, 8, 8, MatrixUse::B, MatrixScope::ThreadGroup>;
using AccumulatorMatrix =
    Matrix<ComponentType::F16, 8, 8, MatrixUse::Accumulator, MatrixScope::ThreadGroup>;

float Sigmoid(float value)
{
    return 1.0f / (1.0f + exp(-value));
}

[numthreads(32, 1, 1)]
void main()
{
    InputMatrix inputs = InputMatrix::Load(
        Inputs, 0, 16, MatrixLayout::RowMajor);
    HiddenWeightMatrix hiddenWeights = HiddenWeightMatrix::Load(
        HiddenWeights, 0, 16, MatrixLayout::RowMajor);
    AccumulatorMatrix hidden = Multiply(inputs, hiddenWeights);

    for (uint element = 0; element < hidden.Length(); ++element)
        hidden.Set(element, (half)Sigmoid(hidden.Get(element)));

    HiddenActivationMatrix activations =
        hidden.Cast<ComponentType::F16, MatrixUse::A>();
    OutputWeightMatrix outputWeights = OutputWeightMatrix::Load(
        OutputWeights, 0, 16, MatrixLayout::RowMajor);
    AccumulatorMatrix logits = Multiply(activations, outputWeights);

    for (uint element = 0; element < logits.Length(); ++element)
    {
        uint2 coordinate = logits.GetCoordinate(element);
        float bias = coordinate.y == 0 ? -10.0f : 0.0f;
        logits.Set(element, (half)Sigmoid(logits.Get(element) + bias));
    }

    logits.Store(Outputs, 0, 16, MatrixLayout::RowMajor);
}
