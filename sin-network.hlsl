#include <dx/linalg.h>

using namespace dx::linalg;

// Each hidden-layer row contains four FP16 weights followed by padding to the
// 16-byte stride required by thread-scope matrix loads.
ByteAddressBuffer HiddenWeights : register(t0);
StructuredBuffer<float> Inputs : register(t1);
StructuredBuffer<float> OutputWeights : register(t2);
RWStructuredBuffer<float> Outputs : register(u0);

cbuffer Constants : register(b0)
{
    uint InputCount;
}

using HiddenMatrix =
    Matrix<ComponentType::F16, 16, 4, MatrixUse::A, MatrixScope::Thread>;

[numthreads(64, 1, 1)]
void main(uint index : SV_DispatchThreadID)
{
    if (index >= InputCount)
        return;

    float x = Inputs[index];
    half4 features = half4((half)x, 0.0h, 0.0h, 1.0h);

    HiddenMatrix weights = HiddenMatrix::Load<MatrixLayoutEnum::RowMajor>(
        HiddenWeights, 0, 16);
    vector<half, 16> bias = HiddenWeights.Load<vector<half, 16> >(256);
    vector<half, 16> hidden = MultiplyAdd<half>(weights, features, bias);

    float result = OutputWeights[16];
    for (uint neuron = 0; neuron < 16; ++neuron)
        result += OutputWeights[neuron] * tanh((float)hidden[neuron]);

    Outputs[index] = result;
}