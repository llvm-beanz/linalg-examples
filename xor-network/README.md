# XOR classifier D3D12 sample

The XOR sample is a companion to the [sine-network tutorial](../sin-network/README.md).
Read that tutorial first for the shared D3D12 host setup, preview runtime,
run-time shader compilation, synchronization, and error handling. This guide
follows the same execution flow but concentrates on what XOR does differently:
it evaluates the same padded batch with threadgroup-scope matrices and with
smaller wave-scope GEMM tiles.

After completing this walkthrough, you should be able to explain how the four
truth-table rows become cooperative matrices, how the two implementations
execute the same network, and how the host compares their probability matrices.

## Dependencies and prerequisites

Use the [tested dependency versions](../sin-network/README.md#tested-dependency-versions)
and [prerequisites](../sin-network/README.md#prerequisites) from the sine sample.
Both samples use the same preview DXC, Agility SDK, WARP package, Windows SDK,
and build tools. The XOR build stages those dependencies independently in its
own build directory.

The XOR-specific source files are:

- [`xor-network.cpp`](xor-network.cpp): packs the matrices, creates raw views,
  dispatches both shaders, and compares their results
- [`xor-network.hlsl`](xor-network.hlsl): evaluates both network layers with
  threadgroup-scope matrices
- [`xor-network-wave.hlsl`](xor-network-wave.hlsl): evaluates both layers as
  tiled wave-scope GEMM operations
- [`CMakeLists.txt`](CMakeLists.txt): builds the host and stages its runtime files

## Build and run

From the repository root, run:

```powershell
cmake -S xor-network -B build/xor-network -G Ninja `
  '-DCMAKE_BUILD_TYPE=Release' `
  '-DDXC_VERSION=1.10.2605.37-preview' `
  '-DAGILITY_SDK_VERSION=1.721.3-preview' `
  '-DWARP_VERSION=1.65535.20-preview' `
  '-DLINALG_USE_PREVIEW_HEADERS=ON'
cmake --build build/xor-network
./build/xor-network/xor-network.exe `
  ./build/xor-network/xor-network.hlsl `
  ./build/xor-network/xor-network-wave.hlsl
```

A successful run ends with:

```text
0 XOR 0: threadgroup = 0 (0.000045), wave = 0 (0.000045)
0 XOR 1: threadgroup = 1 (0.999512), wave = 1 (0.999512)
1 XOR 0: threadgroup = 1 (0.999512), wave = 1 (0.999512)
1 XOR 1: threadgroup = 0 (0.000046), wave = 0 (0.000046)
Maximum implementation difference: 0.000000
Both implementations classified XOR correctly and agree within 0.001000.
```

## 1. Start with the batched network equation

The sine sample assigns one scalar input to each shader thread. XOR instead puts
all four truth-table inputs into the rows of one matrix:

$$
X =
\begin{bmatrix}
0 & 0 & 0 & 1 \\
0 & 1 & 0 & 1 \\
1 & 0 & 0 & 1 \\
1 & 1 & 0 & 1
\end{bmatrix}
$$

The logical matrix is padded with four zero rows and four zero columns to form
the physical `8 x 8` matrix used by both shaders. The padding allows the wave
implementation to divide each GEMM into supported `4 x 4` tiles. The last
active component incorporates the hidden bias into matrix multiplication. Both
shaders evaluate:

$$
H = \sigma(XW_h), \qquad P = \sigma(HW_o + b_o)
$$

The first two hidden columns approximate OR and AND:

$$
h_0 = 20x + 20y - 10, \qquad h_1 = 20x + 20y - 30
$$

The first output column combines them with weights $[20, -20, 0, 0]^T$ and
bias $-10$. The other columns and the two extra hidden neurons use zero weights.
The remaining rows and columns are also zero-padded. The logical network remains
`2 -> 4 -> 1`; only its physical matrix representation is `8 x 8`.

## 2. Understand the matrix resource contract

The [sine resource contract](../sin-network/README.md#2-understand-the-shared-cpugpu-resource-contract)
explains descriptor tables, SRVs, UAVs, and matching HLSL registers. XOR keeps
the same three-SRV and one-UAV table, but every resource is a raw matrix buffer:

| C++ resource | HLSL declaration | Register | Matrix role |
| --- | --- | --- | --- |
| `inputBuffer` | `Inputs` | `t0` | FP16 `8 x 8` input (`A`) |
| `hiddenBuffer` | `HiddenWeights` | `t1` | FP16 `8 x 8` weights (`B`) |
| `outputWeightBuffer` | `OutputWeights` | `t2` | FP16 `8 x 8` weights (`B`) |
| two output buffers | `Outputs` | `u0` | FP16 `8 x 8` probabilities |

Each matrix occupies 128 bytes. Its eight FP16 elements per row produce a
16-byte row stride. The host creates two descriptor tables with the same three
SRVs and a different output UAV, so each pipeline writes an independent result.
There is no root constant because the dimensions are fixed in the shaders.

## 3. Select WARP for the threadgroup profile

The [sine runtime section](../sin-network/README.md#3-select-the-preview-d3d12-runtime)
explains the exported SDK version and staged `D3D12` directory. XOR uses the same
mechanism but deliberately creates the staged WARP adapter rather than choosing
a hardware adapter first.

The hardware adapter used during development supports the sine sample's
thread-scope vector-matrix operation, but not XOR's threadgroup profile. Preview
WARP reports both the `8 x 8 x 8` threadgroup operation and a four-lane
`4 x 4 x 4` wave operation. Explicit WARP selection makes the comparison
reproducible on that configuration.

## 4. Compile the same shader model

XOR uses the same run-time DXC path and options described in the
[sine shader-compilation section](../sin-network/README.md#4-compile-the-compute-shader-at-run-time):
entry point `main`, target `cs_6_10`, native 16-bit types, and the staged include
directory for `dx/linalg.h`. The host compiles both shader files with those
options and creates one compute pipeline for each implementation.

## 5. Query both matrix profiles

After creating WARP, `ReportLinearAlgebraSupport` checks linear-algebra tier 1
and then queries `D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREADGROUP_MATRIX_MULTIPLY`.
The threadgroup profile specifies:

- wave size 4
- FP16 `A`, `B`, and accumulator components
- an $M=8$, $K=8$, $N=8$ multiplication shape

The host then queries `D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY`
for a four-lane wave, FP16 components, and a supported `4 x 4 x 4` shape. These
queries differ from the sine sample's `THREAD_VECTOR_MATRIX_MULTIPLY` query. The
host stops before pipeline creation if either required profile is unavailable.

## 6. Pack the input and weight matrices

The host allocates three arrays of 64 `uint16_t` values and uses the same
`FloatToHalf` conversion explained by the
[sine data-preparation section](../sin-network/README.md#6-prepare-the-network-and-input-data).

For each of the first four rows, the host writes $x$ and $y$ into columns 0 and
1, leaves column 2 at zero, and writes one into column 3. It packs the logical
weights into the upper-left `4 x 4` block of each physical `8 x 8` matrix. All
remaining elements stay zero.

## 7. Create raw matrix views

All three SRVs use `DXGI_FORMAT_R32_TYPELESS` with
`D3D12_BUFFER_SRV_FLAG_RAW`. Each view exposes 32 32-bit elements, which cover
the 128 bytes occupied by 64 FP16 matrix components. Each output uses a raw UAV
with the same byte size. The eight-descriptor heap contains one four-descriptor
table per implementation.

The root signature contains only the descriptor table. For the general root
signature and descriptor-heap flow, see the
[sine descriptor section](../sin-network/README.md#7-create-descriptors-and-the-root-signature).

## 8. Follow the two matrix implementations

### Threadgroup matrix

One 32-thread group collectively owns each matrix. The shader first loads the
batch as `MatrixUse::A` and the hidden weights as `MatrixUse::B`:

```hlsl
InputMatrix inputs = InputMatrix::Load(
  Inputs, 0, 16, MatrixLayout::RowMajor);
HiddenWeightMatrix hiddenWeights = HiddenWeightMatrix::Load(
  HiddenWeights, 0, 16, MatrixLayout::RowMajor);
AccumulatorMatrix hidden = Multiply(inputs, hiddenWeights);
```

`Multiply` differs from sine's per-thread `MultiplyAdd`: it returns an
accumulator matrix distributed across the thread group. Each participating
thread visits its implementation-defined local elements through `Length`,
`Get`, and `Set` to apply sigmoid.

The hidden accumulator must become an `A` matrix before it can feed the second
matrix multiplication:

```hlsl
HiddenActivationMatrix activations =
    hidden.Cast<ComponentType::F16, MatrixUse::A>();
```

After the second `Multiply`, `GetCoordinate` identifies elements in the first
output column so the shader can add the output bias. Finally, `Store` writes the
distributed accumulator to `Outputs` in row-major layout. Together, this path
demonstrates `A`, `B`, and accumulator roles, matrix-matrix `Multiply`, element
access, role conversion, and matrix storage.

### Wave matrix tiles

The wave shader dispatches a `2 x 2` grid of groups. Each four-thread group is
one wave and owns one `4 x 4` output tile. For each hidden-column tile, it first
computes the corresponding hidden activation tile as a GEMM over two K tiles:

```hlsl
AccumulatorTile hidden = AccumulatorTile::Splat(0.0h);
for (uint inputColumn = 0; inputColumn < MatrixDimension;
   inputColumn += TileSize)
{
  InputTile inputTile = InputTile::Load(...);
  WeightTile hiddenWeightTile = WeightTile::Load(...);
  hidden.MultiplyAccumulate(inputTile, hiddenWeightTile);
}
```

After sigmoid and `Cast`, that hidden tile is multiplied by the corresponding
output-weight tile with another `MultiplyAccumulate`. Repeating the outer loop
over hidden columns completes the output GEMM. The wave stores its tile at the
appropriate row and column offset in the shared output buffer. This tiled path
uses `Splat` and `MultiplyAccumulate` rather than the threadgroup shader's
whole-matrix `Multiply`.

## 9. Dispatch both implementations

The resource transitions, command submission, fence, and readback sequence are
the same as the [sine command flow](../sin-network/README.md#9-record-and-submit-the-d3d12-work).
The host binds the first output table and dispatches the threadgroup pipeline,
then switches the pipeline and descriptor table for the wave implementation:

```cpp
commands->Dispatch(1, 1, 1);
commands->SetPipelineState(wavePipeline.Get());
commands->Dispatch(2, 2, 1);
```

The threadgroup shader uses one 32-thread group for the complete `8 x 8`
operation. The wave shader uses four groups, each containing the four lanes
required by one `4 x 4` tile. The outputs are independent, so the dispatches do
not need a UAV barrier between them.

## 10. Validate and compare the results

Each shader writes a row-major FP16 `8 x 8` matrix. The host maps both readback
buffers, converts every element with `HalfToFloat`, and verifies that the
absolute difference is at most `0.001`. It then reads column 0 from the first
four rows, applies a `0.5` threshold, and checks both implementations against
`[0, 1, 1, 0]`. Any non-finite value, excessive difference, or incorrect class
causes the sample to fail.

This differs from the sine sample's numerical error metrics: XOR validates a
decision boundary, while sine reports maximum and RMS approximation error.

## Troubleshooting

For shader compilation, Agility SDK loading, and general D3D12 failures, use the
[sine troubleshooting guide](../sin-network/README.md#troubleshooting).

### A matrix capability query fails

Confirm that the staged WARP package version matches the version listed in the
sine tutorial and that `LINALG_USE_PREVIEW_HEADERS=ON` was used during configure.
This sample requires FP16 `8 x 8 x 8` threadgroup multiplication and FP16
`4 x 4 x 4` multiplication on a four-lane wave. A different adapter or preview
package may expose different profiles.

### The classes are wrong after changing weights

Keep the CPU arrays and both shader interpretations row-major. The host reads
output column 0 at indices `0`, `8`, `16`, and `24`; moving the classifier
output to another column requires changing those readback indices.

## Execution summary

```text
CPU: pad four inputs and two weight matrices to FP16 8 x 8 matrices
  -> query threadgroup 8 x 8 x 8 and wave 4 x 4 x 4 support
  -> compile two shaders and bind independent output UAVs
GPU threadgroup path: Load A and B -> Multiply -> sigmoid via Get/Set
  -> Cast accumulator to A -> Multiply -> add bias via GetCoordinate
GPU wave path: tile 8 x 8 into 4 x 4 matrices
  -> Splat -> tiled MultiplyAccumulate -> sigmoid -> tiled Store
CPU: compare all outputs within 0.001
  -> threshold column 0 from both -> compare with XOR truth table
```
