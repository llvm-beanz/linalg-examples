# XOR classifier D3D12 sample

The XOR sample is a companion to the [sine-network tutorial](../sin-network/README.md).
Read that tutorial first for the shared D3D12 host setup, preview runtime,
run-time shader compilation, synchronization, and error handling. This guide
follows the same execution flow but concentrates on what XOR does differently:
it evaluates a complete batch with threadgroup-scope matrix-matrix operations.

After completing this walkthrough, you should be able to explain how the four
truth-table rows become cooperative matrices, how the two network layers execute
across a thread group, and how the host reads the resulting probability matrix.

## Contents

- [Build and run](#build-and-run)
- [Batched network equation](#1-start-with-the-batched-network-equation)
- [Matrix resources](#2-understand-the-matrix-resource-contract)
- [Threadgroup shader](#8-follow-the-cooperative-shader)
- [Result validation](#10-validate-the-classifier)
- [Troubleshooting](#troubleshooting)

## Dependencies and prerequisites

Use the [tested dependency versions](../sin-network/README.md#tested-dependency-versions)
and [prerequisites](../sin-network/README.md#prerequisites) from the sine sample.
Both samples use the same preview DXC, Agility SDK, WARP package, Windows SDK,
and build tools. The XOR build stages those dependencies independently in its
own build directory.

The XOR-specific source files are:

- [`xor-network.cpp`](xor-network.cpp): packs the matrices, creates raw views,
  dispatches the cooperative shader, and validates the truth table
- [`xor-network.hlsl`](xor-network.hlsl): evaluates both network layers with
  threadgroup-scope matrices
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
./build/xor-network/xor-network.exe ./build/xor-network/xor-network.hlsl
```

A successful run ends with:

```text
0 XOR 0 = 0 (probability 0.000045)
0 XOR 1 = 1 (probability 0.999512)
1 XOR 0 = 1 (probability 0.999512)
1 XOR 1 = 0 (probability 0.000046)
All XOR cases classified correctly.
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

The last component incorporates the hidden bias into matrix multiplication. The
shader evaluates the two layers as:

$$
H = \sigma(XW_h), \qquad P = \sigma(HW_o + b_o)
$$

The first two hidden columns approximate OR and AND:

$$
h_0 = 20x + 20y - 10, \qquad h_1 = 20x + 20y - 30
$$

The first output column combines them with weights $[20, -20, 0, 0]^T$ and
bias $-10$. The other columns and the two extra hidden neurons use zero weights.
They keep both operations at the compact `4 x 4 x 4` shape required by this
example.

## 2. Understand the matrix resource contract

The [sine resource contract](../sin-network/README.md#2-understand-the-shared-cpugpu-resource-contract)
explains descriptor tables, SRVs, UAVs, and matching HLSL registers. XOR keeps
the same three-SRV and one-UAV table, but every resource is a raw matrix buffer:

| C++ resource | HLSL declaration | Register | Matrix role |
| --- | --- | --- | --- |
| `inputBuffer` | `Inputs` | `t0` | FP16 `4 x 4` input (`A`) |
| `hiddenBuffer` | `HiddenWeights` | `t1` | FP16 `4 x 4` weights (`B`) |
| `outputWeightBuffer` | `OutputWeights` | `t2` | FP16 `4 x 4` weights (`B`) |
| `outputBuffer` | `Outputs` | `u0` | FP16 `4 x 4` probabilities |

Each matrix occupies 32 bytes. Its four FP16 elements per row produce an 8-byte
row stride. There is no root constant because the batch and matrix dimensions
are fixed in the shader.

## 3. Select WARP for the threadgroup profile

The [sine runtime section](../sin-network/README.md#3-select-the-preview-d3d12-runtime)
explains the exported SDK version and staged `D3D12` directory. XOR uses the same
mechanism but deliberately creates the staged WARP adapter rather than choosing
a hardware adapter first.

The hardware adapter used during development supports the sine sample's
thread-scope vector-matrix operation, but not XOR's compact threadgroup matrix
profile. Preview WARP reports support for the profile with a four-lane wave and
lets the driver tile the operation across the shader's 32-thread group. Explicit
WARP selection makes the tutorial reproducible on that configuration.

## 4. Compile the same shader model

XOR uses the same run-time DXC path and options described in the
[sine shader-compilation section](../sin-network/README.md#4-compile-the-compute-shader-at-run-time):
entry point `main`, target `cs_6_10`, native 16-bit types, and the staged include
directory for `dx/linalg.h`. The difference is in the matrix operations emitted
by the shader, not in its compilation setup.

## 5. Query threadgroup matrix support

After creating WARP, `ReportLinearAlgebraSupport` checks linear-algebra tier 1
and then queries `D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREADGROUP_MATRIX_MULTIPLY`.
The requested profile specifies:

- wave size 4
- FP16 `A`, `B`, and accumulator components
- an $M=4$, $K=4$, $N=4$ multiplication shape

This differs from the sine sample's
`THREAD_VECTOR_MATRIX_MULTIPLY` capability query. The host stops before pipeline
creation if WARP does not report the threadgroup profile.

## 6. Pack the input and weight matrices

The host allocates three arrays of 16 `uint16_t` values and uses the same
`FloatToHalf` conversion explained by the
[sine data-preparation section](../sin-network/README.md#6-prepare-the-network-and-input-data).

For each input row, the host writes $x$ and $y$ into columns 0 and 1, leaves
column 2 at zero, and writes one into column 3. It then packs `kHiddenWeights`
and `kOutputWeights` in row-major order. Unlike sine's padded hidden rows and
separate bias vector, all three XOR resources are tightly packed matrices.

## 7. Create raw matrix views

All three SRVs use `DXGI_FORMAT_R32_TYPELESS` with
`D3D12_BUFFER_SRV_FLAG_RAW`. Each view exposes eight 32-bit elements, which
cover the 32 bytes occupied by 16 FP16 matrix components. The output uses a raw
UAV with the same byte size.

The root signature contains only the descriptor table. For the general root
signature and descriptor-heap flow, see the
[sine descriptor section](../sin-network/README.md#7-create-descriptors-and-the-root-signature).

## 8. Follow the cooperative shader

One 32-thread group collectively owns each matrix. The shader first loads the
batch as `MatrixUse::A` and the hidden weights as `MatrixUse::B`:

```hlsl
InputMatrix inputs = InputMatrix::Load(
    Inputs, 0, 8, MatrixLayout::RowMajor);
HiddenWeightMatrix hiddenWeights = HiddenWeightMatrix::Load(
    HiddenWeights, 0, 8, MatrixLayout::RowMajor);
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

## 9. Dispatch one cooperative group

The resource transitions, command submission, fence, and readback sequence are
the same as the [sine command flow](../sin-network/README.md#9-record-and-submit-the-d3d12-work).
The dispatch itself differs:

```cpp
commands->Dispatch(1, 1, 1);
```

The single group contains the 32 threads declared by `[numthreads(32, 1, 1)]`.
These threads cooperate on the fixed batch; they do not map one-to-one to the
four input rows.

## 10. Validate the classifier

`Store` writes a row-major FP16 `4 x 4` matrix. The host maps the readback buffer,
uses `HalfToFloat` on the first element of each row, and applies a `0.5`
classification threshold. It compares the classes with `[0, 1, 1, 0]` and
returns failure if any row is incorrect.

This differs from the sine sample's numerical error metrics: XOR validates a
decision boundary, while sine reports maximum and RMS approximation error.

## Troubleshooting

For shader compilation, Agility SDK loading, and general D3D12 failures, use the
[sine troubleshooting guide](../sin-network/README.md#troubleshooting).

### The threadgroup capability query fails

Confirm that the staged WARP package version matches the version listed in the
sine tutorial and that `LINALG_USE_PREVIEW_HEADERS=ON` was used during configure.
This sample requests FP16 `4 x 4 x 4` multiplication with a four-lane wave; a
different adapter or preview package may expose a different profile.

### The classes are wrong after changing weights

Keep the CPU arrays and shader interpretation row-major. The host reads output
column 0 at indices `0`, `4`, `8`, and `12`; moving the classifier output to
another column requires changing those readback indices.

## Execution summary

```text
CPU: pack four FP16 input rows and two FP16 weight matrices
  -> create preview WARP device and query threadgroup 4 x 4 x 4 support
  -> bind three raw SRVs and one raw UAV
GPU thread group: Load A and B -> Multiply -> sigmoid via Get/Set
  -> Cast accumulator to A -> Multiply -> add bias via GetCoordinate
  -> Store FP16 probability matrix
CPU: read output column 0 -> threshold -> compare with XOR truth table
```
