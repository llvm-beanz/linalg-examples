namespace hlsl {

#ifdef __hlsl_dx_compiler
#define SIZE_TYPE int
#else
#define SIZE_TYPE uint
#endif

template <typename T> struct is_arithmetic {
  static const bool value = false;
};

#define __ARITHMETIC_TYPE(type)                                                \
  template <> struct is_arithmetic<type> {                                     \
    static const bool value = true;                                            \
  };

#if __HLSL_ENABLE_16_BIT
__ARITHMETIC_TYPE(uint16_t)
__ARITHMETIC_TYPE(int16_t)
#endif
__ARITHMETIC_TYPE(uint)
__ARITHMETIC_TYPE(int)
__ARITHMETIC_TYPE(uint64_t)
__ARITHMETIC_TYPE(int64_t)
__ARITHMETIC_TYPE(half)
__ARITHMETIC_TYPE(float)
__ARITHMETIC_TYPE(double)

template <bool B, typename T> struct enable_if {};

template <typename T> struct enable_if<true, T> {
  using type = T;
};

} // namespace hlsl

namespace dx {

namespace linalg {

struct ComponentType {
  enum ComponentEnum {
    Invalid = 0,
    I1 = 1,
    I8 = 2,
    U8 = 3,
    I16 = 4,
    U16 = 5,
    I32 = 6,
    U32 = 7,
    I64 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    F64 = 12,
    SNormF16 = 13,
    UNormF16 = 14,
    SNormF32 = 15,
    UNormF32 = 16,
    SNormF64 = 17,
    UNormF64 = 18,
    F8_E4M3 = 19,
    F8_E5M2 = 20,
  };
};
using ComponentEnum = ComponentType::ComponentEnum;

struct MatrixUse {
  enum MatrixUseEnum {
    A = 0,
    B = 1,
    Accumulator = 2,
  };
};
using MatrixUseEnum = MatrixUse::MatrixUseEnum;

struct MatrixScope {
  enum MatrixScopeEnum {
    Thread = 0,
    Wave = 1,
    ThreadGroup = 2,
  };
};
using MatrixScopeEnum = MatrixScope::MatrixScopeEnum;

struct MatrixLayout {
  enum MatrixLayoutEnum {
    RowMajor = 0,
    ColMajor = 1,
    MulOptimal = 2,
    OuterProductOptimal = 3,
  };
};
using MatrixLayoutEnum = MatrixLayout::MatrixLayoutEnum;

namespace __detail {
template <ComponentEnum T> struct ComponentTypeTraits {
  using Type = uint;
  static const bool IsNativeScalar = false;
  static const uint ElementsPerScalar = 4;
};

#define __MATRIX_SCALAR_COMPONENT_MAPPING(enum_val, type)                      \
  template <> struct ComponentTypeTraits<enum_val> {                           \
    using Type = type;                                                         \
    static const bool IsNativeScalar = true;                                   \
    static const uint ElementsPerScalar = 1;                                   \
  };

#if __HLSL_ENABLE_16_BIT
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::I16, int16_t)
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::U16, uint16_t)
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::F16, float16_t)
#endif

__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::I32, int32_t)
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::U32, uint32_t)
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::F32, float)
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::I64, int64_t)
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::U64, uint64_t)
__MATRIX_SCALAR_COMPONENT_MAPPING(ComponentType::F64, double)

} // namespace __detail

template <ComponentEnum ElementType, uint DimA> struct VectorRef {
  ByteAddressBuffer Buf;
  uint Offset;
};

template <typename T, int N, ComponentEnum DT> struct InterpretedVector {
  vector<T, N> Data;
  static const ComponentEnum Interpretation = DT;
  static const SIZE_TYPE Size =
      __detail::ComponentTypeTraits<DT>::ElementsPerScalar * N;
};

template <ComponentEnum DT, typename T, int N>
InterpretedVector<T, N, DT> MakeInterpretedVector(vector<T, N> Vec) {
  InterpretedVector<T, N, DT> IV = {Vec};
  return IV;
}

template <ComponentEnum ComponentTy, SIZE_TYPE M, SIZE_TYPE N,
          MatrixUseEnum Use, MatrixScopeEnum Scope>
class Matrix {
  using ElementType = typename __detail::ComponentTypeTraits<ComponentTy>::Type;
  // If this isn't a native scalar, we have an 8-bit type, so we have 4 elements
  // packed in each scalar value.
  static const uint ElementsPerScalar =
      __detail::ComponentTypeTraits<ComponentTy>::ElementsPerScalar;

  template <ComponentEnum NewCompTy, MatrixUseEnum NewUse = Use,
            bool Transpose = false>
  Matrix<NewCompTy, M, N, NewUse, Scope> Cast();

  template <typename T>
  static typename hlsl::enable_if<hlsl::is_arithmetic<T>::value, Matrix>::type
  Splat(T Val);

  static Matrix Load(ByteAddressBuffer Res, uint StartOffset, uint Stride,
                     MatrixLayoutEnum Layout, uint Align = sizeof(ElementType));

  static Matrix Load(RWByteAddressBuffer Res, uint StartOffset, uint Stride,
                     MatrixLayoutEnum Layout, uint Align = sizeof(ElementType));

  template <typename T>
  static typename hlsl::enable_if<hlsl::is_arithmetic<T>::value, Matrix>::type
  Load(/*groupshared*/ T Arr[], uint StartIdx, uint Stride,
       MatrixLayoutEnum Layout);

  uint Length();

  uint2 GetCoordinate(uint);

  ElementType Get(uint);

  void Set(uint, ElementType);

  void Store(RWByteAddressBuffer Res, uint StartOffset, uint Stride,
             MatrixLayoutEnum Layout, uint Align = sizeof(ElementType));

  template <typename T, SIZE_TYPE Size>
  typename hlsl::enable_if<hlsl::is_arithmetic<T>::value &&
                               (M * N / ElementsPerScalar >= Size),
                           void>::type
  Store(/*groupshared*/ T Arr[Size], uint StartIdx, uint Stride,
        MatrixLayoutEnum Layout);

  // Accumulate methods
  template <MatrixUseEnum UseLocal = Use>
  typename hlsl::enable_if<Use == MatrixUse::Accumulator && UseLocal == Use,
                           void>::type
  InterlockedAccumulate(RWByteAddressBuffer Res, uint StartOffset, uint Stride,
                        MatrixLayoutEnum Layout,
                        uint Align = sizeof(ElementType));

  template <typename T, MatrixUseEnum UseLocal = Use>
  typename hlsl::enable_if<hlsl::is_arithmetic<T>::value &&
                               Use == MatrixUse::Accumulator && UseLocal == Use,
                           void>::type
  InterlockedAccumulate(/*groupshared*/ T Arr[], uint StartIdx, uint Stride,
                        MatrixLayoutEnum Layout);

  template <ComponentEnum LHSTy, ComponentEnum RHSTy,
            MatrixUseEnum UseLocal = Use>
  typename hlsl::enable_if<Use == MatrixUse::Accumulator && UseLocal == Use,
                           void>::type
  Accumulate(const Matrix<LHSTy, M, N, MatrixUse::A, Scope>);

  template <ComponentEnum LHSTy, ComponentEnum RHSTy,
            MatrixUseEnum UseLocal = Use>
  typename hlsl::enable_if<Use == MatrixUse::Accumulator && UseLocal == Use,
                           void>::type
  Accumulate(const Matrix<RHSTy, M, N, MatrixUse::B, Scope>);

  template <ComponentEnum LHSTy, ComponentEnum RHSTy, SIZE_TYPE K,
            MatrixUseEnum UseLocal = Use>
  typename hlsl::enable_if<Use == MatrixUse::Accumulator && UseLocal == Use,
                           void>::type
  MultiplyAccumulate(const Matrix<LHSTy, M, K, MatrixUse::A, Scope>,
                     const Matrix<RHSTy, K, N, MatrixUse::B, Scope>);
};

// Thread-scope Matrices are read-only. Using a template partial specialization
// for this simplifies the SFINAE-foo above.
template <ComponentEnum ComponentTy, SIZE_TYPE M, SIZE_TYPE N,
          MatrixUseEnum Use>
class Matrix<ComponentTy, M, N, Use, MatrixScope::Thread> {
  using ElementType = typename __detail::ComponentTypeTraits<ComponentTy>::Type;

  template <MatrixLayoutEnum Layout>
  static Matrix Load(ByteAddressBuffer Res, uint StartOffset, uint Stride,
                     uint Align = sizeof(ElementType));

  void InterlockedAccumulate(RWByteAddressBuffer Res, uint StartOffset,
                             uint Align = sizeof(ElementType));
};

MatrixUseEnum AccumulatorLayout();

template <ComponentEnum OutTy, ComponentEnum ATy, ComponentEnum BTy,
          SIZE_TYPE M, SIZE_TYPE N, SIZE_TYPE K>
Matrix<OutTy, M, N, MatrixUse::Accumulator, MatrixScope::Wave>
Multiply(const Matrix<ATy, M, K, MatrixUse::A, MatrixScope::Wave>,
         const Matrix<BTy, K, N, MatrixUse::B, MatrixScope::Wave>);

template <ComponentEnum T, SIZE_TYPE M, SIZE_TYPE N, SIZE_TYPE K>
Matrix<T, M, N, MatrixUse::Accumulator, MatrixScope::Wave>
Multiply(const Matrix<T, M, K, MatrixUse::A, MatrixScope::Wave>,
         const Matrix<T, K, N, MatrixUse::B, MatrixScope::Wave>);

template <ComponentEnum OutTy, ComponentEnum ATy, ComponentEnum BTy,
          SIZE_TYPE M, SIZE_TYPE N, SIZE_TYPE K>
Matrix<OutTy, M, N, MatrixUse::Accumulator, MatrixScope::ThreadGroup>
Multiply(const Matrix<ATy, M, K, MatrixUse::A, MatrixScope::ThreadGroup>,
         const Matrix<BTy, K, N, MatrixUse::B, MatrixScope::ThreadGroup>);

template <ComponentEnum T, SIZE_TYPE M, SIZE_TYPE N, SIZE_TYPE K>
Matrix<T, M, N, MatrixUse::Accumulator, MatrixScope::ThreadGroup>
Multiply(const Matrix<T, M, K, MatrixUse::A, MatrixScope::ThreadGroup>,
         const Matrix<T, K, N, MatrixUse::B, MatrixScope::ThreadGroup>);

// Cooperative Vector Replacement API
// Cooperative Vector operates on per-thread vectors multiplying against B
// matrices with thread scope.

template <typename OutputElTy, typename InputElTy, SIZE_TYPE M, SIZE_TYPE K,
          ComponentEnum MatrixDT, MatrixScopeEnum Scope>
vector<OutputElTy, M> Multiply(Matrix<MatrixDT, M, K, MatrixUse::A, Scope>,
                               vector<InputElTy, K>);

template <typename OutputElTy, typename InputElTy, typename BiasElTy,
          SIZE_TYPE M, SIZE_TYPE K, ComponentEnum MatrixDT,
          MatrixScopeEnum Scope>
vector<OutputElTy, M> MultiplyAdd(Matrix<MatrixDT, M, K, MatrixUse::A, Scope>,
                                  vector<InputElTy, K>, vector<BiasElTy, M>);

template <typename OutputElTy, typename InputElTy, ComponentEnum InputInterp,
          typename BiasElTy, SIZE_TYPE M, SIZE_TYPE VecM, SIZE_TYPE K,
          ComponentEnum MatrixDT, MatrixScopeEnum Scope>
typename hlsl::enable_if<
    InterpretedVector<InputElTy, VecM, InputInterp>::Size == M,
    vector<OutputElTy, K> >::type
    MultiplyAdd(Matrix<MatrixDT, M, K, MatrixUse::A, Scope>,
                InterpretedVector<InputElTy, VecM, InputInterp>,
                vector<BiasElTy, K>);

template <typename OutputElTy, typename InputElTy, ComponentEnum BiasElTy,
          SIZE_TYPE M, SIZE_TYPE K, ComponentEnum MatrixDT>
vector<OutputElTy, K>
    MultiplyAdd(Matrix<MatrixDT, M, K, MatrixUse::A, MatrixScope::Thread>,
                vector<InputElTy, M>, VectorRef<BiasElTy, K>);

template <typename OutputElTy, typename InputElTy, ComponentEnum InputInterp,
          ComponentEnum BiasElTy, SIZE_TYPE M, SIZE_TYPE VecM, SIZE_TYPE K,
          ComponentEnum MatrixDT>
typename hlsl::enable_if<
    InterpretedVector<InputElTy, VecM, InputInterp>::Size == M,
    vector<OutputElTy, K> >::type
    MultiplyAdd(Matrix<MatrixDT, M, K, MatrixUse::A, MatrixScope::Thread>,
                InterpretedVector<InputElTy, VecM, InputInterp>,
                VectorRef<BiasElTy, K>);

// Outer product functions
template <ComponentEnum OutTy, MatrixScopeEnum Scope, typename InputElTy,
          SIZE_TYPE M, SIZE_TYPE N>
Matrix<OutTy, M, N, MatrixUse::Accumulator, Scope>
    OuterProduct(vector<InputElTy, M>, vector<InputElTy, N>);

// GEMM implementation for 1024x1024 matrices using 16x16 tiles
template<MatrixScopeEnum Scope = MatrixScope::ThreadGroup>
void Gemm1024x1024_16x16Tiles(
    ByteAddressBuffer MatrixA,           // Input matrix A (1024x1024 half)
    ByteAddressBuffer MatrixB,           // Input matrix B (1024x1024 half)
    RWByteAddressBuffer MatrixC)         // Output matrix C (1024x1024 float)
{
    const uint TILE_SIZE = 16;
    const uint NUM_TILES = 1024 / TILE_SIZE; // 64 tiles per dimension
    const uint MATRIX_SIZE = 1024;
    
    // Matrix type definitions
    using HalfMatrixA = Matrix<ComponentType::F16, TILE_SIZE, TILE_SIZE, MatrixUse::A, Scope>;
    using HalfMatrixB = Matrix<ComponentType::F16, TILE_SIZE, TILE_SIZE, MatrixUse::B, Scope>;
    using FloatAccumulator = Matrix<ComponentType::F32, TILE_SIZE, TILE_SIZE, MatrixUse::Accumulator, Scope>;
    
    // Process matrix multiplication in 16x16 tiles
    for (uint tile_row = 0; tile_row < NUM_TILES; ++tile_row)
    {
        for (uint tile_col = 0; tile_col < NUM_TILES; ++tile_col)
        {
            // Initialize accumulator tile to zero
            FloatAccumulator accumulator_tile = FloatAccumulator::Splat(0.0f);
            
            // Compute dot product over all K tiles
            for (uint k_tile = 0; k_tile < NUM_TILES; ++k_tile)
            {
                // Calculate byte offsets for A and B tiles
                uint a_offset = ((tile_row * TILE_SIZE) * MATRIX_SIZE + (k_tile * TILE_SIZE)) * sizeof(half);
                uint b_offset = ((k_tile * TILE_SIZE) * MATRIX_SIZE + (tile_col * TILE_SIZE)) * sizeof(half);
                
                // Load tiles from ByteAddressBuffers
                HalfMatrixA a_tile = HalfMatrixA::Load(
                    MatrixA, a_offset, MATRIX_SIZE * sizeof(half), MatrixLayout::RowMajor);
                HalfMatrixB b_tile = HalfMatrixB::Load(
                    MatrixB, b_offset, MATRIX_SIZE * sizeof(half), MatrixLayout::RowMajor);
                
                // Perform tile matrix multiplication and accumulate with mixed precision
                accumulator_tile.MultiplyAccumulate(a_tile, b_tile);
            }
            
            // Calculate byte offset for output tile and store result
            uint c_offset = ((tile_row * TILE_SIZE) * MATRIX_SIZE + (tile_col * TILE_SIZE)) * sizeof(float);
            accumulator_tile.Store(MatrixC, c_offset, MATRIX_SIZE * sizeof(float), MatrixLayout::RowMajor);
        }
    }
}

} // namespace linalg
} // namespace dx