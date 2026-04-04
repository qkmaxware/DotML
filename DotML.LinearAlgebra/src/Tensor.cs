using System.Collections;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace DotML;

/// <summary>
/// A zero-copy view over a slice of a tensor (no support for complex tensor operations)
/// </summary>
/// <typeparam name="TNum">tensor element type</typeparam>
public sealed class TensorView<TNum>
: ITensorLike<TNum>
where TNum : INumber<TNum>
{
    private readonly TNum[] elements;
    private readonly int offset;
    public readonly Shape Shape;

    public TensorView(Shape shape, TNum[] data, int offset)
    {
        this.elements = data;
        this.offset = offset;
        this.Shape = shape;
    }

    /// <summary>
    /// Number of dimensions in tensor
    /// </summary>
    public int Rank => Shape.Rank;
    /// <summary>
    /// Length/size of a particular dimension
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>dimension length</returns>
    public int GetDimension(int index) => Shape.Length(index);

    /// <summary>
    /// Multi-dimensional index access to tensor elements
    /// </summary>
    /// <param name="indices">multi-dimensional indices</param>
    /// <returns>element</returns>
    public TNum this[params ReadOnlySpan<int> indices]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            int flatIndex = offset;
            var strides = Shape.AsStrideSpan();
            for (int i = 0; i < indices.Length; i++)
                flatIndex += indices[i] * strides[i];
            return elements[flatIndex];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            int flatIndex = offset;
            var strides = Shape.AsStrideSpan();
            for (int i = 0; i < indices.Length; i++)
                flatIndex += indices[i] * strides[i];
            elements[flatIndex] = value;
        }
    }

    /// <summary>
    /// Get a particular element from the tensor by index
    /// </summary>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public TNum GetElementAt(params int[] indices) => this[indices];

    /// <summary>
    /// Materialize this view into a concreate tensor via element copying
    /// </summary>
    /// <returns>tensor</returns>
    public Tensor<TNum> Materialize()
    {
        Tensor<TNum> tensor = Tensor<TNum>.Defaults(this.Shape);

        var enumerator = Shape.CreateIndexEnumerator();
        Span<int> index = stackalloc int[this.Shape.Rank];

        enumerator.Initialize(index);
        while (enumerator.MoveNext(index))
        {
            tensor[index] = this[index];
        }

        return tensor;
    }
}

/// <summary>
/// A generic tensor storage container
/// </summary>
/// <typeparam name="T">Element type</typeparam>
public class Tensor<TNum>
: IMutableTensorLike<TNum>
where TNum : INumber<TNum>
{
    /// <summary>
    /// Maximum amount of parallelism to use
    /// </summary>
    public static int DegreesOfParallelism
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ParallelOptions.MaxDegreeOfParallelism;
    }

    /// <summary>
    /// A parallel options object preset to the degrees of parallelism to use
    /// </summary>
    private static ParallelOptions ParallelOptions = new ParallelOptions
    {
        // TODO if we need better determination of max threads, put the logic here
        MaxDegreeOfParallelism = Environment.ProcessorCount
    };

    /// <summary>
    /// Smallest chunk size allowed during parallel operations
    /// </summary>
    public const int MinParallelChunkSize = 2048;

    /// <summary>
    /// Shape of the tensor
    /// </summary>
    public Shape Shape
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private set;
    }
    private TNum[] elements;

    /// <summary>
    /// Create a new tensor with the given shape and all its elements set to the default value of TNum
    /// </summary>
    /// <param name="shape">tensor shape</param>
    private Tensor(Shape shape)
    {
        this.Shape = shape;
        this.elements = new TNum[shape.LogicalElementCount()];
    }

    /// <summary>
    /// Create a new tensor with the given shape and its elements provided
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <param name="elements">tensor elements</param>
    private Tensor(Shape shape, TNum[] elements)
    {
        this.Shape = shape;
        this.elements = elements;
    }

    /// <summary>
    /// Return a 0-rank empty tensor with no elements
    /// </summary>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Empty()
    {
        return new Tensor<TNum>(Shape.Scalar);
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to the default value of TNum
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Defaults(Shape shape)
    {
        shape = shape.CloneDimensions();
        TNum[] elems = new TNum[shape.LogicalElementCount()];
        return new Tensor<TNum>(shape, elems);
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to 0
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Zeros(Shape shape)
    {
        shape = shape.CloneDimensions();
        TNum[] elems = new TNum[shape.LogicalElementCount()];
        if (TNum.Zero != default(TNum))
            Array.Fill(elems, TNum.Zero); // Only fill if default is not 0 already
        return new Tensor<TNum>(shape, elems);
    }

    /// <summary>
    /// Create a tensor with the same shape of as another with all elements set to 0
    /// </summary>
    /// <param name="other">tensor whose shape to copy</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> ZerosLike<TNumOther>(Tensor<TNumOther> other) where TNumOther:INumber<TNumOther>
    {
        return Tensor<TNum>.Zeros(other.Shape);
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to 1
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Ones(Shape shape)
    {
        shape = shape.CloneDimensions();
        TNum[] elems = new TNum[shape.LogicalElementCount()];
        Array.Fill(elems, TNum.One);
        return new Tensor<TNum>(shape, elems);
    }

    /// <summary>
    /// Create a tensor with the same shape of as another with all elements set to 1
    /// </summary>
    /// <param name="other">tensor whose shape to copy</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> OnesLike<TNumOther>(Tensor<TNumOther> other) where TNumOther:INumber<TNumOther>
    {
        return Tensor<TNum>.Ones(other.Shape);
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to the given value
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <param name="value">element value</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> ConstantValued(Shape shape, TNum value)
    {
        shape = shape.CloneDimensions();
        TNum[] elems = new TNum[shape.LogicalElementCount()];
        Array.Fill(elems, value);
        return new Tensor<TNum>(shape, elems);
    }

    /// <summary>
    /// Generate a random binary mask tensor of the given shape with the given dropout rate
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <param name="dropoutRate">dropout rate as a normalized percentage</param>
    /// <returns>binary mask tensor</returns>
    public static Tensor<TNum> Mask(Shape shape, double dropoutRate)
    {
        var tensor = Tensor<TNum>.Defaults(shape);
        var rng = System.Random.Shared;
        var zero = TNum.Zero;
        var one = TNum.One;
        tensor.ElementWiseInplace((_) => rng.NextDouble() < dropoutRate ? zero : one);
        return tensor;
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to the given values
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <param name="values">element values</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromFlattenedArray(Shape shape, ReadOnlySpan<TNum> values)
    {
        shape = shape.CloneDimensions();
        TNum[] elems = new TNum[shape.LogicalElementCount()];
        values.Slice(0, Math.Min(values.Length, elems.Length)).CopyTo(elems);
        return new Tensor<TNum>(shape, elems);
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to the given values. The array is reused as the underlying storage so should not be modified after being passed to this function
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <param name="values">element values</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromFlattenedArray(Shape shape, TNum[] values)
    {
        if (shape.StorageElementCount() != values.Length)
            throw new ArgumentException($"Array length {values.Length} does not match the number of elements required by the shape {shape} ({shape.StorageElementCount()} elements)");
        return new Tensor<TNum>(shape, values);
    }

    /// Create a tensor from a C# rectangular array 
    /// </summary>
    /// <param name="values">c# array</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromRectangularArray(Array values)
    {
        if (values is null)
            throw new ArgumentNullException(nameof(values));
        if (typeof(TNum) != values.GetType().GetElementType())
            throw new ArgumentException($"Array doesn't contain values of type {typeof(TNum)}");
        if (values.Rank < 1)
            return Tensor<TNum>.Empty();

        // Determine the shape from the array
        int[] shape = new int[values.Rank];
        for (var i = 0; i < shape.Length; i++)
        {
            shape[i] = values.GetLength(i);
        }

        // Both Tensor & rectangular array elements are stored in row-major order so we can just copy the values
        var vspan = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, TNum>(ref MemoryMarshal.GetArrayDataReference(values)), values.Length);
        var tensor = Tensor<TNum>.Defaults(new Shape(shape));
        vspan.CopyTo(tensor.elements);
        return tensor;
    }
    /// Create a tensor from a C# rectangular array 
    /// </summary>
    /// <param name="values">c# array</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromRectangularArray(TNum[] vector) => FromRectangularArray((Array)vector);
    /// Create a tensor from a C# rectangular array 
    /// </summary>
    /// <param name="values">c# array</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromRectangularArray(TNum[,] matrix) => FromRectangularArray((Array)matrix);
    /// Create a tensor from a C# rectangular array 
    /// </summary>
    /// <param name="values">c# array</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromRectangularArray(TNum[,,] featureSet) => FromRectangularArray((Array)featureSet);
    /// Create a tensor from a C# rectangular array 
    /// </summary>
    /// <param name="values">c# array</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromRectangularArray(TNum[,,,] batchedFeatureSet) => FromRectangularArray((Array)batchedFeatureSet);

    /// <summary>
    /// Create a tensor from a C# jagged array (array of arrays)
    /// </summary>
    /// <param name="array">jagged array</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> FromJaggedArray(Array values)
    {
        // Compute tensor shape
        var shape = new List<int>();
        GetJaggedShape(values, shape, 0);

        // Flatten and copy elements
        var tensor_shape = new Shape(shape.ToArray());
        var tensor_elements = tensor_shape.LogicalElementCount();
        List<TNum> flat = new List<TNum>(tensor_elements); FlattenJagged(values, tensor_shape, 0, flat);
        if (flat.Count != tensor_elements)
            throw new ArgumentException($"Jagged array contained {flat.Count} values but tensor shape expected {tensor_elements} values");

        // Return tensor
        return Tensor<TNum>.FromFlattenedArray(tensor_shape, flat.ToArray());
    }
    /// <summary>
    /// Create a tensor from a C# jagged array (array of arrays)
    /// </summary>
    /// <param name="array">jagged array</param>
    /// <returns>tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> FromJaggedArray(TNum[] vector) => FromJaggedArray((Array)vector);
    /// <summary>
    /// Create a tensor from a C# jagged array (array of arrays)
    /// </summary>
    /// <param name="array">jagged array</param>
    /// <returns>tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> FromJaggedArray(TNum[][] matrix) => FromJaggedArray((Array)matrix);
    /// <summary>
    /// Create a tensor from a C# jagged array (array of arrays)
    /// </summary>
    /// <param name="array">jagged array</param>
    /// <returns>tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> FromJaggedArray(TNum[][][] featureSet) => FromJaggedArray((Array)featureSet);
    /// <summary>
    /// Create a tensor from a C# jagged array (array of arrays)
    /// </summary>
    /// <param name="array">jagged array</param>
    /// <returns>tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> FromJaggedArray(TNum[][][][] batchedFeatureSet) => FromJaggedArray((Array)batchedFeatureSet);
    private static void GetJaggedShape(object? item, List<int> shape, int dim_index)
    {
        if (item is not Array array)
            return;

        if (shape.Count > dim_index)
            shape[dim_index] = Math.Max(shape[dim_index], array.Length);
        else
            shape.Add(array.Length);

        foreach (var child in array)
        {
            GetJaggedShape(child, shape, dim_index + 1);
        }
    }
    private static void FlattenJagged(object? jagged, Shape shape, int dim_index, List<TNum> output)
    {
        if (jagged is Array arr)
        {
            // loop over elements (if the array is smaller than the required shape, pad with 0s aka recursive null; if larger ignore extra)
            // if shape was computed properly array length should always be <= dimension length
            for (var i = 0; i < shape.Length(dim_index); i++)
            {
                if (i < arr.Length)
                {
                    FlattenJagged(arr.GetValue(i), shape, dim_index + 1, output);
                }
                else
                {
                    FlattenJagged(null, shape, dim_index + 1, output);
                }
            }
        }
        else if (jagged is TNum value)
        {
            output.Add(value);
        }
        else
        {
            // Its not a TNum, 
            // Insert TNum.Zero for nulls or incompatible types
            output.Add(TNum.Zero);
        }
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to a value provided by a generating function
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Generate(Shape shape, Func<TNum> generator)
    {
        shape = shape.CloneDimensions();
        TNum[] elems = new TNum[shape.LogicalElementCount()];
        for (var i = 0; i < elems.Length; i++)
        {
            elems[i] = generator();                             // Generate a new value for this spot in the tensor
        }
        return new Tensor<TNum>(shape, elems);
    }

    /// <summary>
    /// Create a tensor of the given shape with all elements set to a value provided by sampling a probability distribution
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <param name="distribution">probability distribution</param>
    /// <returns>tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> Random(Shape shape, IProbabilityDistribution<TNum> distribution) => Generate(shape, distribution.Sample);

    /// <summary>
    /// Create a tensor of the given shape with elemements set to 0 except for the diagonal (same indices) set to 1
    /// </summary>
    /// <param name="shape">tensor shape</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Identity(Shape shape)
    {
        shape = shape.CloneDimensions();
        TNum[] elems = new TNum[shape.LogicalElementCount()];
        Array.Fill(elems, TNum.Zero);

        // For each identical index set to 1
        var len = shape.MinLength();                            // Since the tensor/matrix may not be square iterate along the smallest axis (not a real diagonal)
        Span<int> indices = stackalloc int[shape.Rank];       // ND indices to modify the individual index values of
        for (var i = 0; i < len; i++)
        {
            indices.Fill(i);                                    // ie if i is 5 and rank is 2 this is (5, 5)
            elems[shape.FlattenIndices(indices)] = TNum.One;    // Get the 1d index and set its value to 1
        }

        return new Tensor<TNum>(shape, elems);
    }

    /// <summary>
    /// Create a row matrix (2 dimensions, 1 row) from the given values
    /// </summary>
    /// <param name="values">values</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Row(Span<TNum> values)
    {
        return Tensor<TNum>.FromFlattenedArray(new Shape(1, values.Length), values);
    }

    /// <summary>
    /// Create a column matrix (2 dimensions, 1 column) from the given values
    /// </summary>
    /// <param name="values">values</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Column(Span<TNum> values)
    {
        return Tensor<TNum>.FromFlattenedArray(new Shape(values.Length, 1), values);
    }

    /// <summary>
    /// Create a vector (1 dimensions) from the given values
    /// </summary>
    /// <param name="values">values</param>
    /// <returns>tensor</returns>
    public static Tensor<TNum> Vec(Span<TNum> values)
    {
        return Tensor<TNum>.FromFlattenedArray(new Shape(values.Length), values);
    }

    /// <summary>
    /// Flattened index access to tensor elements
    /// </summary>
    /// <param name="index">flattened index</param>
    /// <returns>tensor element</returns>
    public TNum this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.elements[index];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            this.elements[index] = value;
        }
    }

    /// <summary>
    /// Multi-dimensional index access to tensor elements
    /// </summary>
    /// <param name="indices">Multi-dimensional index</param>
    /// <returns>tensor element</returns>
    public TNum this[params ReadOnlySpan<int> indices]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.elements[Shape.FlattenIndices(indices)];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            this.elements[Shape.FlattenIndices(indices)] = value;
        }
    }

    /// <summary>
    /// Multi-dimensional index access to tensor elements
    /// </summary>
    /// <param name="indices">Multi-dimensional index</param>
    /// <returns>tensor element</returns>
    public TNum this[params ReadOnlySpan<Index> indices]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Span<int> ints = stackalloc int[indices.Length];
            for (var i = 0; i < indices.Length; i++)
                ints[i] = indices[i].GetOffset(Shape.Length(i));
            return this.elements[Shape.FlattenIndices(ints)];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            Span<int> ints = stackalloc int[indices.Length];
            for (var i = 0; i < indices.Length; i++)
                ints[i] = indices[i].GetOffset(Shape.Length(i));
            this.elements[Shape.FlattenIndices(ints)] = value;
        }
    }

    /// <summary>
    /// Get a particular element from the tensor by index
    /// </summary>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public TNum GetElementAt(params int[] indices) => this[indices];

    /// <summary>
    /// Set a particular element in the tensor by index
    /// </summary>
    /// <param name="value">value to store</param>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public void SetElementAt(TNum value, params int[] indices) => this[indices] = value;

    /// <summary>
    /// Operator for .Slice
    /// </summary>
    /// <param name="ranges">ranges to slice over</param>
    /// <returns>sliced tensor</returns>
    public Tensor<TNum> this[params ReadOnlySpan<Range> ranges]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this.Slice(ranges);
    }

    /// <summary>
    /// Check if this tensor represents a scalar value (only 1)
    /// </summary>
    public bool IsScalar
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.elements.Length <= 1;
        }
    }

    /// <summary>
    /// Check if this tensor represents a vector (only 1 dimension)
    /// </summary>
    public bool IsVector
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.Shape.Rank == 1;
        }
    }

    /// <summary>
    /// Check if this tensor represents a matrix (2 dimensions, row & column)
    /// </summary>
    public bool IsMatrix
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.Shape.Rank == 2;
        }
    }

    /// <summary>
    /// Check if this tensor represents a row matrix (2 dimensions but only 1 row)
    /// </summary>
    public bool IsRowMatrix
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.Shape.Rank == 2 && this.Shape.Length(0) == 1;
        }
    }

    /// <summary>
    /// Check if this tensor represents a column matrix (2 dimensions but only 1 column)
    /// </summary>
    public bool IsColumnMatrix
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.Shape.Rank == 2 && this.Shape.Length(1) == 1;
        }
    }

    /// <summary>
    /// Check if this tensor represents a square matrix (2 dimensions of equal length)
    /// </summary>
    public bool IsSquareMatrix
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            return this.Shape.Rank == 2 && this.Shape.Length(1) == this.Shape.Length(0);
        }
    }

    /// <summary>
    /// Rank of the tensor (shortcut for Shape.Rank)
    /// </summary>
    public int Rank
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Shape.Rank;
    }

    /// <summary>
    /// Length/size of a particular dimension
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>dimension length</returns>
    public int GetDimension(int index) => Shape.Length(index);

    /// <summary>
    /// Number of elements in the tensor
    /// </summary>
    public int ElementCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this.elements.Length;
    }

    /// <summary>
    /// Reshape this tensor into another shape by creating a new tensor with the same elements but a the new shape
    /// </summary>
    /// <param name="shape">new shape</param>
    /// <returns>reshaped tensor</returns>
    /// <exception cref="Exception">thrown if the number of elements in the new shape do not match the current number of elements</exception>
    public Tensor<TNum> Reshape(Shape shape)
    {
        var element_count = shape.StorageElementCount();
        if (element_count != this.elements.Length)
            throw new Exception($"Cannot reshape to {shape}({element_count} elements) because the underlying storage of {this.elements.Length} elements is not compatible");

        return new Tensor<TNum>(shape, (TNum[])this.elements.Clone());
    }
    /// <summary>
    /// Reshape this tensor into another shape, modifying the existing shape rather than creating a new tensor
    /// </summary>
    /// <param name="shape">new shape</param>
    /// <returns>reshaped tensor</returns>
    /// <exception cref="Exception">thrown if the number of elements in the new shape do not match the current number of elements</exception>
    public void ReshapeInplace(Shape shape)
    {
        var element_count = shape.StorageElementCount();
        if (element_count != this.elements.Length)
            throw new Exception($"Cannot reshape to {shape}({element_count} elements) because the underlying storage of {this.elements.Length} elements is not compatible");

        this.Shape = shape;
    }
    /// <summary>
    /// Reshape this tensor into another shape, this tensor will share its underlying array with the original tensor and as such one should be careful of modifying the resulting tensor instance
    /// </summary>
    /// <param name="shape">new shape</param>
    /// <returns>reshaped tensor</returns>
    /// <exception cref="Exception">thrown if the number of elements in the new shape do not match the current number of elements</exception>
    public Tensor<TNum> ReshapeShared(Shape shape)
    {
        var element_count = shape.StorageElementCount();
        if (element_count != this.elements.Length)
            throw new Exception($"Cannot reshape to {shape}({element_count} elements) because the underlying storage of {this.elements.Length} elements is not compatible");

        return new Tensor<TNum>(shape, this.elements);
    }

    /// <summary>
    /// Perform an elementwise transformation of the tensor elements
    /// </summary>
    /// <param name="transformation">transformation</param>
    /// <returns>tensor with same shape but transformed elements</returns>
    public Tensor<TResult> ElementWise<TResult>(Func<TNum, TResult> transformation)
    where TResult : INumber<TResult>
    {
        int spanlength = this.elements.Length;
        var chunksize = Math.Max(spanlength / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (spanlength + chunksize - 1) / chunksize;

        var tensor = new TResult[this.elements.Length];

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, spanlength);
            int length = end - start;

            Span<TResult> output = tensor.AsSpan(start, length);
            ReadOnlySpan<TNum> elements = this.elements.AsSpan(start, length);
            for (var i = 0; i < length; i++)
            {
                output[i] = transformation(elements[i]);
            }
        });
        
        return new Tensor<TResult>(this.Shape, tensor);
    }
    /// <summary>
    /// Perform an elementwise transformation of the tensor elements storing the results in-place
    /// </summary>
    /// <param name="transformation">transformation</param>
    public void ElementWiseInplace(Func<TNum, TNum> transformation)
    {
        int spanlength = this.elements.Length;
        var chunksize = Math.Max(spanlength / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (spanlength + chunksize - 1) / chunksize;

        var tensor = this.elements;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, spanlength);
            int length = end - start;

            Span<TNum> output = tensor.AsSpan(start, length);
            ReadOnlySpan<TNum> elements = this.elements.AsSpan(start, length);
            for (var i = 0; i < length; i++)
            {
                output[i] = transformation(elements[i]);
            }
        });
    }
    /// <summary>
    /// <para>
    /// Perform an elementwise transformation using a vectorized function (SIMD)
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// var result = tensor.ElementWise(
    ///     v => Vector.SquareRoot(v),
    ///     x => TNum.Sqrt(x)
    /// );
    /// </code>
    /// </para>
    /// </summary>
    /// <param name="vectorizedTransformation">SIMD transformation</param>
    /// <param name="scalarTransformation">Scalar fallback transformation</param>
    /// <returns>tensor with same shape but transformed elements</returns>
    public Tensor<TNum> ElementWise(
        Func<Vector<TNum>, Vector<TNum>> vectorizedTransformation,
        Func<TNum, TNum> scalarTransformation)
    {
        var src = this.elements;
        var len = src.Length;
        var dst = new TNum[len];

        var chunksize = Math.Max(len / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (len + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, len);
            int length = end - start;

            // Vector part
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;
                for (; i < simdLimit; i += simdLength)
                {
                    var v = new Vector<TNum>(src, start + i);
                    vectorizedTransformation(v).CopyTo(dst, start + i);
                }
            }
            // Scalar fallback for remaining elements
            for (; i < length; i++)
            {
                dst[start + i] = scalarTransformation(src[start + i]);
            }
        });

        return new Tensor<TNum>(this.Shape, dst);
    }

    /// <summary>
    /// <para>
    /// Perform an elementwise in-place transformation using a vectorized function (SIMD)
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// tensor.ElementWiseInplace(
    ///     v => Vector.SquareRoot(v),
    ///     x => TNum.Sqrt(x)
    /// );
    /// </code>
    /// </para>
    /// </summary>
    /// <param name="vectorizedTransformation">SIMD transformation</param>
    /// <param name="scalarTransformation">Scalar fallback transformation</param>
    public void ElementWiseInplace(
        Func<Vector<TNum>, Vector<TNum>> vectorizedTransformation,
        Func<TNum, TNum> scalarTransformation)
    {
        var src = this.elements;
        var dst = this.elements;
        var len = dst.Length;

        var chunksize = Math.Max(len / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (len + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, len);
            int length = end - start;

            // Vector part
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;
                for (; i < simdLimit; i += simdLength)
                {
                    var v = new Vector<TNum>(src, start + i);
                    vectorizedTransformation(v).CopyTo(dst, start + i);
                }
            }
            // Scalar fallback for remaining elements
            for (; i < length; i++)
            {
                dst[start + i] = scalarTransformation(src[start + i]);
            }
        });
    }

    /// <summary>
    /// Perform an elementwise transformation of the tensor elements with another tensor's elements
    /// </summary>
    /// <typeparam name="TNumOther">2nd tensor's element type</typeparam>
    /// <typeparam name="TNumResult">resulting tensor's element type</typeparam>
    /// <param name="other">tensor to perform elementwise operations with</param>
    /// <param name="transformation">transformation</param>
    /// <returns>tensor with the same shape but transformed elements</returns>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public Tensor<TNumResult> ElementWiseBinary<TNumOther, TNumResult>(Tensor<TNumOther> other, Func<TNum, TNumOther, TNumResult> transformation)
    where TNumOther : INumber<TNumOther>
    where TNumResult : INumber<TNumResult>
    {
        if (!this.Shape.Equals(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var srcA = this.elements;
        var srcB = other.elements;
        var dst = new TNumResult[this.elements.Length];
        var len = dst.Length;

        var chunksize = Math.Max(len / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (len + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, len);
            int length = end - start;

            for (var i = 0; i < length; i++)
            {
                dst[start + i] = transformation(srcA[start + i], srcB[i]);
            }
        });

        return new Tensor<TNumResult>(this.Shape, dst);
    }
    /// <summary>
    /// Perform an elementwise transformation of the tensor elements with another tensor's elements
    /// </summary>
    /// <typeparam name="TNumOther">2nd tensor's element type</typeparam>
    /// <typeparam name="TNumResult">resulting tensor's element type</typeparam>
    /// <param name="other">tensor to perform elementwise operations with</param>
    /// <param name="vectorizedTransformation">transformation to use for vectorizable components</param>
    /// <param name="scalarTransformation">transformation to use for scalar components</param>
    /// <returns>tensor with the same shape but transformed elements</returns>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public Tensor<TNumResult> ElementWiseBinary<TNumOther, TNumResult>(
        Tensor<TNumOther> other,
        Func<Vector<TNum>, Vector<TNumOther>, Vector<TNumResult>> vectorizedTransformation,
        Func<TNum, TNumOther, TNumResult> scalarTransformation
    )
    where TNumOther : INumber<TNumOther>
    where TNumResult : INumber<TNumResult>
    {
        if (!this.Shape.Equals(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var srcA = this.elements;
        var srcB = other.elements;
        var dst = new TNumResult[this.elements.Length];
        var len = dst.Length;

        var chunksize = Math.Max(len / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (len + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, len);
            int length = end - start;

            // Vector part
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;
                for (; i < simdLimit; i += simdLength)
                {
                    var aVec = new Vector<TNum>(srcA, start + i);
                    var bVec = new Vector<TNumOther>(srcB, start + i);
                    vectorizedTransformation(aVec, bVec).CopyTo(dst, start + i);
                }
            }
            // Scalar fallback for remaining elements
            for (; i < length; i++)
            {
                dst[start + i] = scalarTransformation(srcA[start + i], srcB[start + i]);
            }

        });

        return new Tensor<TNumResult>(this.Shape, dst);
    }
    /// <summary>
    /// Perform an elementwise transformation of the tensor elements with another tensor's elements storing the results in-place
    /// </summary>
    /// <typeparam name="TNumOther">2nd tensor's element type</typeparam>
    /// <param name="other">tensor to perform elementwise operations with</param>
    /// <param name="transformation">transformation</param>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public void ElementWiseBinaryInplace<TNumOther>(Tensor<TNumOther> other, Func<TNum, TNumOther, TNum> transformation)
    where TNumOther : INumber<TNumOther>
    {

        if (!this.Shape.Equals(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var srcA = this.elements;
        var srcB = other.elements;
        var dst = this.elements;
        var len = dst.Length;

        var chunksize = Math.Max(len / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (len + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, len);
            int length = end - start;

            for (var i = 0; i < length; i++)
            {
                dst[start + i] = transformation(srcA[start + i], srcB[start + i]);
            }
        });
    }

    /// <summary>
    /// Perform an elementwise transformation of the tensor elements with two other tensor's elements
    /// </summary>
    /// <typeparam name="TNumOtherA">2nd tensor's element type</typeparam>
    /// <typeparam name="TNumOtherB">3rd tensor's element type</typeparam>
    /// <typeparam name="TNumResult">resulting tensor's element type</typeparam>
    /// <param name="otherA">2nd tensor</param>
    /// <param name="otherB">3rd tensor</param>
    /// <param name="transformation">transformation</param>
    /// <returns>tensor with the same shape but transformed elements</returns>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensors are incompatible with this tensor</exception>
    public Tensor<TNumResult> ElementWiseTernary<TNumOtherA, TNumOtherB, TNumResult>(Tensor<TNumOtherA> otherA, Tensor<TNumOtherB> otherB, Func<TNum, TNumOtherA, TNumOtherB, TNumResult> transformation)
    where TNumOtherA : INumber<TNumOtherA>
    where TNumOtherB : INumber<TNumOtherB>
    where TNumResult : INumber<TNumResult>
    {
        if (!this.Shape.Equals(otherA.Shape) || !this.Shape.Equals(otherB.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var tensor = new TNumResult[this.elements.Length];
        for (var i = 0; i < tensor.Length; i++)
        {
            tensor[i] = transformation(this.elements[i], otherA.elements[i], otherB.elements[i]);
        }
        return new Tensor<TNumResult>(this.Shape, tensor);
    }
    /// <summary>
    /// Perform an elementwise transformation of the tensor elements with two other tensor's elements
    /// </summary>
    /// <typeparam name="TNumOtherA">2nd tensor's element type</typeparam>
    /// <typeparam name="TNumOtherB">3rd tensor's element type</typeparam>
    /// <param name="otherA">2nd tensor</param>
    /// <param name="otherB">3rd tensor</param>
    /// <param name="transformation">transformation</param>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensors are incompatible with this tensor</exception>
    public void ElementWiseTernary<TNumOtherA, TNumOtherB>(Tensor<TNumOtherA> otherA, Tensor<TNumOtherB> otherB, Func<TNum, TNumOtherA, TNumOtherB, TNum> transformation)
    where TNumOtherA : INumber<TNumOtherA>
    where TNumOtherB : INumber<TNumOtherB>
    {
        if (!this.Shape.Equals(otherA.Shape) || !this.Shape.Equals(otherB.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var tensor = this.elements;
        for (var i = 0; i < tensor.Length; i++)
        {
            tensor[i] = transformation(this.elements[i], otherA.elements[i], otherB.elements[i]);
        }
    }

    /// <summary>
    /// Clip (limit) the values of the tensor to be within the given min and max values
    /// </summary>
    /// <param name="min">minimum value</param>
    /// <param name="max">maximum value</param>
    public void Clip(TNum min, TNum max)
    {
        if (min > max)
            (max, min) = (min, max); // Swap

        var elements = this.elements;
        for (var i = 0; i < elements.Length; i++)
        {
            if (elements[i] < min)
                elements[i] = min;
            else if (elements[i] > max)
                elements[i] = max;
        }
    }

    /// <summary>
    /// Clip (limit) the values of the tensor to be greater than the given minimum value
    /// </summary>
    /// <param name="min">minimum value</param>
    public void ClipMinimum(TNum min)
    {
        var elements = this.elements;
        for (var i = 0; i < elements.Length; i++)
        {
            if (elements[i] < min)
                elements[i] = min;
        }
    }

    /// <summary>
    /// Clip (limit) the values of the tensor to be smaller than the given max value
    /// </summary>
    /// <param name="max">maximum value</param>
    public void ClipMaximum(TNum max)
    {
        var elements = this.elements;
        for (var i = 0; i < elements.Length; i++)
        {
            if (elements[i] > max)
                elements[i] = max;
        }
    }


    /// <summary>
    /// Elementwise absolute value
    /// </summary>
    /// <returns>tensor with each element as its absolute value</returns>
    public Tensor<TNum> Abs()
    {
        var length = this.elements.Length;
        var tensor = new TNum[length];

        // Vectorized elements
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            int simdLength = Vector<TNum>.Count;
            int simdLimit = length - simdLength + 1;

            for (; i < simdLimit; i += simdLength)
            {
                var va = new Vector<TNum>(this.elements, i);
                Vector.Abs(va).CopyTo(tensor, i);
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Abs(this.elements[i]);
        }
        return new Tensor<TNum>(this.Shape, tensor);
    }
    /// <summary>
    /// Elementwise absolute value in-place
    /// </summary>
    /// <returns>tensor with each element as its absolute value</returns>
    public void AbsInplace()
    {
        var length = this.elements.Length;
        var tensor = this.elements;

        // Vectorized elements
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            int simdLength = Vector<TNum>.Count;
            int simdLimit = length - simdLength + 1;

            for (; i < simdLimit; i += simdLength)
            {
                var va = new Vector<TNum>(this.elements, i);
                Vector.Abs(va).CopyTo(tensor, i);
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Abs(this.elements[i]);
        }
    }

    /// <summary>
    /// Elementwise negation value
    /// </summary>
    /// <returns>tensor with each element negated</returns>
    public Tensor<TNum> Negate()
    {
        var length = this.elements.Length;
        var tensor = new TNum[length];

        // Vectorized elements
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            int simdLength = Vector<TNum>.Count;
            int simdLimit = length - simdLength + 1;

            for (; i < simdLimit; i += simdLength)
            {
                var va = new Vector<TNum>(this.elements, i);
                Vector.Negate(va).CopyTo(tensor, i);
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = -this.elements[i];
        }
        return new Tensor<TNum>(this.Shape, tensor);
    }
    /// <summary>
    /// Elementwise negation value
    /// </summary>
    /// <returns>tensor with each element negated</returns>
    public void NegateInplace()
    {
        var length = this.elements.Length;
        var tensor = this.elements;

        // Vectorized elements
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            int simdLength = Vector<TNum>.Count;
            int simdLimit = length - simdLength + 1;

            for (; i < simdLimit; i += simdLength)
            {
                var va = new Vector<TNum>(this.elements, i);
                Vector.Negate(va).CopyTo(tensor, i);
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = -this.elements[i];
        }
    }

    /// <summary>
    /// Operator for .Negate 
    /// </summary>
    /// <param name="value">tensor to negate</param>
    /// <returns>element-wise tensor negation</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> operator -(Tensor<TNum> value) => value.Negate();

    /// <summary>
    /// Elementwise addition
    /// </summary>
    /// <param name="other">tensor to be added to</param>
    /// <returns>element-wise tensor addition</returns>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public Tensor<TNum> AddWith(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = new TNum[length];

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseAdd(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            );
        }

        return new Tensor<TNum>(this.Shape, tensor);
    }
    /// <summary>
    /// Elementwise addition in-place
    /// </summary>
    /// <param name="other">tensor to be added to</param>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public void AddWithInplace(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = this.elements;

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseAdd(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            ); // Get's inlined here for maximum performance
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ElementWiseAdd(
        int spanlength, TNum[] output, int oOffset, TNum[] lhs, int lOffset, TNum[] rhs, int rOffset
    )
    {
        var chunksize = Math.Max(spanlength / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (spanlength + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, spanlength);
            int length = end - start;

            // Vectorized elements
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;

                for (; i < simdLimit; i += simdLength)
                {
                    var va = new Vector<TNum>(lhs, lOffset + start + i);
                    var vb = new Vector<TNum>(rhs, rOffset + start + i);
                    var vr = va + vb;
                    vr.CopyTo(output, oOffset + start + i);
                }
            }
            // Remaining elements
            for (; i < length; i++)
            {
                output[oOffset + start + i] = lhs[lOffset + start + i] + rhs[rOffset + start + i];
            }
        });
    }

    /// <summary>
    /// Operator for .AddWith 
    /// </summary>
    /// <param name="lhs">left-hand side tensor</param>
    /// <param name="rhs">right-hand side tensor</param>
    /// <returns>element-wise tensor addition</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> operator +(Tensor<TNum> lhs, Tensor<TNum> rhs) => lhs.AddWith(rhs);

    /// <summary>
    /// Elementwise subtraction
    /// </summary>
    /// <param name="other">tensor to be subtracted with</param>
    /// <returns>element-wise tensor subtraction</returns>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public Tensor<TNum> SubtractWith(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = new TNum[length];

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseSubtract(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            ); // Get's inlined here for maximum performance
        }

        return new Tensor<TNum>(this.Shape, tensor);
    }
    /// <summary>
    /// Elementwise subtraction in-place
    /// </summary>
    /// <param name="other">tensor to be subtracted with</param>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public void SubtractWithInplace(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = this.elements;

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseSubtract(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            ); // Get's inlined here for maximum performance
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ElementWiseSubtract(
        int spanlength, TNum[] output, int oOffset, TNum[] lhs, int lOffset, TNum[] rhs, int rOffset
    )
    {
        var chunksize = Math.Max(spanlength / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (spanlength + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, spanlength);
            int length = end - start;

            // Vectorized elements
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;

                for (; i < simdLimit; i += simdLength)
                {
                    var va = new Vector<TNum>(lhs, lOffset + start + i);
                    var vb = new Vector<TNum>(rhs, rOffset + start + i);
                    var vr = va - vb;
                    vr.CopyTo(output, oOffset + start + i);
                }
            }
            // Remaining elements
            for (; i < length; i++)
            {
                output[oOffset + start + i] = lhs[lOffset + start + i] - rhs[rOffset + start + i];
            }
        });
    }


    /// <summary>
    /// Operator for .SubtractWith 
    /// </summary>
    /// <param name="lhs">left-hand side tensor</param>
    /// <param name="rhs">right-hand side tensor</param>
    /// <returns>element-wise tensor subtraction</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> operator -(Tensor<TNum> lhs, Tensor<TNum> rhs) => lhs.SubtractWith(rhs);

    /// <summary>
    /// Elementwise multiplication
    /// </summary>
    /// <param name="other">tensor to be multiplied with</param>
    /// <returns>element-wise tensor multiplication</returns>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public Tensor<TNum> HadamardWith(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = new TNum[length];

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseMultiply(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            ); // Get's inlined here for maximum performance
        }

        return new Tensor<TNum>(this.Shape, tensor);
    }
    /// <summary>
    /// Elementwise multiplication in-place
    /// </summary>
    /// <param name="other">tensor to be multiplied with</param>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public void HadamardWithInplace(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = this.elements;

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseMultiply(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            ); // Get's inlined here for maximum performance
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ElementWiseMultiply(
        int spanlength, TNum[] output, int oOffset, TNum[] lhs, int lOffset, TNum[] rhs, int rOffset
    )
    {
        var chunksize = Math.Max(spanlength / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (spanlength + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, spanlength);
            int length = end - start;

            // Vectorized elements
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;

                for (; i < simdLimit; i += simdLength)
                {
                    var va = new Vector<TNum>(lhs, lOffset + start + i);
                    var vb = new Vector<TNum>(rhs, rOffset + start + i);
                    var vr = va * vb;
                    vr.CopyTo(output, oOffset + start + i);
                }
            }
            // Remaining elements
            for (; i < length; i++)
            {
                output[oOffset + start + i] = lhs[lOffset + start + i] * rhs[rOffset + start + i];
            }
        });
    }

    /// <summary>
    /// Scale all elements of the tensor
    /// </summary>
    /// <param name="other">scalar to multiply values with</param>
    /// <returns>element-wise tensor multiplication</returns>
    public Tensor<TNum> ScaleBy(TNum other)
    {
        var src = this.elements;
        var len = src.Length;
        var dst = new TNum[len];

        var chunksize = Math.Max(len / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (len + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, len);
            int length = end - start;

            // Vectorized elements
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;
                Vector<TNum> constant = new Vector<TNum>(other);

                for (; i < simdLimit; i += simdLength)
                {
                    var va = new Vector<TNum>(src, start + i);
                    var vr = va * constant;
                    vr.CopyTo(dst, start + i);
                }
            }
            // Remaining elements
            for (; i < length; i++)
            {
                dst[start + i] = src[start + i] * other;
            }
        });

        return new Tensor<TNum>(this.Shape, dst);
    }
    /// <summary>
    /// Scale all elements of the tensor in-place
    /// </summary>
    /// <param name="other">scalar to multiply values with</param>
    public void ScaleByInplace(TNum other)
    {
        var src = this.elements;
        var len = src.Length;
        var dst = this.elements;

        var chunksize = Math.Max(len / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (len + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, len);
            int length = end - start;

            // Vectorized elements
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;
                Vector<TNum> constant = new Vector<TNum>(other);

                for (; i < simdLimit; i += simdLength)
                {
                    var va = new Vector<TNum>(src, start + i);
                    var vr = va * constant;
                    vr.CopyTo(dst, start  +i);
                }
            }
            // Remaining elements
            for (; i < length; i++)
            {
                dst[start + i] = src[start + i] * other;
            }
        });
    }

    /// <summary>
    /// Operator for .ScaleBy 
    /// </summary>
    /// <param name="lhs">left-hand side tensor</param>
    /// <param name="rhs">right-hand side scalar value</param>
    /// <returns>element-wise tensor multiplication</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> operator *(Tensor<TNum> lhs, TNum rhs) => lhs.ScaleBy(rhs);

    /// <summary>
    /// Operator for .ScaleBy 
    /// </summary>
    /// <param name="lhs">left-hand side tensor</param>
    /// <param name="rhs">right-hand side scalar value</param>
    /// <returns>element-wise tensor division</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> operator /(Tensor<TNum> lhs, TNum rhs) => lhs.ScaleBy(TNum.One / rhs);

    /// <summary>
    /// Operator for .ScaleBy 
    /// </summary>
    /// <param name="lhs">left-hand side tensor</param>
    /// <param name="rhs">right-hand side scalar value</param>
    /// <returns>element-wise tensor multiplication</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> operator *(TNum lhs, Tensor<TNum> rhs) => rhs.ScaleBy(lhs);

    /// <summary>
    /// Elementwise division
    /// </summary>
    /// <param name="other">tensor to be divided with</param>
    /// <returns>element-wise tensor division</returns>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public Tensor<TNum> ElementWiseDivisionWith(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = new TNum[length];

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseDivision(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            ); // Get's inlined here for maximum performance
        }

        return new Tensor<TNum>(this.Shape, tensor);
    }
    /// <summary>
    /// Elementwise division in-place
    /// </summary>
    /// <param name="other">tensor to be divided with</param>
    /// <exception cref="InvalidOperationException">thrown when the shape of the other tensor is incompatible with this tensor</exception>
    public void ElementWiseDivisionInplaceWith(Tensor<TNum> other)
    {
        if (!this.Shape.AreTrailingDimensions(other.Shape))
        {
            throw new InvalidOperationException("Tensors have incompatible dimensions for elementwise operations");
        }

        var length = this.elements.Length;
        var tensor = this.elements;

        var slice_length = other.Shape.LogicalElementCount();
        var slice_count = length / slice_length; // Should be a whole number because the last dimensions are shared and any batch dimensions on this are just multiples of the last dimensions

        for (var i = 0; i < slice_count; i++)
        {
            ElementWiseDivision(
                slice_length,
                tensor,
                i * slice_length,
                this.elements,
                i * slice_length,
                other.elements,
                0
            ); // Get's inlined here for maximum performance
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ElementWiseDivision(
        int spanlength, TNum[] output, int oOffset, TNum[] lhs, int lOffset, TNum[] rhs, int rOffset
    )
    {
        var chunksize = Math.Max(spanlength / DegreesOfParallelism, MinParallelChunkSize);
        int chunkCount = (spanlength + chunksize - 1) / chunksize;

        Parallel.For(0, chunkCount, (chunkIdx) =>
        {
            // Start, end, length within region spanned by oOffset + spanlength etc.
            int start = chunkIdx * chunksize;
            int end = Math.Min(start + chunksize, spanlength);
            int length = end - start;

            // Vectorized elements
            int i = 0;
            if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
            {
                int simdLength = Vector<TNum>.Count;
                int simdLimit = length - simdLength + 1;

                for (; i < simdLimit; i += simdLength)
                {
                    var va = new Vector<TNum>(lhs, lOffset + start + i);
                    var vb = new Vector<TNum>(rhs, rOffset + start + i);
                    var vr = va * vb;
                    vr.CopyTo(output, oOffset + start + i);
                }
            }
            // Remaining elements
            for (; i < length; i++)
            {
                output[oOffset + start + i] = lhs[lOffset + start + i] * rhs[rOffset + start + i];
            }
        });
    }

    /// <summary>
    /// Shift all values over by one to insert this value onto the first element of the tensor
    /// </summary>
    /// <param name="first">new first element</param>
    public void ShiftOntoBeginning(TNum first)
    {
        TNum temp = first;
        for (var i = 0; i < this.elements.Length; i++)
        {
            TNum next = this.elements[i];
            this.elements[i] = temp;
            temp = next;
        }
    }

    /// <summary>
    /// Shift all values back by one to make this element the new last element of the tensor
    /// </summary>
    /// <param name="last">new last element</param>
    public void ShiftOntoEnd(TNum last)
    {
        TNum temp = last;
        for (var i = this.elements.Length - 1; i >= 0; i--)
        {
            TNum next = this.elements[i];
            this.elements[i] = temp;
            temp = next;
        }
    }

    /// <summary>
    /// Operator for .ShiftOntoEnd
    /// </summary>
    /// <param name="first">tensor</param>
    /// <param name="last">element to shift onto the end of the tensor</param>
    /// <returns>original tensor modified with the new values</returns>
    public static Tensor<TNum> operator <<(Tensor<TNum> first, TNum last)
    {
        first.ShiftOntoEnd(last);
        return first;
    }

    /// <summary>
    /// Simple matrix multiplication, tensors must be of rank 2
    /// </summary>
    /// <param name="other">tensor to multiply with</param>
    /// <returns>result of matrix multiplication</returns>
    /// <exception cref="InvalidOperationException">thrown if rank is invalid or dimensions are not compatible with matrix multiplication</exception>
    public Tensor<TNum> MatMul(Tensor<TNum> other)
    {
        var a = this;
        var b = other;

        if (a.Shape.Rank != 2 || a.Shape.Rank != b.Shape.Rank)
            throw new InvalidOperationException("Shape rank mismatch for matrix multiplication, number of dimensions must be equal to 2");

        var row_idx = a.Shape.Rank - 2;
        var col_idx = a.Shape.Rank - 1;
        int a_rows = a.Shape.Length(row_idx);
        int a_cols = a.Shape.Length(col_idx);
        int b_rows = b.Shape.Length(row_idx);
        int b_cols = b.Shape.Length(col_idx);

        ReadOnlySpan<TNum> a_span = a.AsSpan();
        ReadOnlySpan<TNum> b_span = b.AsSpan();

        if (a_cols != b_rows)
            throw new InvalidOperationException("Inner dimensions are not compatible for matrix multiplication");

        int rows = a_rows;
        int cols = b_cols;
        int innerDim = a_cols;

        var result = Tensor<TNum>.Zeros(new Shape(rows, cols));
        var r_span = result.AsSpan();

        MatMul(a_span, a_rows, a_cols, b_span, b_rows, b_cols, r_span);

        return result;
    }

    /// <summary>
    /// Performs matrix multiplication as if this matrix was transposed before multiplication
    /// </summary>
    /// <param name="other">tensor to multiply with</param>
    /// <returns>result of multiplying this transposed with other</returns>
    public Tensor<TNum> TransposedMatMul(Tensor<TNum> other)
    {
        var a = this;

        int m = a.Shape.Length(0); // M
        int k = a.Shape.Length(1); // K

        int b_m = other.Shape.Length(0); // M (must match A's first dim)
        int n = other.Shape.Length(1);   // N

        if (a.Rank != 2 || other.Rank != 2 || m != b_m)
            throw new ArithmeticException($"Incompatible shapes for Aᵗ * B: A[{m},{k}]ᵗ * B[{b_m},{n}]");

        // Result: [K, N]
        var result = Tensor<TNum>.Zeros(new Shape(k, n));
        var aSpan = a.AsSpan();        // [M x K], row-major
        var bSpan = other.AsSpan();    // [M x N], row-major
        var resSpan = result.AsSpan(); // [K x N], row-major

        int aStride = k;
        int bStride = n;
        int rStride = n;

        // k, j, i ordering. optimal for transposed multiplication
        Parallel.For(0, k, ParallelOptions, (k) =>
        {
            var resultSpan = result.AsSpan();
            var aSpan = a.AsSpan();
            var bSpan = other.AsSpan();

            // aRow: this is A's k-th column, i.e. Aᵗ[k] = A[:,k]
            int aColOffset = k; // offset within each A[i,k] = A[i * aStride + k]

            // B's k-th row (B[k,:]) — sequential access
            int bRowOffset = k * bStride;

            for (int j = 0; j < n; j++) // each col in B
            {
                TNum sum = TNum.Zero;
                for (int i = 0; i < m; i++) // shared dim
                {
                    TNum aVal = aSpan[i * aStride + k]; // A[i, p] → Aᵗ[p, i]
                    TNum bVal = bSpan[i * bStride + j]; // B[i, j]
                    sum += aVal * bVal;
                }

                resultSpan[k * rStride + j] = sum;
            }
        });
        return result;
    }

    /// <summary>
    /// Simple matrix/vector multiplication, matrix must be of rank 2 and have equal number of columns to vector length
    /// </summary>
    /// <param name="vector">vector to multiply with</param>
    /// <param name="bias">bias vector</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TNum> MatMulVector(Vec<TNum> vector, ReadOnlySpan<TNum> bias = default) => this.MatMulVector(vector.AsSpan(), bias);

    /// <summary>
    /// Simple matrix/vector multiplication, matrix must be of rank 2 and have equal number of columns to vector length
    /// </summary>
    /// <param name="vector">vector to multiply with</param>
    /// <param name="bias">bias vector</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public Tensor<TNum> MatMulVector(ReadOnlySpan<TNum> vector, ReadOnlySpan<TNum> bias = default)
    {
        if (this.Shape.Rank != 2 || this.Shape.Length(^1) != vector.Length)
            throw new InvalidOperationException("Shape or rank mismatch for matrix/vector multiplication");

        int rows = this.Shape.Length(^2);
        int innerDim = vector.Length;

        var mul = Tensor<TNum>.Zeros(new Shape(rows));
        var result = mul.AsSpan();

        if (!bias.TryCopyTo(result))
            throw new InvalidOperationException("Bias vector could not be added to results due to length mismatch");

        for (var i = 0; i < rows; i++)
        {
            TNum sum = result[i];
            var row = this.AsSpan(i * innerDim, innerDim);
            for (int j = 0; j < innerDim; j++)
            {
                sum += row[j] * vector[j];
            }
            result[i] = sum;
        }

        return mul;
    }

    /// <summary>
    /// Simple matrix/vector multiplication, matrix must be of rank 2 and have equal number of columns to vector length. The last dimension is used as the vector dimension and the preceding dimensions as batches.
    /// </summary>
    /// <param name="batchedVectors">batch of vectors to multiply with</param>
    /// <param name="bias">bias vector</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TNum> MatMulEachVector(Tensor<TNum> batchedVectors, TNum[]? bias = default) => MatMulEachVector(^1, batchedVectors, bias);

    /// <summary>
    /// Simple matrix/vector multiplication, matrix must be of rank 2 and have equal number of columns to vector length
    /// </summary>
    /// <param name="dimension">The dimension indicating the start of the vector</param>
    /// <param name="batchedVectors">batch of vectors to multiply with</param>
    /// <param name="bias">bias vector</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public Tensor<TNum> MatMulEachVector(Index dimension, Tensor<TNum> batchedVectors, TNum[]? bias = default)
    {
        var axis = batchedVectors.NormalizeAxis(dimension);
        var vecLength = batchedVectors.Shape.Length(axis) * batchedVectors.Shape.Stride(axis);
        var batches = axis == 0 ? 1 : batchedVectors.Shape.Length(0..axis);
        if (this.Shape.Rank != 2 || this.Shape.Length(^1) != vecLength)
            throw new InvalidOperationException($"Matrix {this.Shape} must be 2D and its column count ({this.Shape.Length(^1)}) must match vector length ({vecLength})");

        int rows = this.Shape.Length(^2);
        int innerDim = vecLength;

        var mul = Tensor<TNum>.Zeros(batchedVectors.Shape.SliceAndAppend(0..dimension, rows));

        if (bias is not null && bias.Length != rows)
            throw new InvalidOperationException("Bias vector could not be added to results due to length mismatch");

        Parallel.For(0, batches, ParallelOptions, (b) =>
        {
            var result = mul.AsSpan(b * rows, rows);
            var vector = batchedVectors.AsSpan(b * vecLength, vecLength);

            bias?.CopyTo(result);
            for (var i = 0; i < rows; i++)
            {
                TNum sum = result[i];
                var row = this.AsSpan(i * innerDim, innerDim);
                for (int j = 0; j < innerDim; j++)
                {
                    sum += row[j] * vector[j];
                }
                result[i] = sum;
            }
        });

        return mul;
    }

    /// <summary>
    /// Performs matrix multiplication between this matrix and each matrix in the batch dimensions of the other tensor.
    /// Equivalent to applying the same matrix multiply across a batch.
    /// </summary>
    /// <param name="other">tensor to multiply with</param>
    /// <returns>result of matrix multiplication</returns>
    /// <exception cref="InvalidOperationException">thrown if rank is invalid or dimensions are not compatible with matrix multiplication</exception>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public Tensor<TNum> MatMulEach(Tensor<TNum> other)
    {
        var a = this; // A single matrix
        var b = other;

        if (a.Shape.Rank != 2 || b.Shape.Rank < 2)
            throw new InvalidOperationException("This tensor must be rank 2, and 'other' tensor must be at least rank 2");

        var a_row_idx = a.Shape.Rank - 2;
        var a_col_idx = a.Shape.Rank - 1;
        int a_rows = a.Shape.Length(a_row_idx);
        int a_cols = a.Shape.Length(a_col_idx);
        var b_row_idx = b.Shape.Rank - 2;
        var b_col_idx = b.Shape.Rank - 1;
        int b_rows = b.Shape.Length(b_row_idx);
        int b_cols = b.Shape.Length(b_col_idx);

        ReadOnlySpan<TNum> a_span = a.AsSpan();
        ReadOnlySpan<TNum> b_span = b.AsSpan();

        if (a_cols != b_rows)
            throw new InvalidOperationException("Inner dimensions are not compatible for matrix multiplication");

        int rows = a_rows;
        int cols = b_cols;
        int innerDim = a_cols;

        var shape = new int[other.Rank];
        shape[^2] = rows;
        shape[^1] = cols;
        for (var i = 0; i < shape.Length - 2; i++)
        {
            shape[i] = other.Shape.Length(i);
        }

        var result = Tensor<TNum>.Zeros(new Shape(shape));
        var r_span = result.AsSpan();
        var r_batch_size = rows * cols;

        var b_batch_size = b_rows * b_cols;
        var batches = b_span.Length / b_batch_size;
        for (var batch = 0; batch < batches; batch++)
        {
            MatMul(a_span, a_rows, a_cols, b_span.Slice(batch * b_batch_size, b_batch_size), b_rows, b_cols, r_span.Slice(batch * r_batch_size, r_batch_size));
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MatMul(
        ReadOnlySpan<TNum> a, int a_rows, int a_cols,
        ReadOnlySpan<TNum> b, int b_rows, int b_cols,
        Span<TNum> result
    )
    {
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            int vecSize = Vector<TNum>.Count;

            // Do matrix multiplication
            // K-Major loop (keeps all access row-major)
            for (var k = 0; k < a_cols; k++)
            {
                for (var i = 0; i < a_rows; i++)
                {
                    var a_val = a[i * a_cols + k];
                    var a_vec = new Vector<TNum>(a_val);

                    int j = 0;
                    for (; j <= b_cols - vecSize; j += vecSize)
                    {
                        int b_base = k * b_cols + j;
                        int r_base = i * b_cols + j;

                        var b_vec = new Vector<TNum>(b.Slice(b_base, vecSize)); // May be able to be replaced by (Unsafe.As<TNum, TNum[]>(ref Unsafe.Add(ref bRef, b_base)), 0)
                        var r_vec = new Vector<TNum>(result.Slice(r_base, vecSize));

                        var result_vec = r_vec + a_vec * b_vec;

                        result_vec.CopyTo(result.Slice(r_base, vecSize));
                    }

                    // Handle leftover elements (scalar fallback)
                    for (; j < b_cols; j++)
                    {
                        int b_idx = k * b_cols + j;
                        int r_idx = i * b_cols + j;

                        result[r_idx] += a_val * b[b_idx];
                    }
                }
            }

        }
        // Scalar fallback
        else
        {
            // Do matrix multiplication
            // K-Major loop (keeps all access row-major)
            for (var k = 0; k < a_cols; k++)
            {
                for (var i = 0; i < a_rows; i++)
                {
                    var a_val = a[i * a_cols + k];

                    for (int j = 0; j < b_cols; j++)
                    {
                        var r_idx = i * b_cols + j;
                        var b_val = b[k * b_cols + j];

                        result[r_idx] += a_val * b_val;
                    }
                }
            }
        }
    }

    private static Shape ComputeMatMulBroadcastShape(Shape a, Shape b)
    {
        // Assume a: [..., M, K], b: [..., K, N]
        // First, check if ranks are at least 2
        if (a.Rank < 2 || b.Rank < 2)
            throw new InvalidOperationException("Both tensors must be at least rank 2 for matmul.");

        int aRank = a.Rank;
        int bRank = b.Rank;

        // Extract batch dimensions (exclude last 2 dims)
        var aBatchDims = a.AsDimensionSpan().Slice(0, aRank - 2);
        var bBatchDims = b.AsDimensionSpan().Slice(0, bRank - 2);

        int maxBatchRank = Math.Max(aBatchDims.Length, bBatchDims.Length);
        int[] resultBatchDims = new int[maxBatchRank + 2];

        // Align batch dims from the right (like numpy/pytorch)
        for (int i = 0; i < maxBatchRank; i++)
        {
            int aIndex = aBatchDims.Length - 1 - i;
            int bIndex = bBatchDims.Length - 1 - i;

            int aDim = aIndex >= 0 ? aBatchDims[aIndex] : 1;
            int bDim = bIndex >= 0 ? bBatchDims[bIndex] : 1;

            if (aDim == bDim || aDim == 1 || bDim == 1)
                resultBatchDims[maxBatchRank - 1 - i] = Math.Max(aDim, bDim);
            else
                throw new InvalidOperationException($"Cannot broadcast batch dimensions at position {i}: {aDim} vs {bDim}");
        }

        // Determine output matrix dimensions
        int M = a.Length(aRank - 2);
        int K_a = a.Length(aRank - 1);
        int K_b = b.Length(bRank - 2);
        int N = b.Length(bRank - 1);

        if (K_a != K_b)
            throw new InvalidOperationException($"Matrix dimensions do not align for matmul: {K_a} vs {K_b}");

        // Construct final broadcasted shape: [...broadcasted_batches, M, N]
        resultBatchDims[^2] = M;
        resultBatchDims[^1] = N;

        return new Shape(resultBatchDims);
    }
    /// <summary>
    /// Perform batched matrix multiplication. Batch dimensions must be broadcastable
    /// </summary>
    /// <param name="other">tensor to multiply with</param>
    /// <returns>batched matrix product</returns>
    /// <exception cref="InvalidOperationException">thrown if matrix multiplication cannot be performed or if batch dimensions are not broadcastable</exception>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public Tensor<TNum> BatchedMatMul(Tensor<TNum> other)
    {
        // Shapes must have same number of dimensions (at least 2)
        if (this.Rank < 2 || other.Rank < 2)
            throw new InvalidOperationException("Shape rank mismatch for batched matrix multiplication, number of dimensions must be greater or equal to 2");

        var broadcast_shape = ComputeMatMulBroadcastShape(this.Shape, other.Shape);                   // Compute a general broadcast shape (all dimensions)
        var a = this.ReshapeShared(this.Shape.BroadcastTo(broadcast_shape, 0, broadcast_shape.Rank - 2));   // Broadcast to the computed shape, preserve the mat-mul dimensions (last 2)
        var b = other.ReshapeShared(other.Shape.BroadcastTo(broadcast_shape, 0, broadcast_shape.Rank - 2)); // Broadcast to the computed shape, preserve the mat-mul dimensions (last 2)

        var a_rank = a.Shape.Rank;
        var b_rank = b.Shape.Rank;

        // Last 2 dims must be matrix multiplication compatible
        var a_row_idx = a_rank - 2;
        var a_col_idx = a_rank - 1;
        var b_row_idx = b_rank - 2;
        var b_col_idx = b_rank - 1;
        int a_rows = a.Shape.Length(a_row_idx);
        int a_cols = a.Shape.Length(a_col_idx);
        int b_rows = b.Shape.Length(b_row_idx);
        int b_cols = b.Shape.Length(b_col_idx);

        if (a_cols != b_rows)
            throw new InvalidOperationException("Inner dimensions are not compatible for matrix multiplication");

        // Compute the dimensions for output
        int a_matsize = a_rows * a_cols;
        int b_matsize = a_cols * b_cols;
        int r_matsize = a_rows * b_cols;

        int[] r_shape = new int[broadcast_shape.Rank]; // The actual shape of the output (most copied from the broadcast shape, the last 2 from mat-mul)
        r_shape[r_shape.Length - 2] = a_rows;
        r_shape[r_shape.Length - 1] = b_cols;
        int r_count = r_matsize;
        for (var i = 0; i < r_shape.Length - 2; i++)
        {
            var dim_length = broadcast_shape.Length(i);
            r_shape[i] = dim_length;
            r_count *= dim_length;
        }
        if (r_count == 0)
            return new Tensor<TNum>(new Shape(r_shape), Array.Empty<TNum>());

        ReadOnlySpan<int> a_strides = a.Shape.AsStrideSpan();
        ReadOnlySpan<int> b_strides = b.Shape.AsStrideSpan();

        // Compute number of batches of matrices to multiply
        int batches = Math.Max(1, r_count / r_matsize);
        const int StackBatchThreshold = 32;
        Span<int> a_offsets = batches <= StackBatchThreshold ? stackalloc int[batches] : new int[batches];
        Span<int> b_offsets = batches <= StackBatchThreshold ? stackalloc int[batches] : new int[batches];
        for (int batch = 0; batch < batches; batch++)
        {
            int a_offset = 0, b_offset = 0;
            var tmp = batch;
            if (r_shape.Length > 2)
            {
                for (int d = r_shape.Length - 3; d >= 0; d--)
                {
                    int idx = tmp % r_shape[d];
                    tmp /= r_shape[d];

                    a_offset += a_strides[d] * idx;
                    b_offset += b_strides[d] * idx;
                }
            }
            a_offsets[batch] = a_offset;
            b_offsets[batch] = b_offset;
        }

        // Allocate
        TNum[] r_values = new TNum[r_count];
        if (TNum.Zero != default(TNum))
            Array.Fill(r_values, TNum.Zero);    // If default(TNum) is not zero, fill with 0's so we can be assured that all values are 0 before incrementing

        // For each 2D matrix in batches
        Span<TNum> r_span = r_values.AsSpan();
        Span<TNum> a_span = a.elements.AsSpan();
        Span<TNum> b_span = b.elements.AsSpan();

        // Vectorized multiply
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            int vecSize = Vector<TNum>.Count;
            for (var batch = 0; batch < batches; batch++)
            {
                // Extract the 2d matrix "region" of the tensor
                int a_offset = a_offsets[batch];
                int b_offset = b_offsets[batch];
                int r_offset = batch * r_matsize;

                // Do matrix multiplication
                // K-Major loop (keeps all access row-major)
                for (var k = 0; k < a_cols; k++)
                {
                    for (var i = 0; i < a_rows; i++)
                    {
                        var a_val = a_span[a_offset + i * a_cols + k];
                        var a_vec = new Vector<TNum>(a_val);

                        int j = 0;
                        for (; j <= b_cols - vecSize; j += vecSize)
                        {
                            int b_base = b_offset + k * b_cols + j;
                            int r_base = r_offset + i * b_cols + j;

                            var b_vec = new Vector<TNum>(b_span.Slice(b_base, vecSize)); // May be able to be replaced by (Unsafe.As<TNum, TNum[]>(ref Unsafe.Add(ref bRef, b_base)), 0)
                            var r_vec = new Vector<TNum>(r_span.Slice(r_base, vecSize));

                            var result = r_vec + a_vec * b_vec;

                            result.CopyTo(r_span.Slice(r_base, vecSize));
                        }

                        // Handle leftover elements (scalar fallback)
                        for (; j < b_cols; j++)
                        {
                            int b_idx = b_offset + k * b_cols + j;
                            int r_idx = r_offset + i * b_cols + j;

                            r_span[r_idx] += a_val * b_span[b_idx];
                        }
                    }
                }
            }
        }
        // Scalar fallback
        else
        {
            for (var batch = 0; batch < batches; batch++)
            {
                // Extract the 2d matrix "region" of the tensor
                int a_offset = a_offsets[batch];
                int b_offset = b_offsets[batch];
                int r_offset = batch * r_matsize;

                // Do matrix multiplication
                // K-Major loop (keeps all access row-major)
                for (var k = 0; k < a_cols; k++)
                {
                    for (var i = 0; i < a_rows; i++)
                    {
                        var a_val = a_span[a_offset + i * a_cols + k];

                        for (int j = 0; j < b_cols; j++)
                        {
                            var r_idx = r_offset + i * b_cols + j;
                            var b_val = b_span[b_offset + k * b_cols + j];

                            r_span[r_idx] += a_val * b_val;
                        }
                    }
                }
            }
        }

        // Done
        return new Tensor<TNum>(new Shape(r_shape), r_values);
    }

    /// <summary>
    /// Perform a 2D convolution of this tensor with the given kernel tensor.
    /// The input tensor is expected to have shape [...batches, channels, rows, columns].
    /// If rank is less than 4 the input tensor is broadcasted to rank 4
    /// </summary>
    /// <param name="kernels">
    /// Kernel tensor with shape [...outChannels, inChannelsPerGroup, kernelHeight, kernelWidth]
    /// If rank is less than 4 the kernel tensor is broadcasted to rank 4.
    /// </param>
    /// <param name="groups">
    /// Number of groups for grouped convolution (must divide input and output channels).
    /// If groups == 1 standard convolution is performed, if groups == inChannels == outChannels depthwise convolution is performed
    /// </param>
    /// <param name="strideX">Horizontal stride</param>
    /// <param name="strideY">Vertical stride</param>
    /// <param name="dilationX">Kernel horizontal dilation (spacing of the kernel)</param>
    /// <param name="dilationY">Kernel vertical dilation (spacing of the kernel)</param>
    /// <param name="padLeft">Padding on the left side of the input</param>
    /// <param name="padRight">Padding on the right side of the input</param>
    /// <param name="padTop">Padding on the top side of the input</param>
    /// <param name="padBottom">Padding on the bottom side of the input</param>
    /// <returns>Tensor resulting from the convolution with shape [batches, outChannels, outRows, outColumns]</returns>
    /// <exception cref="ArgumentException">Thrown if the input channels, output channels, or groups are incompatible or invalid</exception>
    public Tensor<TNum> Convolve2D(Tensor<TNum> kernels, int groups = 1, int strideX = 1, int strideY = 1, int dilationX = 1, int dilationY = 1, int padLeft = 0, int padRight = 0, int padTop = 0, int padBottom = 0, ReadOnlySpan<TNum> bias = default)
    {
        // Normalize all tensors to 4D (expand or reduce as required)
        var input = this.ReshapeShared(this.Shape.NormalizeRank(4));            // [batch, channels, rows, columns]
        kernels = kernels.ReshapeShared(kernels.Shape.NormalizeRank(4));        // [outChannels, inChannelsPerGroup, kernelHeight, kernelWidth]

        var batch = input.Shape.Length(0);
        var inChannels = input.Shape.Length(1);
        var inHeight = input.Shape.Length(2);
        var inWidth = input.Shape.Length(3);
        TNum[] inData = input.elements;

        var outChannels = kernels.Shape.Length(0);
        var inChannelsPerGroup = inChannels / groups;
        var outChannelsPerGroup = outChannels / groups;

        if (inChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by the number of groups.");
        if (outChannels % groups != 0)
            throw new ArgumentException("Output channels must be divisible by the number of groups.");
        if (kernels.Shape.Length(1) != inChannelsPerGroup)
            throw new ArgumentException("Kernel input channels do not match expected channels per group.");

        var kernelHeight = kernels.Shape.Length(2);
        var kernelWidth = kernels.Shape.Length(3);
        TNum[] kernelData = kernels.elements;

        var outHeight = (inHeight + padTop + padBottom - dilationY * (kernelHeight - 1) - 1) / strideY + 1;
        var outWidth = (inWidth + padLeft + padRight - dilationX * (kernelWidth - 1) - 1) / strideX + 1;
        if (outHeight <= 0 || outWidth <= 0)
            throw new ArgumentException("Invalid output dimensions. Check padding, stride, and dilation.");

        var outputShape = new Shape(batch, outChannels, outHeight, outWidth);
        var outputTensor = Tensor<TNum>.Defaults(outputShape);
        TNum[] outputData = outputTensor.elements;

        var inStrides_0 = input.Shape.Stride(0);
        var inStrides_1 = input.Shape.Stride(1);
        var inStrides_2 = input.Shape.Stride(2);
        var inStrides_3 = input.Shape.Stride(3);

        var kerStrides_0 = kernels.Shape.Stride(0);
        var kerStrides_1 = kernels.Shape.Stride(1);
        var kerStrides_2 = kernels.Shape.Stride(2);
        var kerStrides_3 = kernels.Shape.Stride(3);

        var outStrides_0 = outputShape.Stride(0);
        var outStrides_1 = outputShape.Stride(1);
        var outStrides_2 = outputShape.Stride(2);
        var outStrides_3 = outputShape.Stride(3);

        for (var b = 0; b < batch; b++)
        {
            int b_inStrides0 = b * inStrides_0;
            int b_outStrides0 = b * outStrides_0;

            for (int g = 0; g < groups; g++)
            {
                int inOffset = g * inChannelsPerGroup;
                int outOffset = g * outChannelsPerGroup;

                // --- Convolution Starts Here ---
                for (var oc = 0; oc < outChannelsPerGroup; oc++)
                //Parallel.For(0, outChannelsPerGroup, oc =>
                {
                    int fullOutChannel = outOffset + oc;
                    int fullOutChannel_outStrides1 = fullOutChannel * outStrides_1;
                    int outOffset_oc_kerStrides0 = (outOffset + oc) * kerStrides_0;
                    int result_offset_part0 = b_outStrides0 + fullOutChannel_outStrides1;

                    var initial = oc < bias.Length ? bias[oc] : TNum.Zero;

                    //for (int oy = 0; oy < outHeight; oy++)
                    Parallel.For(0, outHeight, ParallelOptions, oy =>
                    {
                        var inputSpan = inData.AsSpan();
                        var kernelSpan = kernelData.AsSpan();
                        var outputSpan = outputData.AsSpan();

                        ref TNum inputRef = ref MemoryMarshal.GetReference(inputSpan);
                        ref TNum kernelRef = ref MemoryMarshal.GetReference(kernelSpan);
                        ref TNum outputRef = ref MemoryMarshal.GetReference(outputSpan);

                        int inYBase = oy * strideY - padTop;
                        int oy_outStrides2 = oy * outStrides_2;
                        int result_offset_part1 = result_offset_part0 + oy_outStrides2;

                        for (int ox = 0; ox < outWidth; ox++)
                        {
                            int inXBase = ox * strideX - padLeft;
                            int ox_outStrides3 = ox * outStrides_3;
                            TNum sum = initial;

                            for (int ic = 0; ic < inChannelsPerGroup; ic++)
                            {
                                int fullInChannel = inOffset + ic;
                                int fullInChannel_inStrides1 = fullInChannel * inStrides_1;
                                int in_offset_part0 = b_inStrides0 + fullInChannel_inStrides1;
                                int ic_kerStrides1 = ic * kerStrides_1;
                                int out_offset_part0 = outOffset_oc_kerStrides0 + ic_kerStrides1;

                                for (int ky = 0; ky < kernelHeight; ky++)
                                {
                                    int inY = inYBase + ky * dilationY;
                                    if (inY < 0 || inY >= inHeight) continue;

                                    int inY_inStrides2 = inY * inStrides_2;
                                    int in_offset_part1 = in_offset_part0 + inY_inStrides2;
                                    int ky_kerStrides2 = ky * kerStrides_2;
                                    int out_offset_part1 = out_offset_part0 + ky_kerStrides2;

                                    for (int kx = 0; kx < kernelWidth; kx++)
                                    {
                                        int inX = inXBase + kx * dilationX;
                                        if (inX < 0 || inX >= inWidth) continue;

                                        int inIdx = in_offset_part1 + inX * inStrides_3;
                                        int kerIdx = out_offset_part1 + kx * kerStrides_3;

                                        sum += Unsafe.Add(ref inputRef, inIdx) * Unsafe.Add(ref kernelRef, kerIdx);
                                        // sum += inData[inIdx] * kernelData[kerIdx];
                                    }
                                }
                            }

                            int outIdx = result_offset_part1 + ox_outStrides3;
                            Unsafe.Add(ref outputRef, outIdx) = Unsafe.Add(ref outputRef, outIdx) + sum;
                            // outputData[outIdx] += sum;
                        }
                    }
                    ); // If Parallel.For is enabled
                }
                //); // If Parallel.For is enabled
                // --- Convolution Ends Here ---
            }
        }

        return outputTensor;
    }

    /// <summary>
    /// Perform a 2D transposed convolution of this tensor with the given kernel tensor.
    /// The input tensor is expected to have shape [...batches, channels, rows, columns].
    /// If rank is less than 4 the input tensor is broadcasted to rank 4
    /// </summary>
    /// <param name="kernels">
    /// Kernel tensor with shape [...inChannelsGrouped, outChannelsPerGroup, kernelHeight, kernelWidth]
    /// If rank is less than 4 the kernel tensor is broadcasted to rank 4.
    /// </param>
    /// <param name="groups">Number of groups for grouped transposed convolution</param>
    /// <param name="strideX">Horizontal stride</param>
    /// <param name="strideY">Vertical stride</param>
    /// <param name="dilationX">Kernel horizontal dilation (spacing of the kernel)</param>
    /// <param name="dilationY">Kernel vertical dilation (spacing of the kernel)</param>
    /// <param name="inPadLeft">Padding on the left side of the input (cropping)</param>
    /// <param name="inPadRight">Padding on the right side of the input (cropping)</param>
    /// <param name="inPadTop">Padding on the top side of the input (cropping)</param>
    /// <param name="inPadBottom">Padding on the bottom side of the input (cropping)</param>
    /// <param name="outPadLeft">Padding on the left side of the output (expansion)</param>
    /// <param name="outPadRight">Padding on the right side of the output (expansion)</param>
    /// <param name="outPadTop">Padding on the top side of the output (expansion)</param>
    /// <param name="outPadBottom">Padding on the bottom side of the output (expansion)</param>
    /// <returns>Tensor resulting from the transposed convolution with shape [batches, outChannels, outRows, outColumns]</returns>
    /// <exception cref="ArgumentException">Thrown if the input channels, output channels, or groups are incompatible or invalid</exception>
    public Tensor<TNum> TransposeConvolve2D(Tensor<TNum> kernels, int groups = 1, int strideX = 1, int strideY = 1, int dilationX = 1, int dilationY = 1, int inPadLeft = 0, int inPadRight = 0, int inPadTop = 0, int inPadBottom = 0, int outPadLeft = 0, int outPadRight = 0, int outPadTop = 0, int outPadBottom = 0, ReadOnlySpan<TNum> bias = default)
    {
        // Normalize all tensors to 4D (expand or reduce as required)
        var input = this.ReshapeShared(this.Shape.NormalizeRank(4));            // [batch, channels, rows, columns]
        kernels = kernels.ReshapeShared(kernels.Shape.NormalizeRank(4));        // [inChannelsGrouped, outChannelsPerGroup, kernelHeight, kernelWidth]

        var batch = input.Shape.Length(0);
        var inChannels = input.Shape.Length(1);
        var inHeight = input.Shape.Length(2);
        var inWidth = input.Shape.Length(3);
        var inData = input.elements;

        var inChannelsPerGroup = inChannels / groups;
        var outChannelsPerGroup = kernels.Shape.Length(1);
        var outChannels = outChannelsPerGroup * groups;

        if (inChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by the number of groups.");
        if (kernels.Shape.Length(0) != inChannelsPerGroup)
            throw new ArgumentException("Kernel input channels do not match expected channels per group.");
        if (kernels.Shape.Length(1) * groups != outChannels)
            throw new ArgumentException("Kernel output channels do not match expected channels per group.");

        var kernelHeight = kernels.Shape.Length(2);
        var kernelWidth = kernels.Shape.Length(3);
        var kernelData = kernels.elements;

        // Compute output size (based on standard transposed conv formula)
        var outHeight = (inHeight - 1) * strideY - inPadTop - inPadBottom + dilationY * (kernelHeight - 1) + 1 + outPadTop + outPadBottom;
        var outWidth = (inWidth - 1) * strideX - inPadLeft - inPadRight + dilationX * (kernelWidth - 1) + 1 + outPadLeft + outPadRight;

        var outputShape = new Shape(batch, outChannels, outHeight, outWidth);
        var outputBatchStride = outputShape.Stride(0);
        var outputChannelStride = outputShape.Stride(1);
        var outputTensor = Tensor<TNum>.Defaults(outputShape);
        for (var b = 0; b < batch; b++) {
            var batchOffset = b * outputBatchStride;
            for (var oc = 0; oc < outChannels; oc++)
            {
                var biasv = oc < bias.Length ? bias[oc] : TNum.Zero;
                var channelOffset = oc * outputChannelStride;
                outputTensor.AsSpan(batchOffset + channelOffset).Fill(biasv); // Fill initial bias in all output channels
            }
        }
        var outputData = outputTensor.elements;

        var inStrides_0 = input.Shape.Stride(0);
        var inStrides_1 = input.Shape.Stride(1);
        var inStrides_2 = input.Shape.Stride(2);
        var inStrides_3 = input.Shape.Stride(3);

        var kerStrides_0 = kernels.Shape.Stride(0);
        var kerStrides_1 = kernels.Shape.Stride(1);
        var kerStrides_2 = kernels.Shape.Stride(2);
        var kerStrides_3 = kernels.Shape.Stride(3);

        var outStrides_0 = outputShape.Stride(0);
        var outStrides_1 = outputShape.Stride(1);
        var outStrides_2 = outputShape.Stride(2);
        var outStrides_3 = outputShape.Stride(3);

        TNum zero = TNum.Zero;

        // Pre-computations (avoid computing inside the loops)
        int[] kx_dilations = new int[kernelWidth];
        int[] ky_dilations = new int[kernelHeight];
        for (var kx = 0; kx < kernelWidth; kx++) kx_dilations[kx] = kx * dilationX;
        for (var ky = 0; ky < kernelHeight; ky++) ky_dilations[ky] = ky * dilationY;

        // Here be giants vvvv
        Parallel.For(0, batch, ParallelOptions, (b) =>
        //for (int b = 0; b < batch; b++)
        {
            // Setup spans and references
            Span<TNum> inputSpan = inData;
            Span<TNum> kernelSpan = kernelData;
            Span<TNum> outputSpan = outputData;

            ref TNum inputRef = ref MemoryMarshal.GetReference(inputSpan);
            ref TNum kernelRef = ref MemoryMarshal.GetReference(kernelSpan);
            ref TNum outputRef = ref MemoryMarshal.GetReference(outputSpan);

            ref int kx_dilationsRef = ref MemoryMarshal.GetArrayDataReference(kx_dilations);
            ref int ky_dilationsRef = ref MemoryMarshal.GetArrayDataReference(ky_dilations);

            var b_inStrides_0 = b * inStrides_0;
            var b_outStrides_0 = b * outStrides_0;

            for (int g = 0; g < groups; g++)
            {
                int inOffset = g * inChannelsPerGroup;
                int outOffset = g * outChannelsPerGroup;

                for (int ic = 0; ic < inChannelsPerGroup; ic++)
                {
                    int fullInChannel = inOffset + ic;
                    var in_offset_part0 = b_inStrides_0 + fullInChannel * inStrides_1;
                    var ic_kerStrides_0 = ic * kerStrides_0;

                    for (int iy = 0; iy < inHeight; iy++)
                    {
                        int outYBase = iy * strideY - inPadTop + outPadTop;
                        int iy_inStrides_2 = iy * inStrides_2;
                        var in_offset_part1 = in_offset_part0 + iy_inStrides_2;

                        for (int ix = 0; ix < inWidth; ix++)
                        {
                            int in_idx = in_offset_part1 + ix * inStrides_3;
                            TNum val = Unsafe.Add(ref inputRef, in_idx); // inData[in_idx];
                            if (val == zero) continue; // Skip 0's (can result in real wins when ReLU is used)

                            int outXBase = ix * strideX - inPadLeft + outPadLeft;

                            for (int oc = 0; oc < outChannelsPerGroup; oc++)
                            {
                                int fullOutChannel = outOffset + oc;
                                var out_offset_part0 = b_outStrides_0 + fullOutChannel * outStrides_1;
                                var oc_kerStrides_1 = oc * kerStrides_1;
                                var ker_offset_part0 = ic_kerStrides_0 + oc_kerStrides_1;

                                ref TNum kerBase = ref Unsafe.Add(ref kernelRef, ker_offset_part0);
                                ref TNum outBase = ref Unsafe.Add(ref outputRef, out_offset_part0);

                                for (int ky = 0; ky < kernelHeight; ky++)
                                {
                                    int outY = outYBase + Unsafe.Add(ref ky_dilationsRef, ky);
                                    if (outY < 0 || outY >= outHeight) continue;

                                    var kerIndexBase = ky * kerStrides_2;
                                    var outIdxBase = outY * outStrides_2;

                                    for (int kx = 0; kx < kernelWidth; kx++)
                                    {
                                        int outX = outXBase + Unsafe.Add(ref kx_dilationsRef, kx);
                                        if (outX < 0 || outX >= outWidth) continue;

                                        int kerIdx = kerIndexBase + kx * kerStrides_3;
                                        int outIdx = outIdxBase + outX * outStrides_3;

                                        ref TNum dst = ref Unsafe.Add(ref outBase, outIdx);
                                        TNum src = Unsafe.Add(ref kerBase, kerIdx);
                                        dst = dst + val * src;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        //}
        });

        return outputTensor;
    }
/*  public Tensor<TNum> TransposeConvolve2D(Tensor<TNum> kernels, int groups = 1, int strideX = 1, int strideY = 1, int dilationX = 1, int dilationY = 1, int inPadLeft = 0, int inPadRight = 0, int inPadTop = 0, int inPadBottom = 0, int outPadLeft = 0, int outPadRight = 0, int outPadTop = 0, int outPadBottom = 0,  ReadOnlySpan<TNum> bias = default)
    {
        // Normalize all tensors to 4D (expand or reduce as required)
        var input = this.ReshapeShared(this.Shape.NormalizeRank(4));            // [batch, channels, rows, columns]
        kernels = kernels.ReshapeShared(kernels.Shape.NormalizeRank(4));        // [inChannelsGrouped, outChannelsPerGroup, kernelHeight, kernelWidth]

        var batch = input.Shape.Length(0);
        var inChannels = input.Shape.Length(1);
        var inHeight = input.Shape.Length(2);
        var inWidth = input.Shape.Length(3);
        var inData = input.elements;

        var inChannelsPerGroup = inChannels / groups;
        var outChannelsPerGroup = kernels.Shape.Length(1);
        var outChannels = outChannelsPerGroup * groups;

        if (inChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by the number of groups.");
        if (kernels.Shape.Length(0) != inChannelsPerGroup)
            throw new ArgumentException("Kernel input channels do not match expected channels per group.");
        if (kernels.Shape.Length(1) * groups != outChannels)
            throw new ArgumentException("Kernel output channels do not match expected channels per group.");

        var kernelHeight = kernels.Shape.Length(2);
        var kernelWidth = kernels.Shape.Length(3);
        var kernelData = kernels.elements;

        // Compute output size (based on standard transposed conv formula)
        var outHeight = (inHeight - 1) * strideY - inPadTop - inPadBottom + dilationY * (kernelHeight - 1) + 1 + outPadTop + outPadBottom;
        var outWidth = (inWidth - 1) * strideX - inPadLeft - inPadRight + dilationX * (kernelWidth - 1) + 1 + outPadLeft + outPadRight;

        var outputShape = new TensorShape(batch, outChannels, outHeight, outWidth);
        var outputTensor = Tensor<TNum>.Defaults(outputShape);
        var outputData = outputTensor.elements;

        var inStrides_0 = input.Shape.Stride(0);
        var inStrides_1 = input.Shape.Stride(1);
        var inStrides_2 = input.Shape.Stride(2);
        var inStrides_3 = input.Shape.Stride(3);

        var kerStrides_0 = kernels.Shape.Stride(0);
        var kerStrides_1 = kernels.Shape.Stride(1);
        var kerStrides_2 = kernels.Shape.Stride(2);
        var kerStrides_3 = kernels.Shape.Stride(3);

        var outStrides_0 = outputShape.Stride(0);
        var outStrides_1 = outputShape.Stride(1);
        var outStrides_2 = outputShape.Stride(2);
        var outStrides_3 = outputShape.Stride(3);

        TNum zero = TNum.Zero;

        // Here be giants vvvv
        for (int b = 0; b < batch; b++)
        {
            var b_inStrides_0 = b * inStrides_0;
            var b_outStrides_0 = b * outStrides_0;

            for (int g = 0; g < groups; g++)
            {
                int inOffset = g * inChannelsPerGroup;
                int outOffset = g * outChannelsPerGroup;

                //Parallel.For(0, outChannelsPerGroup, oc => {      
                for (int oc = 0; oc < outChannelsPerGroup; oc++)
                {
                    int fullOutChannel = outOffset + oc;
                    var out_offset_part0 = b_outStrides_0 + fullOutChannel * outStrides_1;
                    var oc_kerStrides_1 = oc * kerStrides_1;

                    TNum initial = oc < bias.Length ? bias[oc] : TNum.Zero;

                    Parallel.For(0, outHeight, oy =>
                    {
                        // Need to have these here because ref types cannot be captured by anonymous functions
                        ref TNum inputRef = ref MemoryMarshal.GetArrayDataReference(inData);
                        ref TNum kernelRef = ref MemoryMarshal.GetArrayDataReference(kernelData);
                        ref TNum outputRef = ref MemoryMarshal.GetArrayDataReference(outputData);
                        //for (int oy = 0; oy < outHeight; oy++)
                        {
                            int outYBase = oy - outPadTop + inPadTop;
                            int oy_outStrides_2 = oy * outStrides_2;
                            var out_offset_part1 = out_offset_part0 + oy_outStrides_2;

                            for (int ox = 0; ox < outWidth; ox++)
                            {
                                int outXBase = ox - outPadLeft + inPadLeft;
                                int ox_outStrides_3 = ox * outStrides_3;
                                var out_idx = out_offset_part1 + ox_outStrides_3;

                                TNum sum = initial;

                                for (int ic = 0; ic < inChannelsPerGroup; ic++)
                                {
                                    int fullInChannel = inOffset + ic;
                                    var ic_kerStrides_0 = ic * kerStrides_0;
                                    var ker_offset_part0 = ic_kerStrides_0 + oc_kerStrides_1;

                                    for (int ky = 0; ky < kernelHeight; ky++)
                                    {
                                        if (!TryComputeInputCoord(oy, strideY, inPadTop, ky * dilationY, out int iy) || iy >= inHeight)
                                            continue;

                                        var kerIndexBase = ky * kerStrides_2;
                                        var in_offset_part1 = b_inStrides_0 + fullInChannel * inStrides_1 + iy * inStrides_2;

                                        for (int kx = 0; kx < kernelWidth; kx++)
                                        {
                                            if (!TryComputeInputCoord(ox, strideX, inPadLeft, kx * dilationX, out int ix) || ix >= inWidth)
                                                continue;

                                            int in_idx = in_offset_part1 + ix * inStrides_3;
                                            TNum val = Unsafe.Add(ref inputRef, in_idx); //TNum val = inData[in_idx]; 

                                            int kerIdx = kerIndexBase + kx * kerStrides_3;
                                            TNum src = Unsafe.Add(ref kernelRef, kerIdx); //TNum src = kernelData[kerIdx];
                                            sum += val == zero ? TNum.Zero : val * src;
                                        }
                                    }
                                }

                                Unsafe.Add(ref outputRef, out_idx) = sum; //outputData[out_idx] = sum;
                            }
                        }
                    });
                } //});
            }
        }

        return outputTensor;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryComputeInputCoord(int outputCoord, int stride, int pad, int kernelDilatedCoord, out int inputCoord)
    {
        int numerator = outputCoord + pad - kernelDilatedCoord;
        if (numerator < 0)
        {
            inputCoord = -1;
            return false;
        }

        int remainder;
        int quotient = Math.DivRem(numerator, stride, out remainder);
        if (remainder != 0)
        {
            inputCoord = -1;
            return false;
        }
        inputCoord = quotient;
        return true;
    }

    // Left for clarity for the methods ^^
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int transConv_compute_input_y(int outputY, int strideY, int padTop, int padBottom, int dilationY, int kernelY)
    {
        return (outputY + padTop - dilationY * kernelY) / strideY;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int transConv_compute_input_x(int outputX, int strideX, int padLeft, int padRight, int dilationX, int kernelX)
    {
        return (outputX + padLeft - dilationX * kernelX) / strideX;
    }*/

    /// <summary>
    /// Fill the tensor with all values being the same
    /// </summary>
    /// <param name="value">Value to fill across the tensor</param>
    public void FillConstant(TNum value)
    {
        var values = this.elements.AsSpan();
        values.Fill(value);
    }

    /// <summary>
    /// Fill the tensor with all zeros
    /// </summary>
    public void FillZeros()
    {
        var values = this.elements.AsSpan();
        values.Fill(TNum.Zero);
    }

    /// <summary>
    /// Fill the tensor with all zeros
    /// </summary>
    public void FillOnes()
    {
        var values = this.elements.AsSpan();
        values.Fill(TNum.One);
    }

    /// <summary>
    /// Fill a tensor with values being generated by a function
    /// </summary>
    /// <param name="generator">generator function</param>
    public void FillGenerated(Func<TNum> generator)
    {
        for (var i = 0; i < this.elements.Length; i++)
        {
            this.elements[i] = generator();
        }
    }

    /// <summary>
    /// Normalize a potentially negative axis index to only positive indices in tensor rank (inlined to reduce call overhead)
    /// </summary>
    /// <param name="axis">axis to normalize</param>
    /// <returns>positive axis index</returns>
    /// <exception cref="ArgumentOutOfRangeException">thrown if axis index is out of bounds</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int NormalizeAxis(int axis)
    {
        var rank = Shape.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis is out of range");
        return axis;
    }

    /// <summary>
    /// Normalize a potentially negative axis index to only positive indices in tensor rank (inlined to reduce call overhead)
    /// </summary>
    /// <param name="axis">axis to normalize</param>
    /// <returns>positive axis index</returns>
    /// <exception cref="ArgumentOutOfRangeException">thrown if axis index is out of bounds</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int NormalizeAxis(Index axis)
    {
        var rank = Shape.Rank;
        var axisi = axis.GetOffset(rank);
        if (axisi < 0 || axisi >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis is out of range");
        return axisi;
    }

    /// <summary>
    /// Pads the tensor along the last<paramref name="paddings"/> dimensions, with each tuple specifying the number of elements to pad before and after the data along that axis.
    /// Negative padding results in truncation of values long that axis by skipping elements that would be outside of the standard range.
    /// </summary>
    /// <param name="padValue">constant value to pad with</param>
    /// <param name="paddings">padding values for last dimensions</param>
    /// <returns>padded tensor</returns>
    public Tensor<TNum> Pad(TNum padValue, params ReadOnlySpan<(int Before, int After)> paddings)
    {
        if (paddings.Length == 0)
            return this;

        var shape = this.Shape;
        var rank = shape.Rank;
        if (rank == 0)
            throw new InvalidOperationException("Cannot pad a scalar tensor.");

        int[] paddedShape = new int[rank];
        for (var i = 0; i < rank; i++)
        {
            // Paddings array aligns to the right of the shape so that missing value represent the highest dimensions rather than the lowest
            var padding_index = (i - rank) + paddings.Length;
            if (padding_index >= 0 && padding_index < paddings.Length)
            {
                var padding = paddings[padding_index];
                paddedShape[i] = padding.Before + shape.Length(i) + padding.After;
            }
            else
            {
                paddedShape[i] = shape.Length(i);
            }
            if (paddedShape[i] < 0)
                throw new ArgumentException($"Padding for axis {i} results in invalid dimension length, final dimension length must be a positive integer");
        }

        var result = Tensor<TNum>.ConstantValued(new Shape(paddedShape), padValue);

        Span<int> destIndices = stackalloc int[rank];

        var ienumerator = shape.CreateIndexEnumerator();
        Span<TNum> elements = this.elements;
        Span<int> indices = stackalloc int[shape.Rank];
        int flatIndex = 0;

        ienumerator.Initialize(indices, ref flatIndex);
        while (ienumerator.MoveNext(indices, ref flatIndex))
        {
            for (int i = 0; i < rank; i++)
            {
                var padding_index = (i - rank) + paddings.Length;
                if (padding_index >= 0 && padding_index < paddings.Length)
                {
                    var destIndexValue = indices[i] + paddings[padding_index].Before;
                    if (destIndexValue < 0 || destIndexValue >= paddedShape[i])
                        goto loop_continue; // A negative index means we don't copy this over (negative padding = truncation?)
                    destIndices[i] = destIndexValue;
                }
                else
                {
                    destIndices[i] = indices[i];
                }
            }

            result[destIndices] = elements[flatIndex];
        loop_continue:
            continue;
        }

        return result;
    }

    /// <summary>
    /// Pad the tensor along the last 2 dimensions (rows/columns) with the given padding amounts
    /// </summary>
    /// <param name="padValue">constant value to pad with</param>
    /// <param name="left">padding on the left hand side of each row</param>
    /// <param name="top">padding on the top of each column</param>
    /// <param name="right">padding on the right hand side of each row</param>
    /// <param name="bottom">padding on the bottom of each column</param>
    /// <returns>padded tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TNum> Pad(TNum padValue, int left = 0, int top = 0, int right = 0, int bottom = 0) => Pad(padValue, (left, right), (top, bottom));

    /// <summary>
    /// Extract/Slice the tensor to get a subset of the tensor elements
    /// </summary>
    /// <param name="slices">left-aligned slices per dimension, if less than rank remaining dimensions are sliced entirely</param>
    /// <returns>sliced tensor</returns>
    /// <exception cref="ArgumentException">thrown when slices are invalid</exception>
    public Tensor<TNum> Slice(params ReadOnlySpan<Range> slices)
    {
        int rank = Shape.Rank;
        if (slices.Length > rank)
            throw new ArgumentException("Too many slices for tensor rank");

        // Normalize ranges
        // Make sure all starts are <= ends using GetOffsetAndLength
        Range[] normalizedRanges = new Range[rank];
        int sliceElementCount = 1;
        for (var i = 0; i < rank; i++)
        {
            var dimLength = Shape.Length(i);
            if (i < slices.Length)
            {
                var slice = slices[i];
                var (start, length) = slice.GetOffsetAndLength(dimLength);
                var end = start + length;
                sliceElementCount *= length;
                normalizedRanges[i] = new Range(start, end);
            }
            else
            {
                sliceElementCount *= dimLength;
                normalizedRanges[i] = new Range(0, dimLength);
            }
        }

        TNum[] resultData = new TNum[sliceElementCount];
        Shape resultShape = new Shape(normalizedRanges.Select((x, i) => x.End.Value - x.Start.Value).ToArray());
        Tensor<TNum> result = new Tensor<TNum>(resultShape, resultData);

        Span<int> sourceIndex = stackalloc int[rank];

        var ienumerator = resultShape.CreateIndexEnumerator();
        Span<int> resultIndex = stackalloc int[resultShape.Rank];

        ienumerator.Initialize(resultIndex);
        while (ienumerator.MoveNext(resultIndex))
        {
            // Compute source index
            for (int d = 0; d < rank; d++)
                sourceIndex[d] = normalizedRanges[d].Start.Value + resultIndex[d];

            result[resultIndex] = this[sourceIndex];
        }

        return result;
    }

    /// <summary>
    /// Extract/Slice the tensor to get a subset of the tensor elements
    /// </summary>
    /// <param name="axis">axis to slice</param>
    /// <param name="range">range to sliec over</param>
    /// <returns>sliced tensor</returns>
    /// <exception cref="ArgumentException">thrown when slices are invalid</exception>
    public Tensor<TNum> SliceAlong(Index axis, Range range)
    {
        var s = this.Shape;
        var r = s.Rank;
        var abs = NormalizeAxis(axis);
        Span<Range> ranges = stackalloc Range[r];
        for (var i = 0; i < r; i++)
        {
            ranges[i] = i == abs ? range : new Range(0, s.Length(i));
        }
        return Slice(ranges);
    }

    /// <summary>
    /// Extract a row of the tensor at the given row index
    /// </summary>
    /// <param name="indices">index to the precise row</param>
    /// <returns>span over the row values</returns>
    /// <exception cref="ArgumentException">thrown when an index is invalid or the wrong number of indices are provided</exception>
    public Span<TNum> ViewRow(params ReadOnlySpan<Index> indices)
    {
        if (indices.Length != Rank - 1)
            throw new ArgumentException("To extract a row span you must provide indices for all dimensions except the last one");
            
        // Compute the flat offset to the start of the span
        int offset = 0;
        var strides = Shape.AsStrideSpan();
        for (int i = 0; i < indices.Length; i++)
        {
            offset += NormalizeAxis(indices[i]) * strides[i];
        }
    
        // Deterime the row length
        var rowLength = Shape.Length(^1);
    
        // Slice
        return this.elements.AsSpan(offset, rowLength);
    }

    /// <summary>
    /// Extract a submatrix of the tensor at the given matrix index
    /// </summary>
    /// <param name="indices">index to the submatrix</param>
    /// <returns>span over the rows and columns of the submatrix</returns>
    /// <exception cref="ArgumentException">thrown when an index is invalid or the wrong number of indices are provided</exception>
    public Span2D<TNum> ViewSubmatrix(params ReadOnlySpan<Index> indices)
    {
        if (indices.Length != Rank - 2)
            throw new ArgumentException("To extract a submatrix span you must provide indices for all dimensions except the last two");

        // Compute the flat offset to the start of the span
        int offset = 0;
        var strides = Shape.AsStrideSpan();
        for (int i = 0; i < indices.Length; i++)
        {
            offset += NormalizeAxis(indices[i]) * strides[i];
        }

        // Deterime the row length
        var size = Shape.Stride(^2);
        var rows = Shape.Length(^2);
        var cols = Shape.Length(^1);

        // Slice and wrap as a 2D span
        return new Span2D<TNum>(this.elements.AsSpan(offset, size), rows, cols);
    }
    
    /// <summary>
    /// Extract a submatrix of the tensor at the given matrix index
    /// </summary>
    /// <param name="indices">index to the submatrix</param>
    /// <returns>span over the rows and columns of the submatrix</returns>
    /// <exception cref="ArgumentException">thrown when an index is invalid or the wrong number of indices are provided</exception>
    public Span2D<TNum> ViewSubmatrix(params ReadOnlySpan<int> indices)
    {
        if (indices.Length != Rank - 2)
            throw new ArgumentException("To extract a submatrix span you must provide indices for all dimensions except the last two");
    
        // Compute the flat offset to the start of the span
        int offset = 0;
        var strides = Shape.AsStrideSpan();
        for (int i = 0; i < indices.Length; i++)
        {
            offset += NormalizeAxis(indices[i]) * strides[i];
        }
     
        // Deterime the row length
        var size = Shape.Stride(^2);
        var rows = Shape.Length(^2);
        var cols = Shape.Length(^1);
    
        // Slice and wrap as a 2D span
        return new Span2D<TNum>(this.elements.AsSpan(offset, size), rows, cols);
    }

    /// <summary>
    /// Construct a view over a given slice of the tensor this view can be used for basic operations only. The underlying array is shared between the view and the tensor. 
    /// </summary>
    /// <param name="slices">left-aligned slices per dimension, if less than rank remaining dimensions are sliced entirely</param>
    /// <returns>view of tensor over slice</returns>
    /// <exception cref="ArgumentException">thrown when slices are invalid</exception>
    public TensorView<TNum> View(params ReadOnlySpan<Range> slices)
    {
        int rank = Shape.Rank;
        if (slices.Length > rank)
            throw new ArgumentException("Too many slices for tensor rank");

        // Normalize ranges
        // Make sure all starts are <= ends using GetOffsetAndLength
        Range[] normalizedRanges = new Range[rank];
        int sliceElementCount = 1;
        int offset = 0;
        for (var i = 0; i < rank; i++)
        {
            var dimLength = Shape.Length(i);
            if (i < slices.Length)
            {
                var slice = slices[i];
                var (start, length) = slice.GetOffsetAndLength(dimLength);
                var end = start + length;
                sliceElementCount *= length;
                normalizedRanges[i] = new Range(start, end);
                offset += start * Shape.Stride(i);
            }
            else
            {
                sliceElementCount *= dimLength;
                normalizedRanges[i] = new Range(0, dimLength);
            }
        }

        Shape resultShape = new Shape(normalizedRanges.Select((x, i) => x.End.Value - x.Start.Value).ToArray());
        return new TensorView<TNum>(resultShape, this.elements, offset);
    }

    // TODO Extract (rows/columns) or Slice 
    // TODO Iterate over dimensions (row / column etc)

    /// <summary>
    /// Reduce a given axis in the tensor by applying a reduction function along that axis
    /// </summary>
    /// <typeparam name="TAccumulate">Reduced or accumulated element type</typeparam>
    /// <param name="axis">axis to reduce</param>
    /// <param name="seed">initial reducer value</param>
    /// <param name="reducer">reducer or accumulator function to be invoked on each element</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>Tensor with reduced shape</returns>
    /// <exception cref="ArgumentNullException">thrown if the reducer is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">thrown if the axis is invalid</exception>
    public Tensor<TAccumulate> Reduce<TAccumulate>(Index axis, TAccumulate seed, Func<TAccumulate, TNum, TAccumulate> reducer, bool keepdim = true)
    where TAccumulate : INumber<TAccumulate>
    {
        if (reducer is null)
            throw new ArgumentNullException(nameof(reducer));
        var shape = this.Shape;
        var rank = shape.Rank;
        if (rank == 0)
            throw new InvalidOperationException("Cannot reduce a scalar tensor.");

        // Handle negative axis
        var positive_axis = NormalizeAxis(axis);

        // Create output shape
        int[] reducedShape;
        if (keepdim)
        {
            reducedShape = new int[rank];
            for (int i = 0; i < rank; i++)
                reducedShape[i] = (i == positive_axis) ? 1 : shape.Length(i);
        }
        else
        {
            reducedShape = new int[rank - 1];
            for (int i = 0, j = 0; i < rank; i++)
            {
                if (i == positive_axis) continue;
                reducedShape[j++] = shape.Length(i);
            }
        }

        var resultShape = new Shape(reducedShape);
        var result = Tensor<TAccumulate>.ConstantValued(resultShape, seed);
        Span<TAccumulate> resultElements = result.elements;

        Span<int> outputIndices = stackalloc int[keepdim ? rank : rank - 1];

        var ienumerator = Shape.CreateIndexEnumerator();
        Span<TNum> elements = this.elements;
        Span<int> indices = stackalloc int[Shape.Rank];
        int flatIndex = 0;

        ienumerator.Initialize(indices, ref flatIndex);
        while (ienumerator.MoveNext(indices, ref flatIndex))
        {
            if (keepdim)
            {
                for (int i = 0; i < rank; i++)
                    outputIndices[i] = (i == positive_axis) ? 0 : indices[i];
            }
            else
            {
                for (int i = 0, j = 0; i < rank; i++)
                {
                    if (i == positive_axis) continue;
                    outputIndices[j++] = indices[i];
                }
            }

            var out_flat = resultShape.FlattenIndices(outputIndices);
            var old = resultElements[out_flat];
            resultElements[out_flat] = reducer(old, elements[flatIndex]);
        }

        return result;
    }
    /// <summary>
    /// Reduce over a given set of axes in the tensor by applying a reduction function along those axes
    /// </summary>
    /// <typeparam name="TAccumulate">Reduced or accumulated element type</typeparam>
    /// <param name="axes">axes to reduce</param>
    /// <param name="seed">initial reducer value</param>
    /// <param name="reducer">reducer or accumulator function to be invoked on each element</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>Tensor with reduced shape</returns>
    /// <exception cref="ArgumentNullException">thrown if the reducer is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">thrown if any axis is invalid</exception>
    public Tensor<TAccumulate> Reduce<TAccumulate>(
        ReadOnlySpan<Index> axes,
        TAccumulate seed,
        Func<TAccumulate, TNum, TAccumulate> reducer,
        bool keepdim = true)
        where TAccumulate : INumber<TAccumulate>
    {
        if (reducer is null)
            throw new ArgumentNullException(nameof(reducer));
        if (axes.Length == 0)
            throw new ArgumentException("Axes must not be empty.", nameof(axes));

        var shape = this.Shape;
        var rank = shape.Rank;

        // Normalize axes (handle negatives)
        Span<int> positiveAxes = stackalloc int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
            positiveAxes[i] = NormalizeAxis(axes[i]);

        // Create output shape
        int[] reducedShape = keepdim ? new int[rank] : new int[rank - positiveAxes.Length];
        {
            int j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (positiveAxes.Contains(i))
                {
                    if (keepdim)
                        reducedShape[i] = 1;
                    // else skip
                }
                else
                {
                    reducedShape[keepdim ? i : j++] = shape.Length(i);
                }
            }
        }

        var resultShape = new Shape(reducedShape);
        var result = Tensor<TAccumulate>.ConstantValued(resultShape, seed);
        Span<TAccumulate> resultElements = result.elements;

        Span<int> outputIndices = stackalloc int[keepdim ? rank : rank - positiveAxes.Length];

        var ienumerator = shape.CreateIndexEnumerator();
        Span<TNum> inputElements = this.elements;
        Span<int> indices = stackalloc int[rank];
        int flatIndex = 0;

        ienumerator.Initialize(indices, ref flatIndex);
        while (ienumerator.MoveNext(indices, ref flatIndex))
        {
            // Build output index based on keepdim and axes
            if (keepdim)
            {
                for (int i = 0; i < rank; i++)
                    outputIndices[i] = positiveAxes.Contains(i) ? 0 : indices[i];
            }
            else
            {
                int j = 0;
                for (int i = 0; i < rank; i++)
                {
                    if (!positiveAxes.Contains(i))
                        outputIndices[j++] = indices[i];
                }
            }

            var outFlat = resultShape.FlattenIndices(outputIndices);
            var oldVal = resultElements[outFlat];
            resultElements[outFlat] = reducer(oldVal, inputElements[flatIndex]);
        }

        return result;
    }

    /// <summary>
    /// Perform reduction on the row dimension (2nd last)
    /// </summary>
    /// <typeparam name="TAccumulate">Reduced or accumulated element type</typeparam>
    /// <param name="seed">initial reducer value</param>
    /// <param name="reducer">reducer or accumulator function to be invoked on each element</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>reduced tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TAccumulate> ReduceRows<TAccumulate>(TAccumulate seed, Func<TAccumulate, TNum, TAccumulate> reducer, bool keepdim = true)
    where TAccumulate : INumber<TAccumulate>
    => Reduce(axis: ^2, seed, reducer, keepdim);

    /// <summary>
    /// Perform reduction on the columns dimension (last)
    /// </summary>
    /// <typeparam name="TAccumulate">Reduced or accumulated element type</typeparam>
    /// <param name="seed">initial reducer value</param>
    /// <param name="reducer">reducer or accumulator function to be invoked on each element</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>reduced tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TAccumulate> ReduceColumns<TAccumulate>(TAccumulate seed, Func<TAccumulate, TNum, TAccumulate> reducer, bool keepdim = true)
    where TAccumulate : INumber<TAccumulate>
    => Reduce(axis: ^1, seed, reducer, keepdim);

    /// <summary>
    /// Optimized sum reduction along the given axis
    /// </summary>
    /// <param name="axis">axis to sum over</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>reduced tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public Tensor<TNum> Sum(Index axis, bool keepdim = true)
    {
        int dim = NormalizeAxis(axis);
        int rank = Shape.Rank;

        if (rank == 0)
            throw new InvalidOperationException("Cannot reduce scalar tensor");

        int axisSize = Shape.Length(dim);
        int axisStride = Shape.Stride(dim);

        // Compute outer and inner sizes
        int outerSize = 1;
        for (int i = 0; i < dim; i++)
            outerSize *= Shape.Length(i);

        int innerSize = 1;
        for (int i = dim + 1; i < rank; i++)
            innerSize *= Shape.Length(i);

        // Output shape
        int[] newShape = new int[keepdim ? rank : rank - 1];
        for (int i = 0, j = 0; i < rank; i++)
        {
            if (i == dim)
            {
                if (keepdim)
                    newShape[i] = 1;
            }
            else
            {
                newShape[keepdim ? i : j++] = Shape.Length(i);
            }
        }

        var output = Tensor<TNum>.Zeros(new Shape(newShape));
        var input = this.AsSpan();
        var outputSpan = output.AsSpan();

        // Fast flat loop over output elements
        for (int outer = 0; outer < outerSize; outer++)
        {
            var outOffsetBase = outer * innerSize;
            var outerAxisStride = outer * axisSize;

            for (int inner = 0; inner < innerSize; inner++)
            {
                int outputOffset = outOffsetBase + inner;
                var innerAxisStride = outerAxisStride * innerSize + inner;

                TNum sum = TNum.Zero;
                for (int a = 0; a < axisSize; a++)
                {
                    int inputOffset = innerAxisStride + a * axisStride;

                    sum += input[inputOffset];
                }

                outputSpan[outputOffset] = sum;
            }
        }

        return output;
    }

    /// <summary>
    /// Optimized sum reduction along the given axes
    /// </summary>
    /// <param name="axes">axes to sum over</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>reduced tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public Tensor<TNum> Sum(ReadOnlySpan<Index> axes, bool keepdim = true)
    {
        var shape = this.Shape;
        int rank = shape.Rank;

        if (axes.Length == 0)
            throw new ArgumentException("Axes must not be empty.", nameof(axes));

        // Normalize and prepare reduction info
        Span<bool> isReduced = stackalloc bool[rank];
        foreach (var ax in axes)
        {
            int norm = NormalizeAxis(ax);
            isReduced[norm] = true;
        }

        // Compute output shape
        int outRank = keepdim ? rank : rank - axes.Length;
        var outShapeArr = new int[outRank];
        {
            int j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (!isReduced[i])
                    outShapeArr[keepdim ? i : j++] = shape.Length(i);
                else if (keepdim)
                    outShapeArr[i] = 1;
            }
        }

        var outShape = new Shape(outShapeArr);
        var result = Tensor<TNum>.Zeros(outShape);

        // Get strides
        var inStrides = shape.AsStrideSpan();
        var outStrides = outShape.AsStrideSpan();

        // Precompute output stride deltas (0 if reduced)
        Span<int> outStridePerInputDim = stackalloc int[rank];
        {
            int j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (!isReduced[i])
                    outStridePerInputDim[i] = outStrides[keepdim ? i : j++];
                else
                    outStridePerInputDim[i] = 0;
            }
        }

        // Begin flat iteration
        int totalInputSize = this.ElementCount;
        var input = this.AsSpan();
        var output = result.AsSpan();

        Span<int> index = stackalloc int[rank]; // reused index
        int inputOffset = 0;
        int outputOffset = 0;

        for (int i = 0; i < totalInputSize; i++)
        {
            // Accumulate
            output[outputOffset] += input[i];

            // Advance index + offsets
            for (int d = rank - 1; d >= 0; d--)
            {
                index[d]++;
                inputOffset += inStrides[d];
                outputOffset += outStridePerInputDim[d];

                if (index[d] < shape.Length(d))
                    break;

                // Carry over
                index[d] = 0;
                inputOffset -= inStrides[d] * shape.Length(d);
                outputOffset -= outStridePerInputDim[d] * shape.Length(d);
            }
        }

        return result;
    }
    //[MethodImpl(MethodImplOptions.AggressiveInlining)]
    //public Tensor<TNum> Sum(ReadOnlySpan<Index> axes, bool keepdim = true) => Reduce(axes, TNum.Zero, static (a, b) => a + b, keepdim);

    /// <summary>
    /// Global sum of all elements in the tensor
    /// </summary>
    /// <returns>sum</returns>
    public TNum Sum()
    {
        TNum total = TNum.Zero;
        foreach (var val in this.elements)
            total += val;

        return total;
    }

    /// <summary>
    /// Mean reduction along the given axis
    /// </summary>
    /// <param name="axis">axis to sum over</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>reduced tensor</returns>
    public Tensor<TNum> Mean(Index axis, bool keepdim = true)
    {
        var positive_axis = NormalizeAxis(axis);

        var sum = Sum(positive_axis, keepdim); // If positive_axis was out of bounds I'd get an exception here before we can move on
        var divisor = TNum.One / TNum.CreateChecked(Shape.Length(positive_axis));

        sum.ScaleByInplace(divisor); // Leverage the vectorized scaling operation
        return sum;
    }

    /// <summary>
    /// Global mean of all elements in the tensor
    /// </summary>
    /// <returns>mean</returns>
    public TNum Mean()
    {
        TNum total = TNum.Zero;
        foreach (var val in this.elements)
            total += val;

        TNum mean = total / TNum.CreateChecked(this.elements.Length);
        return mean;
    }

    /// <summary>
    /// Product of all alements alng the given axis
    /// </summary>
    /// <param name="axis">axis to compute product over</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>reduced tensor</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TNum> Product(Index axis, bool keepdim = true) => this.Reduce(axis, TNum.One, static (acc, val) => acc * val, keepdim);

    /// <summary>
    /// Global product of all elements in the tensor
    /// </summary>
    /// <returns>product</returns>
    public TNum Product()
    {
        TNum total = TNum.One;
        foreach (var val in this.elements)
            total *= val;

        return total;
    }

    /// <summary>
    /// Mirror specific axes of this tensor
    /// </summary>
    /// <param name="axes">Axes to mirror (negative indexing supported)</param>
    /// <returns>Tensor with the provided axes mirrored</returns>
    /// <exception cref="ArgumentException">thrown if the axes are incorrect</exception>
    public Tensor<TNum> Mirror(params ReadOnlySpan<Index> axes)
    {
        var rank = Shape.Rank;

        Span<int> positive_axes = stackalloc int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            var axis_index = axes[i];
            positive_axes[i] = NormalizeAxis(axis_index);
        }

        var mirrored = Tensor<TNum>.Defaults(this.Shape);
        Span<int> target_indices = stackalloc int[rank];

        var ienumerator = Shape.CreateIndexEnumerator();
        Span<TNum> elements = this.elements;
        Span<int> indices = stackalloc int[Shape.Rank];
        int flatIndex = 0;

        ienumerator.Initialize(indices, ref flatIndex);
        while (ienumerator.MoveNext(indices, ref flatIndex))
        {
            indices.CopyTo(target_indices);
            for (int i = 0; i < axes.Length; i++)
            {
                var axis_index = positive_axes[i];
                target_indices[axis_index] = Shape.Length(axis_index) - 1 - target_indices[axis_index];
            }

            mirrored[target_indices] = elements[flatIndex];
        }

        return mirrored;
    }
    
    /// <summary>
    /// Mirror all axes of this tensor
    /// </summary>
    /// <returns>Tensor with the provided axes mirrored</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TNum> Mirror()
    {
        var shape = this.Shape;
        Span<Index> axes = stackalloc Index[shape.Rank];
        for (int i = 0; i < shape.Rank; i++)
            axes[i] = i;
        return Mirror(axes);
    }
    
    /// <summary>
    /// Perform a mirroring along the last 2 axes of this tensor 
    /// </summary>
    /// <param name="x">true if mirroring should occur along the last dimension (width)</param>
    /// <param name="y">true if mirroring should occur along the end last dimension (height)</param>
    /// <returns>mirrored tensor or self if no mirroring took place</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TNum> Mirror(bool x, bool y)
    {
        if (x && y)
            return Mirror(^2, ^1);
        else if (x)
            return Mirror(^1);
        else if (y)
            return Mirror(^2);
        else
            return this;
    }
    
    /// <summary>
    /// Transpose a tensor (reverse its dimensions)
    /// </summary>
    /// <returns>transposition tensor (permutation with reversed dimensions)</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<TNum> Transpose()
    {
        if (Shape.Rank == 2)
            return MatrixTranspose();

        var shape = this.Shape;
        Span<Index> perm = stackalloc Index[shape.Rank];
        for (int i = 0; i < shape.Rank; i++)
            perm[i] = shape.Rank - 1 - i;
        return Permute(perm);
    }

    /// <summary>
    /// <para>
    /// Transpose 2 dimensions of the tensor.
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// var matrixTranspose = tensor.Transpose(^2, ^1);
    /// </code>
    /// </para>
    /// </summary>
    /// <param name="dim0">first dimension</param>
    /// <param name="dim1">second dimension</param>
    /// <returns>Tensor with the 2 dimensions transposed</returns>
    /// <exception cref="InvalidOperationException">thrown if a transposition cannot be performed</exception>
    public Tensor<TNum> Transpose(Index dim0, Index dim1)
    {
        var rank = this.Shape.Rank;
        if (rank < 2)
            throw new InvalidOperationException("Cannot transpose a tensor with rank less than 2");

        int axisI = NormalizeAxis(dim0);
        int axisJ = NormalizeAxis(dim1);
        if (axisI == axisJ)
            return this;

        var orig_shape = this.Shape;
        ReadOnlySpan<int> orig_strides = orig_shape.AsStrideSpan();
        ReadOnlySpan<int> orig_dims = orig_shape.AsDimensionSpan();

        var new_dims = orig_dims.ToArray();
        (new_dims[axisI], new_dims[axisJ]) = (new_dims[axisJ], new_dims[axisI]);
        var new_shape = new Shape(new_dims);
        var new_strides = new_shape.AsStrideSpan();

        Span<int> idxs = stackalloc int[rank];
        int total = this.elements.Length;
        TNum[] new_elements = new TNum[total];

        for (int flat = 0; flat < total; flat++)
        {
            // swap axis in index
            int tmp = idxs[axisI];
            idxs[axisI] = idxs[axisJ];
            idxs[axisJ] = tmp;

            // Compute output flat index
            int outFlat = 0;
            for (int k = 0; k < rank; k++)
            {
                outFlat += idxs[k] * new_strides[k];
            }

            // Copy element
            new_elements[outFlat] = this.elements[flat];

            // Restore indices
            tmp = idxs[axisI];
            idxs[axisI] = idxs[axisJ];
            idxs[axisJ] = tmp;

            // Increment odometer
            for (int k = rank - 1; k >= 0; k--)
            {
                idxs[k]++;
                if (idxs[k] < orig_dims[k])
                    break;

                idxs[k] = 0;
            }
        }

        return new Tensor<TNum>(new_shape, new_elements);
        /*var rank = this.Shape.Rank;
        if (rank < 2)
            throw new InvalidOperationException("Cannot transpose a tensor with rank less than 2");

        int a = NormalizeAxis(dim0);
        int b = NormalizeAxis(dim1);
        if (a == b)
            return this;

        var oldShape = this.Shape;
        ReadOnlySpan<int> strides = oldShape.AsStrideSpan();
        ReadOnlySpan<int> dimsSizes = oldShape.AsDimensionSpan();
        var dims = dimsSizes.ToArray();
        var strs = strides.ToArray();
        (dims[a], dims[b]) = (dims[b], dims[a]);
        (strs[a], strs[b]) = (strs[b], strs[a]);
        var newShape = new Shape(dims, strs);

        // Create result tensor with proper dimensionality
        var result = Tensor<TNum>.Defaults(newShape);
        Span<int> transposed_indices = stackalloc int[rank];

        // Use proper index enumeration for all dimensions
        var ienumerator = oldShape.CreateIndexEnumerator();
        Span<int> indices = stackalloc int[rank];
        int flatIndex = 0;
        Span<TNum> elements = this.elements;

        ienumerator.Initialize(indices, ref flatIndex);
        while (ienumerator.MoveNext(indices, ref flatIndex))
        {
            // Copy indices and swap positions a and b
            indices.CopyTo(transposed_indices);
            (transposed_indices[a], transposed_indices[b]) = (transposed_indices[b], transposed_indices[a]);
            result[transposed_indices] = elements[flatIndex];
        }

        return result;*/
    }

    /// <summary>
    /// Transpose the last two dimensions of a tensor (matrix transpose)
    /// </summary>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException">invalid if there rank is less than 2</exception>
    public Tensor<TNum> MatrixTranspose()
    {
        var size = this.Shape;
        var rank = size.Rank;
        if (rank < 2)
            throw new InvalidOperationException("Cannot transpose a tensor with rank less than 2");
        var rows = size.Length(rank - 2);
        var cols = size.Length(rank - 1);
        var self_shape = Shape.AsDimensionSpan().ToArray();
        self_shape[rank - 2] = cols;
        self_shape[rank - 1] = rows;

        var length = rows * cols;
        var matrixCount = this.elements.Length / length;
        var results = new TNum[this.elements.Length];

        for (var m = 0; m < matrixCount; m++)
        {
            var srcOffset = m * length;
            var dstOffset = m * length;

            for (var r = 0; r < rows; r++)
            {
                var soff = srcOffset + r * cols;
                var roff = dstOffset + r;

                for (var c = 0; c < cols; c++)
                {
                    // Transpose: [r, c] -> [c, r]
                    results[roff + c * rows] = this.elements[soff + c];
                }
            }
        }

        return new Tensor<TNum>(new Shape(self_shape), results);
    }

    /// <summary>
    /// Apply a permutation to the axes of this tensor to produce a new tensor
    /// </summary>
    /// <param name="permutation">list of axis indices in the original tensor to use for the permuted tensor (negative indexing supported)</param>
    /// <returns>permuted tensor</returns>
    /// <exception cref="ArgumentException">thrown if the permutation list has missing or invalid elements</exception>
    public Tensor<TNum> Permute(params ReadOnlySpan<Index> permutation)
    {
        // Safety checks
        var rank = this.Shape.Rank;
        if (permutation.Length != rank)
            throw new ArgumentException("Permutation set must have the same rank as the tensor it is permuting");
        Span<int> positive_permutations = stackalloc int[permutation.Length];
        for (int i = 0; i < permutation.Length; i++)
        {
            var axis_index = permutation[i];
            positive_permutations[i] = NormalizeAxis(axis_index);
        }

        // New shape computation
        int[] new_shape = new int[rank];
        for (var i = 0; i < rank; i++)
        {
            new_shape[i] = Shape.Length(positive_permutations[i]);
        }

        // Do copying and reorganizing
        var result = Tensor<TNum>.Defaults(new Shape(new_shape));
        Span<int> transposed_indices = stackalloc int[rank];

        var ienumerator = Shape.CreateIndexEnumerator();
        Span<int> indices = stackalloc int[Shape.Rank];
        int flatIndex = 0;
        Span<TNum> elements = this.elements;

        ienumerator.Initialize(indices, ref flatIndex);
        while (ienumerator.MoveNext(indices, ref flatIndex))
        {
            for (var i = 0; i < rank; i++)
            {
                transposed_indices[i] = indices[positive_permutations[i]];
            }

            result[transposed_indices] = elements[flatIndex];
        }

        // Return the permuted matrix
        return result;
    }

    /// <summary>
    /// Compute the variance along the given axis
    /// </summary>
    /// <param name="axis">axis</param>
    /// <param name="ddof">degrees of freedom</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>variation tensor</returns>
    /// <exception cref="ArgumentOutOfRangeException">thrown when ddot is greater than or equal to axis length</exception>
    public Tensor<TNum> Variance(Index axis, int ddof = 0, bool keepdim = true)
    {
        int axisIndex = NormalizeAxis(axis);
        int count = this.Shape.Length(axisIndex);

        if (ddof >= count)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be less than the number of elements.");

        // Step 1: Compute mean
        var mean = this.Mean(axis, keepdim: true); // Pree sure this is broken because I don't really support broadcasting in subtracton

        // Step 2: Subtract mean and square (reuse mean tensor for all values)
        ElementWiseSubtract(mean.elements.Length, mean.elements, 0, this.elements, 0, mean.elements, 0); // Difference
        ElementWiseMultiply(mean.elements.Length, mean.elements, 0, mean.elements, 0, mean.elements, 0); // Square

        // Step 3: Reduce (sum of square differences)
        var sumsq = mean.Sum(axis);
        sumsq.ScaleByInplace(TNum.One / TNum.CreateChecked(count - ddof));
        return sumsq;
    }

    /// <summary>
    /// Compute the global variance of all elements in the tensor
    /// </summary>
    /// <param name="ddof">degrees of freedom</param>
    /// <returns>global variance</returns>
    /// <exception cref="ArgumentOutOfRangeException">thrown when ddot is greater than or equal to number of elements</exception>
    public TNum Variance(int ddof = 0)
    {
        int n = this.elements.Length;
        if (ddof >= n)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be less than the number of elements.");

        // Compute mean
        var mean = this.Mean();

        // Compute squared differences
        TNum sumSqDiff = TNum.Zero;
        foreach (var val in this.elements)
        {
            var diff = val - mean;
            sumSqDiff += diff * diff;
        }

        TNum variance = sumSqDiff / TNum.CreateChecked(n - ddof);
        return variance;
    }

    /// <summary>
    /// Compute the variance along the given axis
    /// </summary>
    /// <param name="axis">axis</param>
    /// <param name="ddof">degrees of freedom</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>variation tensor</returns>
    /// <exception cref="ArgumentOutOfRangeException">thrown when ddot is greater than or equal to axis length</exception>
    public Tensor<TNum> Variance(Index axis, out Tensor<TNum> mean, int ddof = 0, bool keepdim = true)
    {
        int axisIndex = NormalizeAxis(axis);
        int count = this.Shape.Length(axisIndex);

        if (ddof >= count)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be less than the number of elements.");

        // Step 1: Compute mean
        mean = this.Mean(axis, keepdim: true); // Pree sure this is broken because I don't really support broadcasting in subtracton

        // Step 2: Subtract mean and square (reuse mean tensor for all values)
        ElementWiseSubtract(mean.elements.Length, mean.elements, 0, this.elements, 0, mean.elements, 0); // Difference
        ElementWiseMultiply(mean.elements.Length, mean.elements, 0, mean.elements, 0, mean.elements, 0); // Square

        // Step 3: Reduce (sum of square differences)
        var sumsq = mean.Sum(axis);
        sumsq.ScaleByInplace(TNum.One / TNum.CreateChecked(count - ddof));
        return sumsq;
    }

    /// <summary>
    /// Compute the global variance of all elements in the tensor
    /// </summary>
    /// <param name="ddof">degrees of freedom</param>
    /// <returns>global variance</returns>
    /// <exception cref="ArgumentOutOfRangeException">thrown when ddot is greater than or equal to number of elements</exception>
    public TNum Variance(out TNum mean, int ddof = 0)
    {
        int n = this.elements.Length;
        if (ddof >= n)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be less than the number of elements.");

        // Compute mean
        mean = this.Mean();

        // Compute squared differences
        TNum sumSqDiff = TNum.Zero;
        foreach (var val in this.elements)
        {
            var diff = val - mean;
            sumSqDiff += diff * diff;
        }

        TNum variance = sumSqDiff / TNum.CreateChecked(n - ddof);
        return variance;
    }

    /// <summary>
    /// Concatenate the data from this tensor and another along the given axis
    /// </summary>
    /// <param name="other">tensor to concatenate with</param>
    /// <param name="axis">axis to concatenate along</param>
    /// <returns>concatenated tensor</returns>
    /// <exception cref="ArgumentException">thrown if concatenation is not possible</exception>
    public Tensor<TNum> Concat(Tensor<TNum> other, Index axis)
    {
        // Compute the minimum rank for concatenation
        var matchedRank = Math.Max(this.Shape.Rank, other.Shape.Rank);
        var dim = axis.GetOffset(matchedRank);
        if (dim < 0)
        {
            matchedRank = matchedRank + Math.Abs(dim);
            dim = axis.GetOffset(matchedRank);
        }

        // Force both tensors to be treated as if they are of the matched rank by padding with leading '1' if rank is to small
        var a_shape = this.Shape.EnsureRank(matchedRank);
        var a = this.ReshapeShared(a_shape);

        var b_shape = other.Shape.EnsureRank(matchedRank);
        var b = other.ReshapeShared(b_shape);

        // Validate the dimensions for compatibility
        for (int i = 0; i < matchedRank; i++)
        {
            if (i != dim && a_shape.Length(i) != b_shape.Length(i))
                throw new ArgumentException("Tensors must have the same shape on all axes except the concatenation axis.");
        }

        // Determine output shape
        var outDims = new int[matchedRank];
        for (int i = 0; i < matchedRank; i++)
        {
            if (i != dim)
                outDims[i] = a_shape.Length(i);
            else
                outDims[i] = a_shape.Length(i) + b_shape.Length(i);
        }
        var outShape = new Shape(outDims);

        // Concatenate
        var outTensor = Tensor<TNum>.Defaults(outShape);
        var outSpan = outTensor.AsSpan();
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        int rank = matchedRank;
        int innerBlockSize = 1;
        for (int i = dim + 1; i < rank; i++)
            innerBlockSize *= outShape.Length(i); // elements per slice

        int outerBlockSize = 1;
        for (int i = 0; i < dim; i++)
            outerBlockSize *= outShape.Length(i); // number of slices

        int aBlockSize = a_shape.Length(dim) * innerBlockSize;
        int bBlockSize = b_shape.Length(dim) * innerBlockSize;
        int outBlockSize = outShape.Length(dim) * innerBlockSize;

        int aOffset = 0;
        int bOffset = 0;
        int outOffset = 0;

        for (int i = 0; i < outerBlockSize; i++)
        {
            // Copy a block
            aSpan.Slice(aOffset, aBlockSize).CopyTo(outSpan.Slice(outOffset, aBlockSize));
            outOffset += aBlockSize;
            aOffset += aBlockSize;

            // Copy b block
            bSpan.Slice(bOffset, bBlockSize).CopyTo(outSpan.Slice(outOffset, bBlockSize));
            outOffset += bBlockSize;
            bOffset += bBlockSize;
        }

        return outTensor;
    }

    /// <summary>
    /// Test if this tensor has any elements that match the provided predicate
    /// </summary>
    /// <param name="predicate">element test condition</param>
    /// <returns>true if any elements match the condition</returns>
    public bool Any(Func<TNum, bool> predicate)
    {
        return this.elements.Any(predicate);
    }

    /// <summary>
    /// Test if this tensor has all elements matching the provided predicate
    /// </summary>
    /// <param name="predicate">element test condition</param>
    /// <returns>true if all elements match the condition</returns>
    public bool All(Func<TNum, bool> predicate)
    {
        return this.elements.All(predicate);
    }

    /// <summary>
    /// Create an exact duplicate of this tensor
    /// </summary>
    /// <returns>tensor</returns>
    public Tensor<TNum> Clone()
    {
        var tensor = Tensor<TNum>.Defaults(this.Shape);
        this.elements.CopyTo(tensor.elements, 0);
        return tensor;
    }

    /// <summary>
    /// Access the tensor elements as a span
    /// </summary>
    /// <returns>span of elements in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<TNum> AsSpan() => elements.AsSpan();

    /// <summary>
    /// Retrieve the span associated with the given subtensor leading indices
    /// </summary>
    /// <param name="indices">leading dimension indices pointing to the start of the subtensor</param>
    /// <returns>span over subtensor elements</returns>
    public Span<TNum> SubtensorSpan(params ReadOnlySpan<int> indices)
    {
        if (indices.Length == 0)
            return elements.AsSpan();

        // Compute the flat offset to the start of the span
        int offset = 0;
        var strides = Shape.AsStrideSpan();
        for (int i = 0; i < indices.Length; i++)
        {
            offset += indices[i] * strides[i];
        }

        // Deterime the span length
        var size = strides[indices.Length - 1];
        return elements.AsSpan(offset, size);
    }

    /// <summary>
    /// Access the tensor elements as a span
    /// </summary>
    /// <returns>span of elements in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<TNum> AsSpan(int start, int length) => elements.AsSpan(start, length);

    /// <summary>
    /// Create a 2D span over the given region to access tensor elements
    /// </summary>
    /// <param name="start">start offset</param>
    /// <param name="rows">number of rows in span</param>
    /// <param name="columns">number of columns in span</param>
    /// <returns>span of elements in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span2D<TNum> AsSpan2D(int start, int rows, int columns) => new Span2D<TNum>(elements.AsSpan(start, rows * columns), rows, columns);

    /// <summary>
    /// Create a 3D span over the given region to access tensor elements
    /// </summary>
    /// <param name="start">start offset</param>
    /// <param name="channels">number of channels in span</param>
    /// <param name="rows">number of rows in span</param>
    /// <param name="columns">number of columns in span</param>
    /// <returns>span of elements in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span3D<TNum> AsSpan3D(int start, int channels, int rows, int columns) => new Span3D<TNum>(elements.AsSpan(start, channels * rows * columns), channels, rows, columns);

    /// <summary>
    /// Access the tensor elements as a span
    /// </summary>
    /// <returns>span of elements in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<TNum> AsSpan(int start) => elements.AsSpan(start);

    /// <summary>
    /// Access the tensor elements as a span
    /// </summary>
    /// <returns>span of elements in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<TNum> AsSpan(Index start) => elements.AsSpan(start);

    /// <summary>
    /// Access the tensor elements as a span
    /// </summary>
    /// <returns>span of elements in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<TNum> AsSpan(Range range) => elements.AsSpan(range);

    /// <summary>
    /// Access the tensor elements as an array (non-copying)
    /// </summary>
    /// <returns>array of elements in row-major order</returns>
    public TNum[] AsArray() => elements;

    /// <summary>
    /// Convert the tensor to a C# rectangular array
    /// </summary>
    /// <returns>rectangular array</returns>
    /// <exception cref="InvalidOperationException">thrown if the tensor shape is invalid</exception>
    public Array ToRectangularArray()
    {
        var shape = this.Shape;
        if (shape.Rank == 0)
            return Array.Empty<TNum>();

        var result = Array.CreateInstance(typeof(TNum), shape.ToArray());
        this.elements.CopyTo(result, 0); // Since both are row-major order we can just copy the values over directly
        return result;
    }

    /// <summary>
    /// Convert a tensor to a C# jagged array (array of arrays)
    /// </summary>
    /// <returns>jagged array</returns>
    public Array ToJaggedArray()
    {
        var shape = this.Shape;
        if (shape.Rank == 0)
            return Array.Empty<TNum>();

        return ToJaggedArray(0, 0);
    }
    private Array ToJaggedArray(int dim, int offset)
    {
        var length = this.Shape.Length(dim);
        var type = typeof(TNum);
        for (int i = Shape.Rank - 1; i > dim; i--)
            type = type.MakeArrayType();
        var result = Array.CreateInstance(type, length);
        if (dim == Shape.Rank - 1)
        {
            for (var i = 0; i < length; i++)
            {
                // Copy items
                result.SetValue(this.elements[offset + i], i);
            }
        }
        else
        {
            int stride = Shape.Stride(dim);

            for (var i = 0; i < length; i++)
            {
                // Recurse down 
                var subOffset = offset + i * stride;
                result.SetValue(ToJaggedArray(dim + 1, subOffset), i);
            }
        }

        return result;
    }

    public override string ToString() => ToString('{', '}');
    /// <summary>
    /// Create a string formatted for compatibility with Wolfram Alpha 
    /// </summary>
    /// <returns>string</returns>
    public string ToWolframString() => ToString('{', '}');
    /// <summary>
    /// Create a string formatted for JSON like documents
    /// </summary>
    /// <returns>string</returns>
    public string ToJsonString() => ToString('[', ']');
    /// <summary>
    /// Write a custom string with the provided opening and closing characters for each dimension
    /// </summary>
    /// <param name="open">open dimension character</param>
    /// <param name="close">close dimension character</param>
    /// <returns>formatted string</returns>
    public string ToString(char open, char close)
    {
        StringBuilder sb = new StringBuilder();
        WriteDimString(0, 0, sb, open, close);
        return sb.ToString();
    }
    private void WriteDimString(int dim, int offset, StringBuilder sb, char dimOpen, char dimClose)
    {
        var length = this.Shape.Length(dim);

        if (dim == Shape.Rank - 1)
        {
            for (var i = 0; i < length; i++)
            {
                // Write item
                if (i != 0)
                    sb.Append(',');
                sb.Append(this.elements[offset + i]);
            }
        }
        else
        {
            int stride = Shape.Stride(dim);

            for (var i = 0; i < length; i++)
            {
                // Write dimension. eg: { ... } or [ ... ] depending on dimOpen and dimClose
                if (i != 0)
                    sb.Append(',');
                sb.Append(dimOpen);
                var subOffset = offset + i * stride;
                WriteDimString(dim + 1, subOffset, sb, dimOpen, dimClose);
                sb.Append(dimClose);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Shuffle(Span<TNum> span) {
        // Fisher-Yates shuffle
        var rng = System.Random.Shared;

        for (var i = span.Length - 1; i >= 1; i--)
        {
            var j = rng.Next(i);

            var ej = span[j];
            span[j] = span[i];
            span[i] = ej;
        }
    }

    /// <summary>
    /// Shuffle the elements of the tensor
    /// </summary>
    public void Shuffle() {
        Shuffle(this.elements);
    }

    /// <summary>
    /// Squeeze the tensor, removing all 1 length dimensions
    /// </summary>
    /// <returns>original tensor but with a new shape</returns>
    public Tensor<TNum> Squeeze()
    {
        var shape = this.Shape;
        var dims = shape.AsDimensionEnumerable();
        var newDims = dims.Where(d => d != 1).ToArray();
        return this.ReshapeShared(new Shape(newDims));
    }

    /// <summary>
    /// Squeeze a give dimension of the tensor removing it if it is of length 1
    /// </summary>
    /// <param name="axis">dimension to squeeze</param>
    /// <returns>original tensor but with a new shape</returns>
    public Tensor<TNum> Squeeze(Index axis)
    {
        var shape = this.Shape;
        var dims = shape.AsDimensionSpan();
        var positiveIndex = this.NormalizeAxis(axis); // Throws if out of bounds

        if (dims[positiveIndex] != 1)
            return this; // No need to squeeze, its not 1

        var newDims = new int[dims.Length - 1];
        if (positiveIndex > 0)
            dims.Slice(0, positiveIndex).CopyTo(newDims);
        if (positiveIndex < dims.Length - 1)
            dims.Slice(positiveIndex + 1).CopyTo(newDims.AsSpan(positiveIndex));

        return this.ReshapeShared(new Shape(newDims));
    }

    /// <summary>
    /// Unsqueeze a tensor by adding a 1 length dimension at the given dimension index
    /// </summary>
    /// <param name="axis">index to insert the dimension at</param>
    /// <returns>original tensor but with a new shape</returns>
    public Tensor<TNum> Unsqueeze(Index axis)
    {
        var shape = this.Shape;
        var dims = shape.AsDimensionSpan();
        var positiveIndex = axis.GetOffset(dims.Length + 1);
        if (positiveIndex < 0 || positiveIndex > dims.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis must be between 0 and Rank (inclusive) for unsqueeze.");

        // Copy before insertion point
        var newDims = new int[dims.Length + 1];
        if (positiveIndex > 0)
            dims.Slice(0, positiveIndex).CopyTo(newDims);

        // Insert the new dimension
        newDims[positiveIndex] = 1;

        // Copy after insertion point
        if (positiveIndex < dims.Length)
            dims.Slice(positiveIndex).CopyTo(newDims.AsSpan(positiveIndex + 1));

        return this.ReshapeShared(new Shape(newDims));
    }

    /// <summary>
    /// Squeeze a tensor by removing dimensions if their length is 1
    /// </summary>
    /// <param name="axes">dimensions to squeeze</param>
    /// <returns>original tensor but with a new shape</returns>
    public Tensor<TNum> Squeeze(params ReadOnlySpan<Index> axes)
    {
        var shape = this.Shape;
        var dims = shape.AsDimensionSpan();
        var rank = dims.Length;
        Span<int> squeezeDims = stackalloc int[axes.Length];
        int squeezeDimsLength = 0;
        for (var i = 0; i < axes.Length; i++)
        {
            var axis = this.NormalizeAxis(axes[i]); // Throws if out of bounds
            if (dims[axis] != 1)
                continue;                           // Not 1, no need to squeeze it
            squeezeDims[squeezeDimsLength++] = axis;
        }

        if (squeezeDimsLength == 0)
            return this;

        squeezeDims.Slice(0, squeezeDimsLength).Sort(); // Sort all valid squeezed dims in increasing order

        var newDims = new int[dims.Length - squeezeDimsLength];
        var dst = 0; var src = 0; var squeezeDim = 0;

        while (src < rank)
        {
            if (squeezeDim < squeezeDimsLength && src == squeezeDims[squeezeDim])
            {
                // Skip this axis
                src++;
                squeezeDim++;
            }
            else
            {
                // Copy the axis
                newDims[dst++] = dims[src++];
            }
        }

        return this.ReshapeShared(new Shape(newDims));
    }

    /// <summary>
    /// Squeeze all dimensions in a given range removing dimensions of length 1 within that range
    /// </summary>
    /// <param name="range">range of dimensions</param>
    /// <returns>original tensor but with a new shape</returns>
    public Tensor<TNum> Squeeze(Range range)
    {
        var shape = this.Shape;
        var dims = shape.AsDimensionEnumerable();
        var (start, length) = range.GetOffsetAndLength(shape.Rank);

        if (start < 0 || start > shape.Rank || start + length < 0 || start + length > shape.Rank)
            throw new ArgumentOutOfRangeException(nameof(range), $"Range is invalid for a tensor of rank {shape.Rank}");

        var newDims = dims.Where((dim, index) => index < start || index >= start + length || dim != 1).ToArray();
        return this.ReshapeShared(new Shape(newDims));
    }

    /// <summary>
    /// Unsqueeze a tensor by adding 1-length dimensions at the given dimension indices
    /// </summary>
    /// <param name="axes">indices to insert the dimensions at</param>
    /// <returns>original tensor but with a new shape</returns>
    public Tensor<TNum> Unsqueeze(params ReadOnlySpan<Index> axes)
    {
        var shape = this.Shape;
        var dims = shape.AsDimensionSpan();
        int originalRank = dims.Length;
        int newRank = originalRank + axes.Length;

        if (axes.Length == 0)
            return this;

        // Normalize axes to positive integers relative to the new shape (rank after insertion)
        Span<int> insertPositions = stackalloc int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            int pos = axes[i].GetOffset(newRank);
            if (pos < 0 || pos > newRank)
                throw new ArgumentOutOfRangeException(nameof(axes), $"Index {axes[i]} is out of range for unsqueeze on rank {newRank}.");
            insertPositions[i] = pos;
        }

        // Sort insertion positions ascending
        insertPositions.Sort();

        // Build new dims array with size increased by number of inserted dims
        var newDims = new int[newRank];

        int src = 0;       // Index in original dims
        int dst = 0;       // Index in newDims
        int insertIdx = 0; // Index in insertPositions

        while (dst < newRank)
        {
            if (insertIdx < insertPositions.Length && dst == insertPositions[insertIdx])
            {
                // Insert a new dim of length 1 at this position
                newDims[dst++] = 1;
                insertIdx++;
            }
            else
            {
                // Copy dim from original tensor
                newDims[dst++] = dims[src++];
            }
        }

        return this.ReshapeShared(new Shape(newDims));
    }

    /// <summary>
    /// Lightweight hash for the tensor including the tensor shape and some of its values
    /// </summary>
    /// <returns>hashcode</returns>
    public override int GetHashCode()
    {
        var hash = new HashCode();

        // Shape encoding
        hash.Add(Shape.Rank);
        foreach (var dim in Shape.AsDimensionSpan())
            hash.Add(dim);

        // Data length
        hash.Add(elements.Length);                      

        // Sample 3 key elements: first, middle, last
        if (elements.Length > 0)
            hash.Add(elements[0]);                      // First element
        if (elements.Length > 2)
            hash.Add(elements[elements.Length / 2]);    // Middle element
        if (elements.Length > 1)
            hash.Add(elements[elements.Length - 1]);    // Last element

        return hash.ToHashCode();
    }

    /// <summary>
    /// Equality comparison
    /// </summary>
    /// <param name="obj">check if this tensor equals the given object</param>
    /// <returns>true if the tensor is equal to the object</returns>
    public override bool Equals(object? obj)
    {
        if (obj is not Tensor<TNum> other)
            return false;
        if (!this.Shape.Equals(other.Shape))
            return false;
        return this.elements.SequenceEqual(other.elements);
    }

    /// <summary>
    /// Equality comparison with a given amount of allowable error
    /// </summary>
    /// <param name="obj">check if this tensor equals the given object</param>
    /// <param name="delta">allowable error for element comparison</param>
    /// <returns>true if the tensor is equal to the object within the allowable error</returns>
    public bool Equals(object? obj, TNum delta)
    {
        if (obj is not Tensor<TNum> other)
            return false;
        if (!this.Shape.Equals(other.Shape))
            return false;
        if (this.ElementCount != other.ElementCount)
            return false;
        for (int i = 0; i < this.elements.Length; i++)
        {
            var a = this.elements[i];
            var b = other.elements[i];
            if (TNum.Abs(a - b) > delta)
                return false;
        }
        return true;
    }
}