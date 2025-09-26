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
/// Row-major enumerator struct for fast index enumeration using Span based memory access
/// </summary>
public readonly struct RowMajorIndexSpanEnumerator
{
    private readonly TensorShape shape;

    public RowMajorIndexSpanEnumerator(TensorShape shape)
    {
        this.shape = shape;
    }

    /// <summary>
    /// Initialize a buffer for enumeration
    /// </summary>
    /// <param name="indices">indices span to initialize</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Initialize(Span<int> indices)
    {
        indices.Fill(0);
        indices[^1] = -1;
    }

    /// <summary>
    /// Initialize an index for enumeration
    /// </summary>
    /// <param name="indices">indices span to initialize</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Initialize(ref int index)
    {
        index = -1;
    }

    /// <summary>
    /// <para>
    /// Advances the provided index buffer to the next index. Before the first call the buffer must be initialized to all -1s see <see cref="Initialize"/>.
    /// </para>
    /// <para>
    /// Usage:
    /// <code>
    /// var enumerator = new RowMajorIndexSpanEnumerator(shape);
    /// Span&lt;int&gt; indices = stackallock int[rank];
    /// 
    /// enumerator.Initialize(indices);
    /// while (enumerator.MoveNext(indices)) { ... }
    /// </code>
    /// </para>
    /// </summary>
    /// <param name="indices">location of the current index and destination to store the next computed index</param>
    /// <returns>true if the next index is valid</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool MoveNext(Span<int> indices)
    {
        Debug.Assert(indices.Length == shape.Rank, "Index span must match shape rank");
        int dim = indices.Length - 1;

        while (dim >= 0)
        {
            indices[dim]++;
            if (indices[dim] < shape.Length(dim))
                return true;

            indices[dim] = 0;
            dim--;
        }

        return false; // All indices exhausted
    }

    /// <summary>
    /// Initialize a buffer for enumeration
    /// </summary>
    /// <param name="indices">indices span to initialize</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Initialize(Span<int> indices, ref int index)
    {
        indices.Fill(0);
        indices[^1] = -1;

        index = -1;
    }

    /// <summary>
    /// <para>
    /// Advances the provided index buffer to the next index. Before the first call the buffer must be initialized to all -1s see <see cref="Initialize"/> and the flat index reference should also be -1.
    /// </para>
    /// <para>
    /// Usage:
    /// <code>
    /// var enumerator = new RowMajorIndexSpanEnumerator(shape);
    /// Span&lt;int&gt; indices = stackallock int[rank];
    /// int flatIndex = 0;
    /// 
    /// enumerator.Initialize(indices, ref flatIndex);
    /// while (enumerator.MoveNext(indices, ref flatIndex)) { ... }
    /// </code>
    /// </para>
    /// </summary>
    /// <param name="indices">location of the current index and destination to store the next computed index</param>
    /// <param name="index">flattened 1d index in the underlying array, respecting stride</param>
    /// <returns>true if the next index is valid</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool MoveNext(Span<int> indices, ref int index)
    {
        var shape = this.shape;
        Debug.Assert(indices.Length == shape.Rank, "Index span must match shape rank");
        int dim = indices.Length - 1;

        while (dim >= 0)
        {
            indices[dim]++;
            index += shape.Stride(dim);
            if (indices[dim] < shape.Length(dim))
                return true;

            int overstep = indices[dim];
            index -= overstep * shape.Stride(dim);
            indices[dim] = 0;
            dim--;
        }

        return false; // All indices exhausted
    }
}

/// <summary>
/// Shape of a tensor
/// </summary>
public readonly struct TensorShape: IShape
{
    private readonly int[] dims;
    private readonly int[] strides;

    public TensorShape(params int[] dims)
    {
        this.dims = dims;
        this.strides = ComputeStrides(dims);
    }

    // For internal methods only, not for general use
    internal TensorShape(int[] dims, int[] strides)
    {
        this.dims = dims;
        this.strides = strides;
    }

    /// <summary>
    /// Clone this shape by copying dimension lengths only. Strides will be recomputed for this shape.
    /// </summary>
    /// <returns>stride array</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorShape CloneDimensions() {
        return new TensorShape(this.dims);
    }

    /// <summary>
    /// compute the values of the stride array
    /// </summary>
    /// <returns>stride array</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int[] ComputeStrides(int[] dims)
    {
        var s = new int[dims.Length];
        int stride = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            s[i] = stride;
            stride *= dims[i];
        }
        return s;
    }

    /// <summary>
    /// Length of the given dimension of the shape
    /// </summary>
    /// <param name="dim">dimension index</param>
    /// <returns>dimension length</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Length(int dim) => dims[dim];

    /// <summary>
    /// Length of the given dimension of the shape
    /// </summary>
    /// <param name="dim">dimension index</param>
    /// <returns>dimension length</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Length(Index index) => dims[index];

    /// <summary>
    /// Rank of the shape (number of dimensions)
    /// </summary>
    /// <returns>rank</returns>
    public int Rank
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => dims.Length;
    }

    /// <summary>
    /// Stride for a given dimension
    /// </summary>
    /// <param name="dim">dimension index</param>
    /// <returns>dimension stride</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Stride(int dim) {
        return strides[dim];
    }

    /// <summary>
    /// Number of logical (virtual) elements represented by the given shape
    /// </summary>
    /// <returns>number of elements</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int LogicalElementCount()
    {
        if (dims.Length == 0)
            return 0;

        int count = 1;
        for (var i = 0; i < dims.Length; i++)
        {
            count *= dims[i];
        }
        return count;
    }

    /// <summary>
    /// Number of actual elements used to represent this tensor in memory
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int StorageElementCount()
    {
        int minOffset = 0;
        int maxOffset = 0;
        ReadOnlySpan<int> dims = this.dims, strides = this.strides;
        for (int i = 0; i < dims.Length; i++)
        {
            int stride = strides[i];
            var extent = dims[i];

            int offset1 = 0;
            int offset2 = (extent - 1) * stride;

            minOffset += Math.Min(offset1, offset2);
            maxOffset += Math.Max(offset1, offset2);
        }

        return maxOffset - minOffset + 1;
    }

    /// <summary>
    /// Enumerate in row-major order over all possible ND indices representable by this shape. The span is populated by the next index each time it is checked
    /// </summary>
    /// <param name="indices">location to store indices</param>
    /// <returns>true if the loop should continue (next index is valid) false otherwise</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public RowMajorIndexSpanEnumerator CreateIndexEnumerator() => new RowMajorIndexSpanEnumerator(this);

    /// <summary>
    /// Transform an ND index for this shape into a row-major order flattened 1d index
    /// </summary>
    /// <param name="indices">nd index congruent with this shape</param>
    /// <returns>1d row-major order flattened index</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int FlattenIndices(ReadOnlySpan<int> indices)
    {
        // Row-major order flattening ND index to 1D index
        int index = 0;
        ReadOnlySpan<int> strides = this.strides;
        var rank = this.dims.Length;

        // Unroll for common ranks
        switch (rank) {
            case 1: 
                return indices[0] * strides[0];
            case 2:
                return indices[0] * strides[0] + indices[1] * strides[1];
            case 3:
                return indices[0] * strides[0] + indices[1] * strides[1] + indices[2] * strides[2];
            case 4:
                return indices[0] * strides[0] + indices[1] * strides[1] + indices[2] * strides[2] + indices[3] * strides[3];
            default:
                for (int i = 0; i < rank; i++)
                {
                    index += indices[i] * strides[i];
                }

                return index;
        }
    }

    /// <summary>
    /// Transform a 1D row-major order flattened index into an ND index
    /// </summary>
    /// <param name="index">1d row-major order flattened index</param>
    /// <param name="indices">where to place the ND computed indices</param>
    /// <returns>nd index congruent with this shape</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UnflattenIndex(int index, Span<int> indices)
    {
        // Row-major order unflattening 1D index to ND index
        for (var i = dims.Length - 1; i >= 0; i--)
        {
            var dim = dims[i];
            indices[i] = index % dim;
            index /= dim;
        }
    }

    /// <summary>
    /// Minimum dimension length
    /// </summary>
    /// <returns>minimum length</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int MinLength()
    {
        if (dims.Length == 0) return 0;
        int min = dims[0];
        for (int i = 1; i < dims.Length; i++)
            if (dims[i] < min) min = dims[i];
        return min;
    }

    /// <summary>
    /// Maximum dimension length
    /// </summary>
    /// <returns>maximum length</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int MaxLength()
    {
        if (dims.Length == 0) return 0;
        int max = dims[0];
        for (int i = 1; i < dims.Length; i++)
            if (dims[i] > max) max = dims[i];
        return max;
    }

    /// <summary>
    /// Create a new shape from the concatenation of this shape with another
    /// </summary>
    /// <param name="rhs">shape to concatenate onto the end</param>
    /// <returns>new shape</returns>
    public TensorShape Append(TensorShape rhs)
    {
        var shape = new int[this.Rank + rhs.Rank];
        this.dims.CopyTo(shape, 0);
        rhs.dims.CopyTo(shape, this.Rank);
        return new TensorShape(shape);
    }

    /// <summary>
    /// Create a new shape from the concatenation of this shape with another
    /// </summary>
    /// <param name="rhs">shape to concatenate onto the start</param>
    /// <returns>new shape</returns>
    public TensorShape Prepend(TensorShape lhs)
    {
        var shape = new int[this.Rank + lhs.Rank];
        lhs.dims.CopyTo(shape, 0);
        this.dims.CopyTo(shape, lhs.Rank);
        return new TensorShape(shape);
    }

    /// <summary>
    /// Create a new shape from a subset of this shape
    /// </summary>
    /// <param name="range">range to slice over</param>
    /// <returns>new shape</returns>
    public TensorShape Slice(Range range)
    {
        return new TensorShape(this.dims[range]);
    }

    /// <summary>
    /// Ensure that the shape has at least the given rank
    /// </summary>
    /// <param name="rank">rank of the shape</param>
    /// <returns>new shape with additional dimensions of length 1 if the shape's rank < length</returns>
    public TensorShape EnsureRank(int rank)
    {
        if (dims.Length >= rank)
            return this;

        int[] new_dims = new int[rank];
        int old_dims_offset = rank - this.dims.Length;
        this.dims.CopyTo(new_dims, old_dims_offset);
        for (var i = 0; i < old_dims_offset; i++)
            new_dims[i] = 1;
        return new TensorShape(new_dims);
    }

    /// <summary>
    /// Normalize the shape to the given rank broadcasting/expanding if required or reducing if required
    /// </summary>
    /// <param name="rank">normalized rank</param>
    /// <returns>tensor shape with the provided rank</returns>
    public TensorShape NormalizeRank(int rank)
    {
        if (rank < 1)
            throw new ArgumentException("Target rank must be >= 1");
        if (dims.Length < rank)
            return EnsureRank(rank);  // Smaller than length, expand to length
        if (dims.Length == rank)
            return this;                // Same length do nothing

        // Larger than length, collapse leading
        var currentRank = this.Rank;
        var collapseTo = Math.Max(0, currentRank - (rank - 1));
        int[] new_dims = new int[rank];
        int collapsed = 1;
        for (int i = 0; i < collapseTo; i++)
        {
            collapsed *= dims[i];
        }
        new_dims[0] = collapsed;
        for (int i = 1; i < rank; i++)
        {
            new_dims[i] = dims[collapseTo + (i - 1)];
        }
        return new TensorShape(new_dims);
    }

    /// <summary>
    /// Test if the last dimensions match the given shape
    /// </summary>
    /// <param name="other">shape of the last dimensions</param>
    /// <returns>true if the last dimensions match the shape, false otherwise</returns>
    public bool AreTrailingDimensions(TensorShape other)
    {
        var self = this.dims;       var self_length = self.Length;          var self_offset = self_length - 1;
        var smaller = other.dims;   var smaller_length = smaller.Length;    var smaller_offset = smaller_length - 1;

        if (smaller_length > self_length)
            return false;

        for (var i = 0; i < smaller_length; i++)
        {
            if (self[self_offset - i] != smaller[smaller_offset - i]) {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Collapse the given dimensions into one dimension with a combined size
    /// </summary>
    /// <param name="range">range to collapse</param>
    /// <returns>tensor shape with the given dimensions collapsed and the rest preserved</returns>
    public TensorShape CollapseDimensions(Range range)
    {
        var old_length = this.dims.Length;
        var (start, length) = range.GetOffsetAndLength(old_length);
        var new_length = old_length - length + 1;

        var new_dims = new int[new_length];

        var product = 1;
        for (var i = start; i < start + length; i++)
        {
            product *= this.dims[i];
        }

        Array.Copy(this.dims, 0, new_dims, 0, start);
        new_dims[start] = product;
        Array.Copy(this.dims, start + length, new_dims, start + 1, old_length - (start + length));

        return new TensorShape(new_dims);
    }

    /// <summary>
    /// Collapse the given dimensions into two dimension (column vector) with a combined size
    /// </summary>
    /// <param name="range">range to collapse</param>
    /// <returns>tensor shape with the given dimensions collapsed and the rest preserved</returns>
    public TensorShape CollapseDimensionsToColumn(Range range)
    {
        var old_length = this.dims.Length;
        var (start, length) = range.GetOffsetAndLength(old_length);
        var new_length = old_length - length + 2;

        var new_dims = new int[new_length];

        var product = 1;
        for (var i = start; i < start + length; i++)
        {
            product *= this.dims[i];
        }

        Array.Copy(this.dims, 0, new_dims, 0, start);
        new_dims[start] = product;
        new_dims[start + 1] = 1;
        Array.Copy(this.dims, start + length, new_dims, start + 2, old_length - (start + length));

        return new TensorShape(new_dims);
    }

    /// <summary>
    /// Collapse the given dimensions into two dimension (row vector) with a combined size
    /// </summary>
    /// <param name="range">range to collapse</param>
    /// <returns>tensor shape with the given dimensions collapsed and the rest preserved</returns>
    public TensorShape CollapseDimensionsToRow(Range range)
    {
        var old_length = this.dims.Length;
        var (start, length) = range.GetOffsetAndLength(old_length);
        var new_length = old_length - length + 2;

        var new_dims = new int[new_length];

        var product = 1;
        for (var i = start; i < start + length; i++)
        {
            product *= this.dims[i];
        }

        Array.Copy(this.dims, 0, new_dims, 0, start);
        new_dims[start] = 1;
        new_dims[start + 1] = product;
        Array.Copy(this.dims, start + length, new_dims, start + 2, old_length - (start + length));

        return new TensorShape(new_dims);
    }

    /// <summary>
    /// Broadcast this shape to match a target shape
    /// </summary>
    /// <param name="target">shape to match</param>
    /// <returns>broadcasted shape</returns>
    public TensorShape BroadcastTo(TensorShape target)
    {
        return BroadcastTo(target, 0, target.Rank); // Broadcast all dims by default
    }

    /// <summary>
    /// Broadcast this shape to match a target shape (over a specific region)
    /// </summary>
    /// <param name="target">shape to match</param>
    /// <param name="start">region to start broadcasting from</param>
    /// <param name="length">length of the region to broadcast</param>
    /// <returns>broadcasted shape</returns>
    public TensorShape BroadcastTo(TensorShape target, int start, int length)
    {
        var sourceDims = this.AsDimensionSpan();
        var sourceStrides = this.strides;
        var targetDims = target.AsDimensionSpan();

        int sourceRank = sourceDims.Length;
        int targetRank = targetDims.Length;

        if (start < 0 || length < 0 || start + length > targetRank)
            throw new ArgumentOutOfRangeException("Invalid broadcast dimension range");

        int[] newStrides = new int[targetRank];
        int[] resultDims = new int[targetRank];

        int rankOffset = targetRank - sourceRank;

        for (int i = targetRank - 1; i >= 0; i--)
        {
            int srcIdx = i - rankOffset;

            int sDim = srcIdx >= 0 ? sourceDims[srcIdx] : 1;
            int sStride = srcIdx >= 0 ? sourceStrides[srcIdx] : 0;
            int tDim = targetDims[i];

            bool allowBroadcast = (i >= start && i < start + length);

            if (!allowBroadcast)
            {
                // Preserve source dim/stride regardless of target
                resultDims[i] = sDim;
                newStrides[i] = sStride;
                continue;
            }

            if (sDim == tDim)
            {
                resultDims[i] = tDim;
                newStrides[i] = sStride;
            }
            else if (sDim == 1)
            {
                resultDims[i] = tDim;
                newStrides[i] = 0; // Broadcasting, reuse same element
            }
            else
            {
                throw new InvalidOperationException($"Cannot broadcast dimension {sDim} to {tDim} at axis {i}");
            }
        }

        return new TensorShape(resultDims, newStrides);
    }

    /// <summary>
    /// Create a shape that both shapes can be broadcasted to
    /// </summary>
    /// <param name="a">first shape</param>
    /// <param name="b">second shape</param>
    /// <returns>broacasted shape</returns>
    /// <exception cref="InvalidOperationException">thrown if the broadcast cannot be performed</exception>
    public static TensorShape ComputeBroadcastShape(TensorShape a, TensorShape b)
    {
        var aDims = a.AsDimensionSpan();
        var bDims = b.AsDimensionSpan();
        int aRank = aDims.Length;
        int bRank = bDims.Length;
        int resultRank = Math.Max(aRank, bRank);

        int[] resultDims = new int[resultRank];

        for (int i = 0; i < resultRank; i++)
        {
            int aIndex = i - (resultRank - aRank);
            int bIndex = i - (resultRank - bRank);

            int aDim = aIndex >= 0 ? aDims[aIndex] : 1;
            int bDim = bIndex >= 0 ? bDims[bIndex] : 1;

            if (aDim == bDim)
                resultDims[i] = aDim;
            else if (aDim == 1)
                resultDims[i] = bDim;
            else if (bDim == 1)
                resultDims[i] = aDim;
            else
                throw new InvalidOperationException($"Shapes {a} and {b} are not broadcastable at axis {i}");
        }

        return new TensorShape(resultDims);
    }

    public static TensorShape operator +(TensorShape a, TensorShape b) => a.Append(b);

    /// <summary>
    /// Convert this shape to a standard array of dimension lengths by copying the dimensions into a new array
    /// </summary>
    /// <returns>dimension length array</returns>
    public int[] ToArray()
    {
        return (int[])(dims.Clone());
    }

    /// <summary>
    /// Access this shape as a span of dimension lengths
    /// </summary>
    /// <returns>dimension length span</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<int> AsDimensionSpan() => dims;

    /// <summary>
    /// Access this shape as an enumerable of dimension lengths
    /// </summary>
    /// <returns></returns>
    public IEnumerable<int> AsDimensionEnumerable()
    {
        var span = dims;
        foreach (var dim in span)
            yield return dim;
    }

    /// <summary>
    /// Access this shape as a span of stride lengths
    /// </summary>
    /// <returns>dimension length span</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<int> AsStrideSpan() => strides;

    /// <summary>
    /// Access this shape as an enumerable of stride lengths
    /// </summary>
    /// <returns></returns>
    public IEnumerable<int> AsStrideEnumerable()
    {
        var span = strides;
        foreach (var dim in span)
            yield return dim;
    }

    public override bool Equals([NotNullWhen(true)] object? obj)
    {
        if (obj is TensorShape shape)
        {
            if (this.dims.Length != shape.dims.Length)
                return false;

            for (var i = 0; i < this.dims.Length; i++)
                if (this.dims[i] != shape.dims[i])
                    return false;
            return true;
        }
        else
        {
            return false;
        }
    }

    public override int GetHashCode()
    {
        if (dims is null) return 0;

        var hash = new HashCode();
        foreach (var item in dims)
        {
            hash.Add(item);
        }
        return hash.ToHashCode();
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append('(');
        for (var i = 0; i < dims.Length; i++)
        {
            if (i != 0)
                sb.Append('x');
            sb.Append(dims[i]);
        }
        sb.Append(')');
        return sb.ToString();
    }
}
