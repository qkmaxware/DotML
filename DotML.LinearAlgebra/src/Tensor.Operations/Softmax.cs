using System.Collections;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Net;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace DotML;

/// <summary>
/// Extension methods adding a Variance and Std operations to supported tensor types
/// </summary>
public static class TensorSoftmax
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int NormalizeAxis(Index axis, Shape Shape)
    {
        var rank = Shape.Rank;
        var axisi = axis.GetOffset(rank);
        if (axisi < 0 || axisi >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis is out of range");
        return axisi;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IncrementIndex(Span<int> index, ReadOnlySpan<int> shape, ReadOnlySpan<int> dimsToUpdate)
    {
        for (int i = dimsToUpdate.Length - 1; i >= 0; i--)
        {
            int dim = dimsToUpdate[i];
            index[dim]++;
            if (index[dim] < shape[dim])
                return true;

            index[dim] = 0;
        }
        return false;
    }

    /// <summary>
    /// Apply the softmax function along the given axis
    /// </summary>
    /// <param name="dim">axis to apply softmax along</param>
    /// <returns>tensor with the softmax function applied to the given axis</returns>
    public static Tensor<TNum> Softmax<TNum>(this Tensor<TNum> self, Index dim)
    where TNum : INumber<TNum>, IExponentialFunctions<TNum>
    {
        var shape = self.Shape;
        var rank = shape.Rank;
        var axis = NormalizeAxis(dim, shape);

        var input = self.AsSpan();
        var output = new TNum[input.Length];

        int classCount = shape.Length(axis);
        ReadOnlySpan<int> dimensions = shape.AsDimensionSpan();
        ReadOnlySpan<int> strides = shape.AsStrideSpan();

        // Determine which dimensions to iterate over (all but axis)
        int nonAxisDimCount = rank - 1;
        Span<int> dimsToUpdate = stackalloc int[nonAxisDimCount];
        {
            int j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (i != axis)
                    dimsToUpdate[j++] = i;
            }
        }

        Span<int> index = stackalloc int[rank];

        do
        {
            // Compute base offset for this slice
            int baseOffset = 0;
            for (int i = 0; i < rank; i++)
            {
                baseOffset += index[i] * strides[i];
            }

            // Compute max for numerical stability
            TNum max = input[baseOffset];
            for (int i = 1; i < classCount; i++)
            {
                int offset = baseOffset + i * strides[axis];
                TNum val = input[offset];
                if (val > max) max = val;
            }

            // Compute exponentials and sum
            TNum sum = TNum.Zero;
            for (int i = 0; i < classCount; i++)
            {
                int offset = baseOffset + i * strides[axis];
                TNum exp = TNum.Exp(input[offset] - max);
                output[offset] = exp;
                sum += exp;
            }

            // Normalize
            TNum invSum = TNum.One / sum;
            for (int i = 0; i < classCount; i++)
            {
                int offset = baseOffset + i * strides[axis];
                output[offset] *= invSum;
            }

        } while (IncrementIndex(index, dimensions, dimsToUpdate));

        return Tensor<TNum>.FromFlattenedArray(shape, output);
    }
}