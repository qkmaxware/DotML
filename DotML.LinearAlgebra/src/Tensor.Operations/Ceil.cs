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
/// Extension methods adding a Ceiling operation to supported tensor types
/// </summary>
public static class TensorCeiling
{
    /// <summary>
    /// Elementwise value ceiling
    /// </summary>
    /// <returns>tensor with each element ceil'd</returns>
    public static Tensor<TNum> Ceiling<TNum>(this Tensor<TNum> self)
    where TNum : IFloatingPoint<TNum>
    {
        var elements = self.AsArray();
        var length = elements.Length;
        var tensor = new TNum[length];

        // Vectorized elements
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            if (typeof(TNum) == typeof(float))
            {
                vectorized_ceil(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_ceil(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Ceiling(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value ceiling in-place
    /// </summary>
    /// <returns>tensor with each element ceil'd</returns>
    public static void CeilingInplace<TNum>(this Tensor<TNum> self)
    where TNum : IFloatingPoint<TNum>
    {
        var elements = self.AsArray();
        var length = elements.Length;
        var tensor = elements;

        // Vectorized elements
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<TNum>.IsSupported)
        {
            if (typeof(TNum) == typeof(float))
            {
                vectorized_ceil(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_ceil(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Ceiling(elements[i]);
        }
    }

    private static void vectorized_ceil(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            Vector.Ceiling(va).CopyTo(dest, index);
        }
    }

    private static void vectorized_ceil(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            Vector.Ceiling(va).CopyTo(dest, index);
        }
    }
}