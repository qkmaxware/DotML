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
/// Extension methods adding a square root operation to supported tensor types
/// </summary>
public static class TensorSqrt
{
    /// <summary>
    /// Elementwise value square root
    /// </summary>
    /// <returns>tensor with each element square rooted</returns>
    public static Tensor<TNum> Sqrt<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IRootFunctions<TNum>
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
                vectorized_sqrt(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_sqrt(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Sqrt(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value square root in-place
    /// </summary>
    /// <returns>tensor with each element square rooted</returns>
    public static void FloorInplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IRootFunctions<TNum>
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
                vectorized_sqrt(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_sqrt(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Sqrt(elements[i]);
        }
    }

    private static void vectorized_sqrt(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            Vector.SquareRoot(va).CopyTo(dest, index);
        }
    }

    private static void vectorized_sqrt(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            Vector.SquareRoot(va).CopyTo(dest, index);
        }
    }
}