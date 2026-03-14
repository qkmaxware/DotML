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
/// Extension methods adding exponential operations to supported tensor types
/// </summary>
public static class TensorExp
{
    #region E^x
    /// <summary>
    /// Elementwise exponential value
    /// </summary>
    /// <returns>tensor with each element applied to exp</returns>
    public static Tensor<TNum> Exp<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IExponentialFunctions<TNum>
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
                vectorized_exp(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_exp(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Exp(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise exponential value in-place
    /// </summary>
    /// <returns>tensor with each element applied to exp</returns>
    public static void ExpInplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IExponentialFunctions<TNum>
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
                vectorized_exp(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_exp(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Exp(elements[i]);
        }
    }
    private static void vectorized_exp(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            Vector.Exp(va).CopyTo(dest, index);
        }
    }
    private static void vectorized_exp(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            Vector.Exp(va).CopyTo(dest, index);
        }
    }
    #endregion

    #region 10^x
    /// <summary>
    /// Elementwise 10 to the power of value
    /// </summary>
    /// <returns>tensor with each element applied to exp</returns>
    public static Tensor<TNum> Exp10<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IExponentialFunctions<TNum>
    {
        var elements = self.AsArray();
        var length = elements.Length;
        var tensor = new TNum[length];

        // Vectorized elements
        int i = 0;
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Exp10(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise  10 to the power of value in-place
    /// </summary>
    /// <returns>tensor with each element applied to exp</returns>
    public static void Exp10Inplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IExponentialFunctions<TNum>
    {
        var elements = self.AsArray();
        var length = elements.Length;
        var tensor = elements;

        // Vectorized elements
        int i = 0;
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Exp10(elements[i]);
        }
    }
    #endregion
}