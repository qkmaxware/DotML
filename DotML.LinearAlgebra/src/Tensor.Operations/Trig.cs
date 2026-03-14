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
/// Extension methods adding Trigonometric operations to supported tensor types
/// </summary>
public static class TensorTrig
{
    #region SIN
    /// <summary>
    /// Elementwise value sin
    /// </summary>
    /// <returns>tensor with the sin of each element</returns>
    public static Tensor<TNum> Sin<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ITrigonometricFunctions<TNum>
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
                vectorized_sin(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_sin(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Sin(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value sin in-place
    /// </summary>
    /// <returns>tensor with the sin of each element</returns>
    public static void SinInplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ITrigonometricFunctions<TNum>
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
                vectorized_sin(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_sin(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Sin(elements[i]);
        }
    }
    private static void vectorized_sin(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            Vector.Sin(va).CopyTo(dest, index);
        }
    }
    private static void vectorized_sin(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            Vector.Sin(va).CopyTo(dest, index);
        }
    }
    #endregion

    #region COS
    /// <summary>
    /// Elementwise value cos
    /// </summary>
    /// <returns>tensor with the cos of each element</returns>
    public static Tensor<TNum> Cos<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ITrigonometricFunctions<TNum>
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
                vectorized_cos(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_cos(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Cos(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value cos in-place
    /// </summary>
    /// <returns>tensor with the cos of each element</returns>
    public static void CosInplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ITrigonometricFunctions<TNum>
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
                vectorized_cos(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_cos(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Cos(elements[i]);
        }
    }
    private static void vectorized_cos(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            Vector.Cos(va).CopyTo(dest, index);
        }
    }
    private static void vectorized_cos(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            Vector.Cos(va).CopyTo(dest, index);
        }
    }
    #endregion

    #region TAN
    /// <summary>
    /// Elementwise value tan
    /// </summary>
    /// <returns>tensor with the tan of each element</returns>
    public static Tensor<TNum> Tan<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ITrigonometricFunctions<TNum>
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
                vectorized_tan(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_tan(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Tan(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value tan in-place
    /// </summary>
    /// <returns>tensor with the tan of each element</returns>
    public static void TanInplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ITrigonometricFunctions<TNum>
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
                vectorized_tan(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_tan(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Tan(elements[i]);
        }
    }
    private static void vectorized_tan(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            var num = Vector.Sin(va);
            var den = Vector.Cos(va);
            Vector.Divide(num, den).CopyTo(dest, index);
        }
    }
    private static void vectorized_tan(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            var num = Vector.Sin(va);
            var den = Vector.Cos(va);
            Vector.Divide(num, den).CopyTo(dest, index);
        }
    }
    #endregion
}