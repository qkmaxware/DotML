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
/// Extension methods adding a Log operation to supported tensor types
/// </summary>
public static class TensorLog
{
    #region Log_{e}(x)
    /// <summary>
    /// Elementwise value base E logarithm
    /// </summary>
    /// <returns>tensor with each element passed to log</returns>
    public static Tensor<TNum> LogE<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ILogarithmicFunctions<TNum>
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
                vectorized_log(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_log(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Log(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value base E logarithm
    /// </summary>
    /// <returns>tensor with each element passed to log</returns>
    public static void LogEInplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ILogarithmicFunctions<TNum>
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
                vectorized_log(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_log(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Log(elements[i]);
        }
    }

    private static void vectorized_log(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            Vector.Log(va).CopyTo(dest, index);
        }
    }

    private static void vectorized_log(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            Vector.Log(va).CopyTo(dest, index);
        }
    }
    #endregion

    #region Log_{2}(x)
    /// <summary>
    /// Elementwise value base 2 logarithm
    /// </summary>
    /// <returns>tensor with each element passed to log</returns>
    public static Tensor<TNum> Log2<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ILogarithmicFunctions<TNum>
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
                vectorized_log2(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_log2(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Log2(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value base 2 logarithm
    /// </summary>
    /// <returns>tensor with each element passed to log</returns>
    public static void Log2Inplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ILogarithmicFunctions<TNum>
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
                vectorized_log2(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_log2(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Log2(elements[i]);
        }
    }

    private static void vectorized_log2(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            Vector.Log2(va).CopyTo(dest, index);
        }
    }

    private static void vectorized_log2(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            Vector.Log2(va).CopyTo(dest, index);
        }
    }
    #endregion

    #region Log_{10}(x)
    /// <summary>
    /// Elementwise value base 10 logarithm
    /// </summary>
    /// <returns>tensor with each element passed to log</returns>
    public static Tensor<TNum> Log10<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ILogarithmicFunctions<TNum>
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
                vectorized_log10(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_log10(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Log10(elements[i]);
        }
        return Tensor<TNum>.FromFlattenedArray(self.Shape, tensor);
    }
    /// <summary>
    /// Elementwise value base 10 logarithm
    /// </summary>
    /// <returns>tensor with each element passed to log</returns>
    public static void Log10Inplace<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, ILogarithmicFunctions<TNum>
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
                vectorized_log10(ref i, Unsafe.As<TNum[], float[]>(ref tensor), Unsafe.As<TNum[], float[]>(ref elements));
            }
            else if (typeof(TNum) == typeof(double))
            {
                vectorized_log10(ref i, Unsafe.As<TNum[], double[]>(ref tensor), Unsafe.As<TNum[], double[]>(ref elements));
            }
        }
        // Remaining elements
        for (; i < length; i++)
        {
            tensor[i] = TNum.Log10(elements[i]);
        }
    }
    private static void vectorized_log10(ref int index, float[] dest, float[] src)
    {
        var length = src.Length;
        int simdLength = Vector<float>.Count;
        int simdLimit = length - simdLength + 1;
        var log_e_10 = Vector.Log(new Vector<float>(10));

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<float>(src, index);
            // Change of base rule log_b(c) = log_x(c) / log_x(b)
            var log_e_c = Vector.Log(va);
            Vector.Divide(log_e_c, log_e_10).CopyTo(dest, index);
        }
    }

    private static void vectorized_log10(ref int index, double[] dest, double[] src)
    {
        var length = src.Length;
        int simdLength = Vector<double>.Count;
        int simdLimit = length - simdLength + 1;
        var log_e_10 = Vector.Log(new Vector<double>(10));

        for (; index < simdLimit; index += simdLength)
        {
            var va = new Vector<double>(src, index);
            // Change of base rule log_b(c) = log_x(c) / log_x(b)
            var log_e_c = Vector.Log(va);
            Vector.Divide(log_e_c, log_e_10).CopyTo(dest, index);
        }
    }
    #endregion
}