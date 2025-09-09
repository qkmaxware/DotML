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
/// Extension methods adding a Min and Max operation to supported tensor types
/// </summary>
public static class TensorMinMax
{
    /// <summary>
    /// Compute the minimum along the specified axis.
    /// </summary>
    /// <param name="axis">The axis to reduce along.</param>
    /// <param name="keepdim">Whether to keep the reduced dimension as size 1. Default is true.</param>
    /// <returns>A tensor with the minimum values along the specified axis.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> Min<TNum>(this Tensor<TNum> self, Index axis, bool keepdim = true)
    where TNum : INumber<TNum>, IMinMaxValue<TNum>
    => self.Reduce(axis, TNum.MaxValue, static (acc, val) => TNum.Min(acc, val), keepdim);

    /// <summary>
    /// Compute the global minimum value of the tensor.
    /// </summary>
    /// <returns>maximum value in the tensor or TNum.MaxValue if the tensor is empty</returns>
    public static TNum Min<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IMinMaxValue<TNum>
    {
        var min = TNum.MaxValue;
        foreach (var v in self.AsSpan())
            if (v < min) min = v;
        return min;
    }

    /// <summary>
    /// Compute the maximum along the specified axis.
    /// </summary>
    /// <param name="axis">The axis to reduce along.</param>
    /// <param name="keepdim">Whether to keep the reduced dimension as size 1. Default is true.</param>
    /// <returns>A tensor with the maximum values along the specified axis.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNum> Max<TNum>(this Tensor<TNum> self, Index axis, bool keepdim = true)
    where TNum : INumber<TNum>, IMinMaxValue<TNum>
    => self.Reduce(axis, TNum.MinValue, static (acc, val) => TNum.Max(acc, val), keepdim);

     /// <summary>
    /// Compute the global maximum value of the tensor.
    /// </summary>
    /// <returns>maximum value in the tensor or TNum.MinValue if the tensor is empty</returns>
    public static TNum Max<TNum>(this Tensor<TNum> self)
    where TNum : INumber<TNum>, IMinMaxValue<TNum>
    {
        var max = TNum.MinValue;
        foreach (var v in self.AsSpan())
            if (v > max) max = v;
        return max;
    }
}