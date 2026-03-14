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
public static class TensorStandardDeviation
{
    /// <summary>
    /// Compute the standard deviation along the given axis
    /// </summary>
    /// <param name="axis">axis</param>
    /// <param name="ddof">degrees of freedom</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>standard deviation tensor</returns>
    public static Tensor<TNum> Std<TNum>(this Tensor<TNum> self, Index axis, int ddof = 0, bool keepdim = true)
    where TNum: INumber<TNum>, IRootFunctions<TNum>
    {
        var variance = self.Variance(axis, ddof, keepdim);
        variance.SqrtInplace();
        return variance;
    }
    
    /// <summary>
    /// Compute the standard deviation along the given axis
    /// </summary>
    /// <param name="axis">axis</param>
    /// <param name="ddof">degrees of freedom</param>
    /// <param name="keepdim">flag to indicate if the reduced dimension is to be kept (at size 1) or removed. Default true</param>
    /// <returns>standard deviation tensor</returns>
    public static Tensor<TNum> Std<TNum>(this Tensor<TNum> self, Index axis, out Tensor<TNum> variance, int ddof = 0, bool keepdim = true)
    where TNum: INumber<TNum>, IRootFunctions<TNum>
    {
        variance = self.Variance(axis, ddof, keepdim);
        return variance.Sqrt();
    }
    
    /// <summary>
    /// Compute the standard deviation of all elements in the tensor
    /// </summary>
    /// <param name="ddof">degrees of freedom</param>
    /// <returns>standard deviation</returns>
    public static TNum Std<TNum>(this Tensor<TNum> self, int ddof = 0)
    where TNum : INumber<TNum>, IRootFunctions<TNum>
    {
        var variance = self.Variance(ddof);
        TNum std = TNum.Sqrt(variance);
        return std;
    }

    /// <summary>
    /// Compute the standard deviation of all elements in the tensor
    /// </summary>
    /// <param name="variance">computed variance</param>
    /// <param name="ddof">degrees of freedom</param>
    /// <returns>standard deviation</returns>
	public static TNum Std<TNum>(this Tensor<TNum> self, out TNum variance, int ddof = 0)
    where TNum : INumber<TNum>, IRootFunctions<TNum>
    {
        variance = self.Variance(ddof);
        TNum std = TNum.Sqrt(variance);
        return std;
    }
}