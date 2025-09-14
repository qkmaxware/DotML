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
/// Extension methods adding a Convert operation to supported tensor types
/// </summary>
public static class TensorConvert
{
    /// <summary>
    /// Convert elements to be of an arbitrary type
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <typeparam name="TNumTo">type to convert to</typeparam>
    /// <returns>tensor of elements of type <see cref="TNumTo"/></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TNumTo> ToType<TNumFrom, TNumTo>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
    where TNumTo : INumber<TNumTo>
        => self.ElementWise(x => (TNumTo)x.ToType(typeof(TNumTo), null));

    /// <summary>
    /// Convert elements to a floating point representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> ToFloat<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToSingle(null));

    /// <summary>
    /// Convert elements to a floating point representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<double> ToDouble<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToDouble(null));

    /// <summary>
    /// Convert elements to a floating point representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<decimal> ToDecimal<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToDecimal(null));

    /// <summary>
    /// Convert elements to a integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<SByte> ToInt8<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToSByte(null));

    /// <summary>
    /// Convert elements to an integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<Int16> ToInt16<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToInt16(null));

    /// <summary>
    /// Convert elements to a integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<Int32> ToInt32<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToInt32(null));

    /// <summary>
    /// Convert elements to a integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<Int64> ToInt64<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToInt64(null));

    /// <summary>
    /// Convert elements to a unsigned integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<Byte> ToUInt8<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToByte(null));

    /// <summary>
    /// Convert elements to an unsigned integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<UInt16> ToUInt16<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToUInt16(null));

    /// <summary>
    /// Convert elements to a unsigned integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<UInt32> ToUInt32<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToUInt32(null));

    /// <summary>
    /// Convert elements to a unsigned integer representation
    /// </summary>
    /// <typeparam name="TNumFrom">type to convert from</typeparam>
    /// <returns>tensor of floats</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<UInt64> ToUInt64<TNumFrom>(this Tensor<TNumFrom> self)
    where TNumFrom : INumber<TNumFrom>, IConvertible
        => self.ElementWise(x => x.ToUInt64(null));
}