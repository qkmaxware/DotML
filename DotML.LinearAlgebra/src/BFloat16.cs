using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;

namespace DotML;

/// <summary>
/// 16bit floating point type with 1 sign bit, 8 exponent bits, and 7 mantissa bits.
/// This is not a standard type, but is commonly used in machine learning applications.
/// </summary>
[TypeConverter(typeof(BFloat16Converter))]
public readonly struct BFloat16
: IConvertible,
INumber<BFloat16>,
IMinMaxValue<BFloat16>,
ITrigonometricFunctions<BFloat16>,
IRootFunctions<BFloat16>,
IExponentialFunctions<BFloat16>,
ILogarithmicFunctions<BFloat16>,
IFloatingPoint<BFloat16>
{
    private readonly ushort value;

    public ushort RawValue => value;

    public float FloatValue => (float)this;

    /// <summary>
    /// Construct a BFloat16 from from a 16bit pattern.
    /// </summary>
    /// <param name="u">bit pattern</param>
    public BFloat16(ushort u)
    {
        value = u;
    }

    /// <summary>
    /// Construct a BFloat16 from a float by truncating the lower 16 bits.
    /// </summary>
    /// <param name="f"></param>
    public BFloat16(float f)
    {
        uint intValue = BitConverter.ToUInt32(BitConverter.GetBytes(f), 0);
        value = (ushort)(intValue >> 16);
    }

    /// <summary>
    /// Convert a BFloat16 to a float
    /// </summary>
    /// <param name="bf"></param>
    public static implicit operator float(BFloat16 bf)
    {
        uint intValue = (uint)(bf.value << 16);
        return BitConverter.ToSingle(BitConverter.GetBytes(intValue), 0);
    }

    /// <summary>
    /// Convert a float to a BFloat16
    /// </summary>
    /// <param name="bf"></param>
    public static explicit operator BFloat16(float f)
    {
        return new BFloat16(f);
    }

    #region IConvertible
    public TypeCode GetTypeCode() => TypeCode.Object;

    public bool ToBoolean(IFormatProvider? provider) {
        return FloatValue == 0.0f;
    }

    public byte ToByte(IFormatProvider? provider) {
        return Convert.ToByte(FloatValue);
    }

    public sbyte ToSByte(IFormatProvider? provider) {
        return Convert.ToSByte(FloatValue);
    }

    public char ToChar(IFormatProvider? provider) {
        return Convert.ToChar(FloatValue);
    }

    public DateTime ToDateTime(IFormatProvider? provider) {
        return Convert.ToDateTime(FloatValue);
    }

    public decimal ToDecimal(IFormatProvider? provider) {
        return Convert.ToDecimal(FloatValue);
    }

    public double ToDouble(IFormatProvider? provider) {
        return Convert.ToDouble(FloatValue);
    }

    public short ToInt16(IFormatProvider? provider) {
        return Convert.ToInt16(FloatValue);
    }

    public int ToInt32(IFormatProvider? provider) {
        return Convert.ToInt32(FloatValue);
    }

    public long ToInt64(IFormatProvider? provider) {
        return Convert.ToInt64(FloatValue);
    }

    public float ToSingle(IFormatProvider? provider) {
        return Convert.ToSingle(FloatValue);
    }

    public string ToString(IFormatProvider? provider) {
        return Convert.ToString(FloatValue);
    }

    public object ToType(Type conversionType, IFormatProvider? provider) {
        return Convert.ChangeType(FloatValue, conversionType);
    }

    public ushort ToUInt16(IFormatProvider? provider) {
        return Convert.ToUInt16(FloatValue);
    }

    public uint ToUInt32(IFormatProvider? provider) {
        return Convert.ToUInt32(FloatValue);
    }

    public ulong ToUInt64(IFormatProvider? provider)
    {
        return Convert.ToUInt64(FloatValue);
    }
    #endregion

    #region INumber<BFloat16>
    public static BFloat16 One { get; } = new BFloat16(1.0f);

    public static int Radix => 10;

    public static BFloat16 Zero { get; } = new BFloat16(0.0f);

    public static BFloat16 AdditiveIdentity => Zero;

    public static BFloat16 MultiplicativeIdentity => One;

    public static BFloat16 MaxValue { get; } = new BFloat16(0x7F7F); // Largest finite positive value
    public static BFloat16 MinValue { get; } = new BFloat16(0xFF7F); // Largest finite negative value

    public static BFloat16 NegativeInfinity { get; } = new BFloat16(0xFF80);
    public static BFloat16 PositiveInfinity { get; } = new BFloat16(0x7F80);
    public static BFloat16 NaN { get; } = new BFloat16(0x7FC0); // Canonical quiet NaN

    public static BFloat16 E => throw new NotImplementedException();

    public static BFloat16 Pi => throw new NotImplementedException();

    public static BFloat16 Tau => throw new NotImplementedException();

    public static BFloat16 NegativeOne => throw new NotImplementedException();

    public int CompareTo(object? obj)
    {
        if (obj is null) return 1;
        if (obj is BFloat16 other)
            return CompareTo(other);
        throw new ArgumentException("Object must be of type BFloat16");
    }

    public int CompareTo(BFloat16 other)
    {
        return FloatValue.CompareTo(other.FloatValue);
    }

    public static BFloat16 Abs(BFloat16 value)
    {
        return new BFloat16(MathF.Abs(value.FloatValue));
    }

    public static bool IsCanonical(BFloat16 value)
    {
        return true;
    }

    public static bool IsComplexNumber(BFloat16 value)
    {
        return false;
    }

    public static bool IsEvenInteger(BFloat16 value)
    {
        float f = value.FloatValue;
        return IsInteger(value) && ((int)f % 2 == 0);
    }

    public static bool IsFinite(BFloat16 value)
    {
        return float.IsFinite(value.FloatValue);
    }

    public static bool IsImaginaryNumber(BFloat16 value)
    {
        return false;
    }

    public static bool IsInfinity(BFloat16 value)
    {
        return float.IsInfinity(value.FloatValue);
    }

    public static bool IsInteger(BFloat16 value)
    {
        float f = value.FloatValue;
        return float.IsFinite(f) && f == MathF.Floor(f);
    }

    public static bool IsNaN(BFloat16 value)
    {
        return float.IsNaN(value.FloatValue);
    }

    public static bool IsNegative(BFloat16 value)
    {
        return (value.RawValue & 0x8000) != 0;
    }

    public static bool IsNegativeInfinity(BFloat16 value)
    {
        return float.IsNegativeInfinity(value.FloatValue);
    }

    public static bool IsNormal(BFloat16 value)
    {
        ushort exponent = (ushort)((value.RawValue >> 7) & 0xFF); // bits 8..15
        return exponent != 0 && exponent != 0xFF;
    }

    public static bool IsOddInteger(BFloat16 value)
    {
        float f = value.FloatValue;
        return IsInteger(value) && ((int)f % 2 != 0);
    }

    public static bool IsPositive(BFloat16 value)
    {
        return (value.RawValue & 0x8000) == 0 && !IsNaN(value);
    }

    public static bool IsPositiveInfinity(BFloat16 value)
    {
        return value.RawValue == 0x7F80;
    }

    public static bool IsRealNumber(BFloat16 value)
    {
        return !IsNaN(value);
    }

    public static bool IsSubnormal(BFloat16 value)
    {
        ushort exponent = (ushort)((value.RawValue >> 7) & 0xFF);
        ushort fraction = (ushort)(value.RawValue & 0x7F);

        return exponent == 0 && fraction != 0;
    }

    public static bool IsZero(BFloat16 value)
    {
        return (value.RawValue & 0x7FFF) == 0;
    }

    public static BFloat16 MaxMagnitude(BFloat16 x, BFloat16 y)
    {
        float fx = x.FloatValue;
        float fy = y.FloatValue;

        float absX = MathF.Abs(fx);
        float absY = MathF.Abs(fy);

        if (absX > absY) return x;
        if (absY > absX) return y;

        // If magnitudes are equal, prefer non-NaN
        return float.IsNaN(fx) ? y : x;
    }

    public static BFloat16 MaxMagnitudeNumber(BFloat16 x, BFloat16 y)
    {
        bool xIsNaN = IsNaN(x);
        bool yIsNaN = IsNaN(y);

        if (xIsNaN && yIsNaN) return x; // convention: return x
        if (xIsNaN) return y;
        if (yIsNaN) return x;

        return MaxMagnitude(x, y);
    }

    public static BFloat16 MinMagnitude(BFloat16 x, BFloat16 y)
    {
        float fx = x.FloatValue;
        float fy = y.FloatValue;

        float absX = MathF.Abs(fx);
        float absY = MathF.Abs(fy);

        if (absX < absY) return x;
        if (absY < absX) return y;

        return float.IsNaN(fx) ? y : x;
    }

    public static BFloat16 MinMagnitudeNumber(BFloat16 x, BFloat16 y)
    {
        bool xIsNaN = IsNaN(x);
        bool yIsNaN = IsNaN(y);

        if (xIsNaN && yIsNaN) return x;
        if (xIsNaN) return y;
        if (yIsNaN) return x;

        return MinMagnitude(x, y);
    }

    public static BFloat16 Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider)
    {
        return new BFloat16(float.Parse(s, style, provider));
    }

    public static BFloat16 Parse(string s, NumberStyles style, IFormatProvider? provider)
    {
        return new BFloat16(float.Parse(s, style, provider));
    }

    public static bool TryConvertFromChecked<TOther>(TOther value, [MaybeNullWhen(false)] out BFloat16 result) where TOther : INumberBase<TOther>
    {
        try
        {
            float f = float.CreateChecked(value);
            result = new BFloat16(f);
            return true;
        }
        catch
        {
            result = default;
            return false;
        }
    }

    public static bool TryConvertFromSaturating<TOther>(TOther value, [MaybeNullWhen(false)] out BFloat16 result) where TOther : INumberBase<TOther>
    {
        try
        {
            float f = float.CreateSaturating(value);
            result = new BFloat16(f);
            return true;
        }
        catch
        {
            result = default;
            return false;
        }
    }

    public static bool TryConvertFromTruncating<TOther>(TOther value, [MaybeNullWhen(false)] out BFloat16 result) where TOther : INumberBase<TOther>
    {
        try
        {
            float f = float.CreateTruncating(value);
            result = new BFloat16(f);
            return true;
        }
        catch
        {
            result = default;
            return false;
        }
    }

    public static bool TryConvertToChecked<TOther>(BFloat16 value, [MaybeNullWhen(false)] out TOther result) where TOther : INumberBase<TOther>
    {
        try
        {
            result = TOther.CreateChecked(value.FloatValue);
            return true;
        }
        catch
        {
            result = default!;
            return false;
        }
    }

    public static bool TryConvertToSaturating<TOther>(BFloat16 value, [MaybeNullWhen(false)] out TOther result) where TOther : INumberBase<TOther>
    {
        try
        {
            result = TOther.CreateSaturating(value.FloatValue);
            return true;
        }
        catch
        {
            result = default!;
            return false;
        }
    }

    public static bool TryConvertToTruncating<TOther>(BFloat16 value, [MaybeNullWhen(false)] out TOther result) where TOther : INumberBase<TOther>
    {
        try
        {
            result = TOther.CreateTruncating(value.FloatValue);
            return true;
        }
        catch
        {
            result = default!;
            return false;
        }
    }

    public static bool TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result)
    {
        result = BFloat16.Zero;
        if (float.TryParse(s, style, provider, out float val))
        {
            result = new BFloat16(val);
            return true;
        }
        else
            return false;
    }

    public static bool TryParse([NotNullWhen(true)] string? s, NumberStyles style, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result)
    {
        result = BFloat16.Zero;
        if (float.TryParse(s, style, provider, out float val))
        {
            result = new BFloat16(val);
            return true;
        }
        else
            return false;
    }

    public bool Equals(BFloat16 other)
    {
        return this.value == other.value;
    }

    public bool TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider)
    {
        return FloatValue.TryFormat(destination, out charsWritten, format, provider);
    }

    public string ToString(string? format, IFormatProvider? formatProvider)
    {
        return FloatValue.ToString(format, formatProvider);
    }

    public static BFloat16 Parse(ReadOnlySpan<char> s, IFormatProvider? provider)
    {
        return new BFloat16(float.Parse(s, provider));
    }

    public static bool TryParse(ReadOnlySpan<char> s, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result)
    {
        result = BFloat16.Zero;
        if (float.TryParse(s, provider, out float val))
        {
            result = new BFloat16(val);
            return true;
        }
        else
            return false;
    }

    public static BFloat16 Parse(string s, IFormatProvider? provider)
    {
        return new BFloat16(float.Parse(s, provider));
    }

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result)
    {
        result = BFloat16.Zero;
        if (float.TryParse(s, provider, out float val))
        {
            result = new BFloat16(val);
            return true;
        }
        else
            return false;
    }

    public static BFloat16 Acos(BFloat16 x) => new BFloat16(float.Acos(x.FloatValue));

    public static BFloat16 AcosPi(BFloat16 x) => new BFloat16(float.AcosPi(x.FloatValue));

    public static BFloat16 Asin(BFloat16 x) => new BFloat16(float.Asin(x.FloatValue));

    public static BFloat16 AsinPi(BFloat16 x) => new BFloat16(float.AsinPi(x.FloatValue));

    public static BFloat16 Atan(BFloat16 x) => new BFloat16(float.Atan(x.FloatValue));

    public static BFloat16 AtanPi(BFloat16 x) => new BFloat16(float.AtanPi(x.FloatValue));

    public static BFloat16 Cos(BFloat16 x) => new BFloat16(float.Cos(x.FloatValue));

    public static BFloat16 CosPi(BFloat16 x) => new BFloat16(float.CosPi(x.FloatValue));

    public static BFloat16 Sin(BFloat16 x) => new BFloat16(float.Sin(x.FloatValue));

    public static (BFloat16 Sin, BFloat16 Cos) SinCos(BFloat16 x)
    {
        var sincos = float.SinCos(x.FloatValue);
        return (new BFloat16(sincos.Sin), new BFloat16(sincos.Cos));
    }

    public static (BFloat16 SinPi, BFloat16 CosPi) SinCosPi(BFloat16 x)
    {
        var sincos = float.SinCosPi(x.FloatValue);
        return (new BFloat16(sincos.SinPi), new BFloat16(sincos.CosPi));
    }

    public static BFloat16 SinPi(BFloat16 x) => new BFloat16(float.SinPi(x.FloatValue));

    public static BFloat16 Tan(BFloat16 x) => new BFloat16(float.Tan(x.FloatValue));

    public static BFloat16 TanPi(BFloat16 x) => new BFloat16(float.TanPi(x.FloatValue));

    public static BFloat16 Cbrt(BFloat16 x) => new BFloat16(float.Cbrt(x.FloatValue));

    public static BFloat16 Hypot(BFloat16 x, BFloat16 y) => new BFloat16(float.Hypot(x.FloatValue, y.FloatValue));

    public static BFloat16 RootN(BFloat16 x, int n) => new BFloat16(float.Hypot(x.FloatValue, n));

    public static BFloat16 Sqrt(BFloat16 x) => new BFloat16(float.Sqrt(x.FloatValue));

    public static BFloat16 Exp(BFloat16 x) => new BFloat16(float.Exp(x.FloatValue));

    public static BFloat16 Exp10(BFloat16 x) => new BFloat16(float.Exp10(x.FloatValue));

    public static BFloat16 Exp2(BFloat16 x) => new BFloat16(float.Exp2(x.FloatValue));

    public static BFloat16 Log(BFloat16 x)=> new BFloat16(float.Log(x.FloatValue));

    public static BFloat16 Log(BFloat16 x, BFloat16 newBase) => new BFloat16(float.Log(x.FloatValue, newBase.FloatValue));

    public static BFloat16 Log10(BFloat16 x) => new BFloat16(float.Log10(x.FloatValue));

    public static BFloat16 Log2(BFloat16 x)=> new BFloat16(float.Log2(x.FloatValue));

    public int GetExponentByteCount() => 1;

    public int GetExponentShortestBitLength()
    {
        // Extract exponent (bits 14-7)
        byte exponent = (byte)((value >> 7) & 0xFF);
        int bitLength = 8;

        while (bitLength > 0 && (exponent & (1 << (bitLength - 1))) == 0)
            bitLength--;

        return bitLength;
    }

    public int GetSignificandBitLength() => 7;

    public int GetSignificandByteCount() => 1;

    public static BFloat16 Round(BFloat16 x, int digits, MidpointRounding mode)
    {
        float rounded = MathF.Round(x.FloatValue, digits, mode);
        return new BFloat16(rounded);
    }

    public bool TryWriteExponentBigEndian(Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length < 1)
        {
            bytesWritten = 0;
            return false;
        }

        byte exponent = (byte)((value >> 7) & 0xFF);
        destination[0] = exponent;
        bytesWritten = 1;
        return true;
    }

    public bool TryWriteExponentLittleEndian(Span<byte> destination, out int bytesWritten)
    {
        return TryWriteExponentBigEndian(destination, out bytesWritten);
    }

    public bool TryWriteSignificandBigEndian(Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length < 1)
        {
            bytesWritten = 0;
            return false;
        }

        byte significand = (byte)(value & 0x7F); // bits 6–0
        destination[0] = significand;
        bytesWritten = 1;
        return true;
    }

    public bool TryWriteSignificandLittleEndian(Span<byte> destination, out int bytesWritten)
    {
        // Same as big endian since it's only 1 byte
        return TryWriteSignificandBigEndian(destination, out bytesWritten);
    }

    public static bool operator >(BFloat16 left, BFloat16 right)
    {
        return left.FloatValue > right.FloatValue;
    }

    public static bool operator >=(BFloat16 left, BFloat16 right)
    {
        return left.FloatValue >= right.FloatValue;
    }

    public static bool operator <(BFloat16 left, BFloat16 right)
    {
        return left.FloatValue < right.FloatValue;
    }

    public static bool operator <=(BFloat16 left, BFloat16 right)
    {
        return left.FloatValue <= right.FloatValue;
    }

    public static BFloat16 operator %(BFloat16 left, BFloat16 right)
    {
        return new BFloat16(left.FloatValue % right.FloatValue);
    }

    public static BFloat16 operator +(BFloat16 left, BFloat16 right)
    {
        return new BFloat16(left.FloatValue + right.FloatValue);
    }

    public static BFloat16 operator --(BFloat16 value)
    {
        return new BFloat16(value.FloatValue - 1.0f);
    }

    public static BFloat16 operator /(BFloat16 left, BFloat16 right)
    {
        return new BFloat16(left.FloatValue / right.FloatValue);
    }

    public static bool operator ==(BFloat16 left, BFloat16 right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(BFloat16 left, BFloat16 right)
    {
        return !left.Equals(right);
    }

    public static BFloat16 operator ++(BFloat16 value)
    {
        return new BFloat16(value.FloatValue + 1.0f);
    }

    public static BFloat16 operator *(BFloat16 left, BFloat16 right)
    {
        return new BFloat16(left.FloatValue * right.FloatValue);
    }

    public static BFloat16 operator -(BFloat16 left, BFloat16 right)
    {
        return new BFloat16(left.FloatValue - right.FloatValue);
    }

    public static BFloat16 operator -(BFloat16 value)
    {
        return new BFloat16(-value.FloatValue);
    }

    public static BFloat16 operator +(BFloat16 value)
    {
        return value;
    }
    #endregion

    public override bool Equals (object? other)
    {
        if (other is BFloat16 bfloatOther)
            return this.Equals(bfloatOther);
        return false;
    }

    public override int GetHashCode() => this.value.GetHashCode();

    public override string ToString() => ((float)this).ToString();
}

/// <summary>
/// A type converter for the BFloat16 type
/// </summary>
public class BFloat16Converter: TypeConverter {
    private TypeConverter floatConverter;

    public BFloat16Converter() {
        floatConverter = TypeDescriptor.GetConverter(typeof(float));
    }

    public override bool CanConvertFrom(ITypeDescriptorContext? context, Type sourceType) {
        return floatConverter.CanConvertFrom(context, sourceType); // If we can convert from float we can convert from bfloat16
    }

    public override object ConvertFrom(ITypeDescriptorContext? context, CultureInfo? culture, object value) {
        return new BFloat16(floatConverter.ConvertFrom(context, culture, value) as float? ?? 0.0f); // Convert from type to float, then to bfloat16
    }

    public override bool CanConvertTo(ITypeDescriptorContext? context, Type? targetType) {
        return floatConverter.CanConvertTo(context, targetType);
    }

    public override object? ConvertTo(ITypeDescriptorContext? context, CultureInfo? culture, object? value, Type targetType) {
        return floatConverter.ConvertTo(context, culture, ((BFloat16)(value ?? new BFloat16(0f))).FloatValue, targetType);
    }
}