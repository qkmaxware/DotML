using System.ComponentModel;
using System.Globalization;

namespace DotML;

/// <summary>
/// 16bit floating point type with 1 sign bit, 8 exponent bits, and 7 mantissa bits.
/// This is not a standard type, but is commonly used in machine learning applications.
/// </summary>
[TypeConverter(typeof(BFloat16Converter))]
public readonly struct BFloat16
: IConvertible
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

    public ulong ToUInt64(IFormatProvider? provider) {
        return Convert.ToUInt64(FloatValue);
    }

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