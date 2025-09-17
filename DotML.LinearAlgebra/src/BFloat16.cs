namespace DotML;

/// <summary>
/// 16bit floating point type with 1 sign bit, 8 exponent bits, and 7 mantissa bits.
/// This is not a standard type, but is commonly used in machine learning applications.
/// </summary>
public readonly struct BFloat16
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
    /// Convert a BFloat16 to a float by shifting the bits to the higher 16 bits.
    /// </summary>
    /// <param name="bf"></param>
    public static implicit operator float(BFloat16 bf)
    {
        uint intValue = (uint)(bf.value << 16);
        return BitConverter.ToSingle(BitConverter.GetBytes(intValue), 0);
    }

    public override string ToString() => ((float)this).ToString();
}