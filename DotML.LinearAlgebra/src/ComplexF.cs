using System.Diagnostics.CodeAnalysis;

namespace System.Numerics;

/// <summary>
/// A complex number z is a number of the form z = x + yi, where x and y
/// are real numbers, and i is the imaginary unit, with the property i2= -1.
/// This version uses floats to represent x and y as opposed to Complex which uses doubles.
/// </summary>
[Serializable]
public readonly struct ComplexF : IEquatable<ComplexF> {

    private readonly float m_real; // Do not rename (binary serialization)
    private readonly float m_imaginary; // Do not rename (binary serialization)

    public static readonly ComplexF Zero = new ComplexF(0.0f, 0.0f);
    public static readonly ComplexF One = new ComplexF(1.0f, 0.0f);
    public static readonly ComplexF ImaginaryOne = new ComplexF(0.0f, 1.0f);
    public static readonly ComplexF NaN = new ComplexF(float.NaN, float.NaN);
    public static readonly ComplexF Infinity = new ComplexF(float.PositiveInfinity, float.PositiveInfinity);

    public float Real { get { return m_real; } }
    public float Imaginary { get { return m_imaginary; } }

    public float Magnitude { get { return MathF.Sqrt(m_real * m_real + m_imaginary * m_imaginary); } }
    public float Phase { get { return MathF.Atan2(m_imaginary, m_real); } }

    public ComplexF(float real, float imaginary) {
        m_real = real;
        m_imaginary = imaginary;
    }

    public static ComplexF FromPolarCoordinates(float magnitude, float phase) {
        return new ComplexF(magnitude * MathF.Cos(phase), magnitude * MathF.Sin(phase));
    }

    public static ComplexF operator -(ComplexF value)  /* Unary negation of a ComplexF number */
    {
        return new ComplexF(-value.m_real, -value.m_imaginary);
    }

    public static ComplexF operator +(ComplexF left, ComplexF right)
    {
        return new ComplexF(left.m_real + right.m_real, left.m_imaginary + right.m_imaginary);
    }

    public static ComplexF operator +(ComplexF left, float right)
    {
        return new ComplexF(left.m_real + right, left.m_imaginary);
    }

    public static ComplexF operator +(float left, ComplexF right)
    {
        return new ComplexF(left + right.m_real, right.m_imaginary);
    }

    public static ComplexF operator -(ComplexF left, ComplexF right)
    {
        return new ComplexF(left.m_real - right.m_real, left.m_imaginary - right.m_imaginary);
    }

    public static ComplexF operator -(ComplexF left, float right)
    {
        return new ComplexF(left.m_real - right, left.m_imaginary);
    }

    public static ComplexF operator -(float left, ComplexF right)
    {
        return new ComplexF(left - right.m_real, -right.m_imaginary);
    }

    public static ComplexF operator *(ComplexF left, ComplexF right)
    {
        // Multiplication:  (a + bi)(c + di) = (ac -bd) + (bc + ad)i
        float result_realpart = (left.m_real * right.m_real) - (left.m_imaginary * right.m_imaginary);
        float result_imaginarypart = (left.m_imaginary * right.m_real) + (left.m_real * right.m_imaginary);
        return new ComplexF(result_realpart, result_imaginarypart);
    }

    public static ComplexF operator *(ComplexF left, float right)
    {
        if (!float.IsFinite(left.m_real))
        {
            if (!float.IsFinite(left.m_imaginary))
            {
                return new ComplexF(float.NaN, float.NaN);
            }

            return new ComplexF(left.m_real * right, float.NaN);
        }

        if (!float.IsFinite(left.m_imaginary))
        {
            return new ComplexF(float.NaN, left.m_imaginary * right);
        }

        return new ComplexF(left.m_real * right, left.m_imaginary * right);
    }

    public static ComplexF operator *(float left, ComplexF right)
    {
        if (!float.IsFinite(right.m_real))
        {
            if (!float.IsFinite(right.m_imaginary))
            {
                return new ComplexF(float.NaN, float.NaN);
            }

            return new ComplexF(left * right.m_real, float.NaN);
        }

        if (!float.IsFinite(right.m_imaginary))
        {
            return new ComplexF(float.NaN, left * right.m_imaginary);
        }

        return new ComplexF(left * right.m_real, left * right.m_imaginary);
    }

    public static ComplexF operator /(ComplexF left, ComplexF right)
    {
        // Division : Smith's formula.
        float a = left.m_real;
        float b = left.m_imaginary;
        float c = right.m_real;
        float d = right.m_imaginary;

        // Computing c * c + d * d will overflow even in cases where the actual result of the division does not overflow.
        if (Math.Abs(d) < Math.Abs(c))
        {
            float doc = d / c;
            return new ComplexF((a + b * doc) / (c + d * doc), (b - a * doc) / (c + d * doc));
        }
        else
        {
            float cod = c / d;
            return new ComplexF((b + a * cod) / (d + c * cod), (-a + b * cod) / (d + c * cod));
        }
    }

    public static ComplexF operator /(ComplexF left, float right)
    {
        // IEEE prohibit optimizations which are value changing
        // so we make sure that behaviour for the simplified version exactly match
        // full version.
        if (right == 0)
        {
            return new ComplexF(float.NaN, float.NaN);
        }

        if (!float.IsFinite(left.m_real))
        {
            if (!float.IsFinite(left.m_imaginary))
            {
                return new ComplexF(float.NaN, float.NaN);
            }

            return new ComplexF(left.m_real / right, float.NaN);
        }

        if (!float.IsFinite(left.m_imaginary))
        {
            return new ComplexF(float.NaN, left.m_imaginary / right);
        }

        // Here the actual optimized version of code.
        return new ComplexF(left.m_real / right, left.m_imaginary / right);
    }

    public static ComplexF operator /(float left, ComplexF right)
    {
        // Division : Smith's formula.
        float a = left;
        float c = right.m_real;
        float d = right.m_imaginary;

        // Computing c * c + d * d will overflow even in cases where the actual result of the division does not overflow.
        if (Math.Abs(d) < Math.Abs(c))
        {
            float doc = d / c;
            return new ComplexF(a / (c + d * doc), (-a * doc) / (c + d * doc));
        }
        else
        {
            float cod = c / d;
            return new ComplexF(a * cod / (d + c * cod), -a / (d + c * cod));
        }
    }

    public static double Abs(ComplexF value)
    {
        return Hypot(value.m_real, value.m_imaginary);
    }

    private static float Hypot(float a, float b)
    {
        // Using
        //   sqrt(a^2 + b^2) = |a| * sqrt(1 + (b/a)^2)
        // we can factor out the larger component to dodge overflow even when a * a would overflow.

        a = MathF.Abs(a);
        b = MathF.Abs(b);

        float small, large;
        if (a < b)
        {
            small = a;
            large = b;
        }
        else
        {
            small = b;
            large = a;
        }

        if (small == 0.0)
        {
            return (large);
        }
        else if (float.IsPositiveInfinity(large) && !float.IsNaN(small))
        {
            // The NaN test is necessary so we don't return +inf when small=NaN and large=+inf.
            // NaN in any other place returns NaN without any special handling.
            return (float.PositiveInfinity);
        }
        else
        {
            float ratio = small / large;
            return (large * MathF.Sqrt(1.0f + ratio * ratio));
        }

    }

    public static ComplexF Conjugate(ComplexF value)
    {
        // Conjugate of a Complex number: the conjugate of x+i*y is x-i*y
        return new ComplexF(value.m_real, -value.m_imaginary);
    }

    public static bool operator ==(ComplexF left, ComplexF right)
    {
        return left.m_real == right.m_real && left.m_imaginary == right.m_imaginary;
    }

    public static bool operator !=(ComplexF left, ComplexF right)
    {
        return left.m_real != right.m_real || left.m_imaginary != right.m_imaginary;
    }

    public override bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is ComplexF other && Equals(other);
    }

    public bool Equals(ComplexF value)
    {
        return m_real.Equals(value.m_real) && m_imaginary.Equals(value.m_imaginary);
    }

    public override int GetHashCode() => HashCode.Combine(m_real, m_imaginary);

    public override string ToString() => $"{m_real} {(m_imaginary >= 0 ? '+' : '-')} {MathF.Abs(m_imaginary)}i";

    public static explicit operator Complex (ComplexF value) {
        return new Complex(value.m_real, value.m_imaginary);
    }
}