using System.Diagnostics.CodeAnalysis;
using System.Drawing;

namespace DotML.Network;

/// <summary>
/// A 2D size structure
/// </summary>
public readonly struct Size2D
: IParsable<Size2D>
{
    /// <summary>
    /// Size in the X direction
    /// </summary>
    public int Width { get; }
    /// <summary>
    /// Size in the Y direction
    /// </summary>
    public int Height { get; }

    public Size2D(int width, int height)
    {
        Width = width;
        Height = height;
    }

    public void Deconstruct(out int width, out int height)
    {
        width = this.Width;
        height = Height;
    }

    public static Size2D Parse(string s, IFormatProvider? provider)
    {
        var parts = s.Split(',');
        if (parts.Length != 2)
            throw new FormatException("Input string was not in a correct format.");
        return new Size2D(int.Parse(parts[0].Trim(), provider), int.Parse(parts[1].Trim(), provider));
    }

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out Size2D result)
    {
        var parts = s?.Split(',');
        if (parts == null || parts.Length != 2)
        {
            result = default;
            return false;
        }
        if (int.TryParse(parts[0].Trim(), provider, out var width) &&
            int.TryParse(parts[1].Trim(), provider, out var height))
        {
            result = new Size2D(width, height);
            return true;
        }
        result = default;
        return false;
    }

    public static implicit operator Size2D((int Width, int Height) tuple) => new Size2D(tuple.Width, tuple.Height);
    public static implicit operator Size2D(int size) => new Size2D(size, size);
}

/// <summary>
/// Stride over a 2D tensor
/// </summary>
public readonly struct Stride2D
: IParsable<Stride2D>
{
    /// <summary>
    /// Stride in the X direction
    /// </summary>
    public int X { get; }
    /// <summary>
    /// Stride in the Y direction
    /// </summary>
    public int Y { get; }

    public Stride2D(int x, int y)
    {
        X = x;
        Y = y;
    }

    public void Deconstruct(out int x, out int y)
    {
        x = X;
        y = Y;
    }

    public static Stride2D Parse(string s, IFormatProvider? provider)
    {
        var parts = s.Split(',');
        if (parts.Length != 2)
            throw new FormatException("Input string was not in a correct format.");
        return new Stride2D(int.Parse(parts[0].Trim(), provider), int.Parse(parts[1].Trim(), provider));
    }

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out Stride2D result)
    {
        var parts = s?.Split(',');
        if (parts == null || parts.Length != 2)
        {
            result = default;
            return false;
        }
        if (int.TryParse(parts[0].Trim(), provider, out var width) &&
            int.TryParse(parts[1].Trim(), provider, out var height))
        {
            result = new Stride2D(width, height);
            return true;
        }
        result = default;
        return false;
    }

    public static implicit operator Stride2D((int X, int Y) tuple) => new Stride2D(tuple.X, tuple.Y);
    public static implicit operator Stride2D(int stride) => new Stride2D(stride, stride);
}

/// <summary>
/// Dilation over a 2D tensor
/// </summary>
public readonly struct Dilation2D
: IParsable<Dilation2D>
{
    /// <summary>
    /// Dilation in the X direction
    /// </summary>
    public int X { get; }
    /// <summary>
    /// Dilation in the Y direction
    /// </summary>
    public int Y { get; }

    public Dilation2D(int x, int y)
    {
        X = x;
        Y = y;
    }

    public void Deconstruct(out int x, out int y)
    {
        x = X;
        y = Y;
    }

    public static Dilation2D Parse(string s, IFormatProvider? provider)
    {
        var parts = s.Split(',');
        if (parts.Length != 2)
            throw new FormatException("Input string was not in a correct format.");
        return new Dilation2D(int.Parse(parts[0].Trim(), provider), int.Parse(parts[1].Trim(), provider));
    }

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out Dilation2D result)
    {
        var parts = s?.Split(',');
        if (parts == null || parts.Length != 2)
        {
            result = default;
            return false;
        }
        if (int.TryParse(parts[0].Trim(), provider, out var width) &&
            int.TryParse(parts[1].Trim(), provider, out var height))
        {
            result = new Dilation2D(width, height);
            return true;
        }
        result = default;
        return false;
    }

    public static implicit operator Dilation2D((int X, int Y) tuple) => new Dilation2D(tuple.X, tuple.Y);
    public static implicit operator Dilation2D(int dilation) => new Dilation2D(dilation, dilation);
}

/// <summary>
/// Padding over a 2D tensor / matrix
/// </summary>
public readonly struct Padding2D
: IParsable<Padding2D>
{
    /// <summary>
    /// Padding on the left side
    /// </summary>
    public int Left { get; }
    /// <summary>
    /// Padding on the top side
    /// </summary>
    public int Top { get; }
    /// <summary>
    /// Padding on the right side
    /// </summary>
    public int Right { get; }
    /// <summary>
    /// Padding on the bottom side
    /// </summary>
    public int Bottom { get; }

    public Padding2D(int left, int top, int right, int bottom)
    {
        Left = left;
        Top = top;
        Right = right;
        Bottom = bottom;
    }

    public void Deconstruct(out int left, out int top, out int right, out int bottom)
    {
        left = Left;
        top = Top;
        right = Right;
        bottom = Bottom;
    }

    public static Padding2D Parse(string s, IFormatProvider? provider)
    {
        var parts = s.Split(',');
        if (parts.Length != 4)
            throw new FormatException("Input string was not in a correct format.");
        return new Padding2D(
            int.Parse(parts[0].Trim(), provider),
            int.Parse(parts[1].Trim(), provider),
            int.Parse(parts[2].Trim(), provider),
            int.Parse(parts[3].Trim(), provider)
        );
    }

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out Padding2D result)
    {
        var parts = s?.Split(',');
        if (parts == null || parts.Length != 4)
        {
            result = default;
            return false;
        }
        if (int.TryParse(parts[0].Trim(), provider, out var left) &&
            int.TryParse(parts[1].Trim(), provider, out var top) &&
            int.TryParse(parts[2].Trim(), provider, out var right) &&
            int.TryParse(parts[3].Trim(), provider, out var bottom))
        {
            result = new Padding2D(left, top, right, bottom);
            return true;
        }
        result = default;
        return false;
    }

    public static implicit operator Padding2D((int Left, int Top, int Right, int Bottom) tuple) => new Padding2D(tuple.Left, tuple.Top, tuple.Right, tuple.Bottom);
    public static implicit operator Padding2D(int pad) => new Padding2D(pad, pad, pad, pad);
}