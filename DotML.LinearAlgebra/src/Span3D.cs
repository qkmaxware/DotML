using System.Runtime.CompilerServices;

namespace DotML;

/// <summary>
/// Treat a span as a 3D matrix like structure
/// </summary>
/// <typeparam name="T">span type</typeparam>
public readonly ref struct Span3D<T>
{
    private readonly Span<T> values;
    public readonly int Channels;
    private readonly int channelStride;
    public readonly int Rows;
    public readonly int Columns;

    public Span3D(Span<T> span, int channels, int rows, int columns)
    {
        values = span;
        Channels = channels;
        channelStride = rows * columns;
        Rows = rows;
        Columns = columns;
    }

    public static Span3D<T> Empty => new(Span<T>.Empty, 0, 0, 0);

    public bool IsEmpty
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.IsEmpty;
    }

    public int Length
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.Length;
    }

    public T this[int channel, int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values[channel * channelStride + row * Columns + col];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => values[channel * channelStride + row * Columns + col] = value;
    }

    public Span2D<T> this[int channel]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => new Span2D<T>(values.Slice(channel * channelStride, channelStride), Rows, Columns);
    }

    public void Clear() => values.Clear();

    public void CopyTo(Span3D<T> destination)
    {
        if (Channels != destination.Channels || Rows != destination.Rows || Columns != destination.Columns)
            throw new ArgumentException("Destination dimensions must match source dimensions.");
        values.CopyTo(destination.values);
    }

    public Span<T>.Enumerator GetEnumerator() => values.GetEnumerator();

    public bool TryCopyTo(Span3D<T> destination)
    {
        if (Channels != destination.Channels || Rows != destination.Rows || Columns != destination.Columns)
            return false;
        return values.TryCopyTo(destination.values);
    }

    public void Fill(T value) => values.Fill(value);

    public ref T GetPinnableReference() => ref values.GetPinnableReference();

    public Span<T> AsSpan() => values;

    public ReadOnlySpan3D<T> AsReadOnly => new ReadOnlySpan3D<T>(this.values, this.Channels, this.Rows, this.Columns);

    public static implicit operator ReadOnlySpan3D<T>(Span3D<T> span) => new ReadOnlySpan3D<T>(span.values, span.Channels, span.Rows, span.Columns);

    public T[,,] ToArray()
    {
        var array = new T[Channels, Rows, Columns];
        for (int chan = 0; chan < Channels; chan++)
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Columns; c++)
                    array[chan, r, c] = this[chan, r, c];
        return array;
    }
}

/// <summary>
/// Treat a span as a readonly 3D matrix like structure
/// </summary>
/// <typeparam name="T">span type</typeparam>
public readonly ref struct ReadOnlySpan3D<T>
{
    private readonly ReadOnlySpan<T> values;
    public readonly int Channels;
    private readonly int channelStride;
    public readonly int Rows;
    public readonly int Columns;

    public ReadOnlySpan3D(ReadOnlySpan<T> span, int channels, int rows, int columns)
    {
        values = span;
        Channels = channels;
        channelStride = rows * columns;
        Rows = rows;
        Columns = columns;
    }

    public static ReadOnlySpan3D<T> Empty => new(Span<T>.Empty, 0, 0, 0);

    public bool IsEmpty
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.IsEmpty;
    }

    public int Length
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.Length;
    }

    public T this[int channel, int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values[channel * channelStride + row * Columns + col];
    }

    public ReadOnlySpan2D<T> this[int channel]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => new ReadOnlySpan2D<T>(values.Slice(channel * channelStride, channelStride), Rows, Columns);
    }

    public void CopyTo(Span3D<T> destination)
    {
        if (Channels != destination.Channels || Rows != destination.Rows || Columns != destination.Columns)
            throw new ArgumentException("Destination dimensions must match source dimensions.");
        values.CopyTo(destination.AsSpan());
    }

    public ReadOnlySpan<T>.Enumerator GetEnumerator() => values.GetEnumerator();

    public bool TryCopyTo(Span3D<T> destination)
    {
        if (Channels != destination.Channels || Rows != destination.Rows || Columns != destination.Columns)
            return false;
        return values.TryCopyTo(destination.AsSpan());
    }

    public ReadOnlySpan<T> AsSpan() => values;

    public T[,,] ToArray()
    {
        var array = new T[Channels, Rows, Columns];
        for (int chan = 0; chan < Channels; chan++)
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Columns; c++)
                    array[chan, r, c] = this[chan, r, c];
        return array;
    }
}