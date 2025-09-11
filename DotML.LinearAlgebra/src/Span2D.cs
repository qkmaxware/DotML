using System.Runtime.CompilerServices;

namespace DotML;

/// <summary>
/// Treat a span as a 2D matrix like structure
/// </summary>
/// <typeparam name="T">span type</typeparam>
public readonly ref struct Span2D<T>
{
    private readonly Span<T> values;
    public readonly int Rows;
    public readonly int Columns;

    public Span2D(Span<T> span, int rows, int columns)
    {
        values = span;
        Rows = rows;
        Columns = columns;
    }

    public static Span2D<T> Empty => new(Span<T>.Empty, 0, 0);

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

    public T this[int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values[row * Columns + col];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => values[row * Columns + col] = value;
    }

    public void Clear() => values.Clear();

    public void CopyTo(Span2D<T> destination)
    {
        if (Rows != destination.Rows || Columns != destination.Columns)
            throw new ArgumentException("Destination dimensions must match source dimensions.");
        values.CopyTo(destination.values);
    }

    public Span<T>.Enumerator GetEnumerator() => values.GetEnumerator();

    public bool TryCopyTo(Span2D<T> destination)
    {
        if (Rows != destination.Rows || Columns != destination.Columns)
            return false;
        return values.TryCopyTo(destination.values);
    }

    public void Fill(T value) => values.Fill(value);

    public ref T GetPinnableReference() => ref values.GetPinnableReference();

    public Span<T> AsSpan() => values;

    public T[,] ToArray()
    {
        var array = new T[Rows, Columns];
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Columns; c++)
                array[r, c] = this[r, c];
        return array;
    }
}

/// <summary>
/// Treat a span as a readonly 2D matrix like structure
/// </summary>
/// <typeparam name="T">span type</typeparam>
public readonly ref struct ReadOnlySpan2D<T>
{
    private readonly ReadOnlySpan<T> values;
    public readonly int Rows;
    public readonly int Columns;

    public ReadOnlySpan2D(Span<T> span, int rows, int columns)
    {
        values = span; // Implicit cast to ReadOnlySpan
        Rows = rows;
        Columns = columns;
    }

    public T this[int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values[row * Columns + col];
    }

    public static Span2D<T> Empty => new(Span<T>.Empty, 0, 0);

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

    public void CopyTo(Span2D<T> destination)
    {
        if (Rows != destination.Rows || Columns != destination.Columns)
            throw new ArgumentException("Destination dimensions must match source dimensions.");
        values.CopyTo(destination.AsSpan());
    }

    public ReadOnlySpan<T>.Enumerator GetEnumerator() => values.GetEnumerator();

    public bool TryCopyTo(Span2D<T> destination)
    {
        if (Rows != destination.Rows || Columns != destination.Columns)
            return false;
        return values.TryCopyTo(destination.AsSpan());
    }

    public ReadOnlySpan<T> AsSpan() => values;

    public T[,] ToArray()
    {
        var array = new T[Rows, Columns];
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Columns; c++)
                array[r, c] = this[r, c];
        return array;
    }
}