using System.Collections;
using System.Data;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Serialization;

namespace DotML;

/// <summary>
/// 3d or tensor shape
/// </summary>
public struct Shape3D {
    public readonly int Channels;
    /// <summary>
    ///  Number of rows
    /// </summary>
    public readonly int Rows; 
    /// <summary>
    /// Number of columns
    /// </summary>
    public readonly int Columns;

    public Shape3D() {}

    public Shape3D(int channel, int rows, int columns) {
        this.Channels = channel;
        this.Rows = rows;
        this.Columns = columns;
    }

    public static bool operator == (Shape3D a, Shape3D b) {
        return a.Channels == b.Channels && a.Rows == b.Rows && a.Columns == b.Columns;
    }
    public static bool operator != (Shape3D a, Shape3D b) {
        return a.Channels != b.Channels || a.Rows != b.Rows || a.Columns != b.Columns;
    }
    public override bool Equals([NotNullWhen(true)] object? obj) {
        return obj is Shape3D s && this.Columns == s.Columns && this.Rows == s.Rows && this.Columns == s.Columns;
    }
    public override int GetHashCode() {
        return HashCode.Combine(this.Channels, this.Rows, this.Columns);
    }

    public static implicit operator Shape3D((int, int, int) tuple) {
        return new Shape3D(tuple.Item1, tuple.Item2, tuple.Item3);
    }

    public void Deconstruct(out int channels, out int rows, out int columns) {
        channels = this.Channels;
        rows = this.Rows;
        columns = this.Columns;
    }

    public override string ToString() => $"{Channels}x{Rows}x{Columns}";
}

/// <summary>
/// 2d or matrix shape
/// </summary>
public struct Shape2D {
    /// <summary>
    ///  Number of rows
    /// </summary>
    public readonly int Rows;
    /// <summary>
    /// Number of columns
    /// </summary>
    public readonly int Columns;

    public Shape2D() {}

    public Shape2D(int rows, int columns) {
        this.Rows = rows;
        this.Columns = columns;
    }

    public static bool operator == (Shape2D a, Shape2D b) {
        return a.Rows == b.Rows && a.Columns == b.Columns;
    }
    public static bool operator != (Shape2D a, Shape2D b) {
        return a.Rows != b.Rows || a.Columns != b.Columns;
    }
    public override bool Equals([NotNullWhen(true)] object? obj) {
        return obj is Shape2D s && this.Rows == s.Rows && this.Columns == s.Columns;
    }
    public override int GetHashCode() {
        return HashCode.Combine(this.Rows, this.Columns);
    }

    public static implicit operator Shape2D((int, int) tuple) {
        return new Shape2D(tuple.Item1, tuple.Item2);
    }

    public void Deconstruct(out int rows, out int columns) {
        rows = this.Rows;
        columns = this.Columns;
    }

    public override string ToString() => $"{Rows}x{Columns}";
}