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
/// Interface for a generic tensor shape
/// </summary>
public interface IShape {
    /// <summary>
    /// Number of dimensions in the shape
    /// </summary>
    public int Rank {get;}
    /// <summary>
    /// Length/size of a particular dimension
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>dimension length</returns>
    public int Length(int index); 
}

/// <summary>
/// 4d or tensor shape
/// </summary>
public struct Shape4D : IShape {
    /// <summary>
    /// Number of batches
    /// </summary>
    public readonly int Batches {get; init;}
    /// <summary>
    /// Number of channels
    /// </summary>
    public readonly int Channels {get; init;}
    /// <summary>
    ///  Number of rows
    /// </summary>
    public readonly int Rows {get; init;}
    /// <summary>
    /// Number of columns
    /// </summary>
    public readonly int Columns {get; init;}
    /// <summary>
    /// Total number of tensor elements contained in the shape
    /// </summary>
    public readonly int Count => Channels * Rows * Columns;
    /// <summary>
    /// Extract the channels, rows, and columns ignoring the batch size
    /// </summary>
    public Shape3D Shape3D => new Shape3D(Channels, Rows, Columns);

    public readonly int Rank => 4;

    public Shape4D() {}

    public Shape4D(int batches, int channel, int rows, int columns) {
        this.Batches = batches;
        this.Channels = channel;
        this.Rows = rows;
        this.Columns = columns;
    }

    public static bool operator == (Shape4D a, Shape4D b) {
        return a.Batches == b.Batches && a.Channels == b.Channels && a.Rows == b.Rows && a.Columns == b.Columns;
    }
    public static bool operator != (Shape4D a, Shape4D b) {
        return a.Batches != b.Batches || a.Channels != b.Channels || a.Rows != b.Rows || a.Columns != b.Columns;
    }
    public override bool Equals([NotNullWhen(true)] object? obj) {
        return obj is Shape4D s && this.Batches == s.Batches && this.Columns == s.Columns && this.Rows == s.Rows && this.Columns == s.Columns;
    }
    public override int GetHashCode() {
        return HashCode.Combine(this.Batches, this.Channels, this.Rows, this.Columns);
    }

    public static implicit operator Shape4D((int, int, int, int) tuple) {
        return new Shape4D(tuple.Item1, tuple.Item2, tuple.Item3, tuple.Item4);
    }

    public IEnumerable<Shape3D> EnumerateSubshapes() {
        var subshape = new Shape3D(this.Channels, this.Rows, this.Columns);
        for (var i = 0; i < this.Batches; i++)
            yield return subshape;
    }

    public void Deconstruct(out int batches, out int channels, out int rows, out int columns) {
        batches = this.Batches;
        channels = this.Channels;
        rows = this.Rows;
        columns = this.Columns;
    }

    public override string ToString() => $"{Batches}x{Channels}x{Rows}x{Columns}";

    public int Length(int index) => index switch {
        0 => Batches,
        1 => Channels,
        2 => Rows,
        3 => Columns,
        _ => 1
    };
}

/// <summary>
/// 3d or tensor shape
/// </summary>
public struct Shape3D : IShape {
    /// <summary>
    /// Number of channels
    /// </summary>
    public readonly int Channels {get; init;}
    /// <summary>
    ///  Number of rows
    /// </summary>
    public readonly int Rows {get; init;} 
    /// <summary>
    /// Number of columns
    /// </summary>
    public readonly int Columns {get; init;}
    /// <summary>
    /// Total number of tensor elements contained in the shape
    /// </summary>
    public readonly int Count => Channels * Rows * Columns;
    /// <summary>
    /// Extract the rows, and columns ignoring the channel count
    /// </summary>
    public Shape2D Shape2D => new Shape2D(Rows, Columns);

    public readonly int Rank => 3;

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

    public IEnumerable<Shape2D> EnumerateSubshapes() {
        var shape2d = new Shape2D(this.Rows, this.Columns);
        for (var i = 0; i < this.Channels; i++)
            yield return shape2d;
    }

    public void Deconstruct(out int channels, out int rows, out int columns) {
        channels = this.Channels;
        rows = this.Rows;
        columns = this.Columns;
    }

    public override string ToString() => $"{Channels}x{Rows}x{Columns}";

    public int Length(int index) => index switch {
        0 => Channels,
        1 => Rows,
        2 => Columns,
        _ => 1
    };
}

/// <summary>
/// 2d or matrix shape
/// </summary>
public struct Shape2D : IShape {
    /// <summary>
    ///  Number of rows
    /// </summary>
    public readonly int Rows {get; init;}
    /// <summary>
    /// Number of columns
    /// </summary>
    public readonly int Columns {get; init;}

    public readonly int Rank => 2;

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

    public static implicit operator Shape3D(Shape2D shape) {
        return new Shape3D(1, shape.Rows, shape.Columns);
    }

    public void Deconstruct(out int rows, out int columns) {
        rows = this.Rows;
        columns = this.Columns;
    }

    public override string ToString() => $"{Rows}x{Columns}";

    public int Length(int index) => index switch {
        0 => Rows,
        1 => Columns,
        _ => 1
    };
}