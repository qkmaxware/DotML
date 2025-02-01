using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace DotML;

/// <summary>
/// Wrapper struct around value array providing vector like functionality. Behaves like pass-by-reference rather than pass-by-value while minimizing heap allocations and dereferences. 
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct Vec<T>: 
    IEnumerable<T>,
    IEquatable<Vec<T>>,
    ITensorLike<T>
where T:INumber<T>
{
    private static T[] NONE = Array.Empty<T>();
    private T[] values; // Literally just a pointer to an array... so size of struct is just int or nint.

    public int Dimensions => 1;
    public int GetDimension(int index) => index switch {
        0 => Dimensionality,
        _ => 1
    };
    public T GetElementAt(params int[] indices) {
        if (indices.Length != 1)
            throw new IndexOutOfRangeException();
        return this[indices[0]];
    }

    /// <summary>
    /// Create an empty vector
    /// </summary>
    public Vec() {
        this.values = NONE;
    }

    /// <summary>
    /// Create a vector of the given size
    /// </summary>
    /// <param name="size">size</param>
    public Vec(int size) {
        this.values = new T[Math.Max(0, size)];
    }

    /// <summary>
    /// Create a vector of the given size with given value for all elements
    /// </summary>
    /// <param name="size">size</param>
    /// <param name="value">filled value</param>
    public Vec(int size, T value) {
        this.values = new T[Math.Max(0, size)];
        this.values.AsSpan().Fill(value);
    }

    /// <summary>
    /// Create a vector of the given size with given value for all elements
    /// </summary>
    /// <param name="size">size</param>
    /// <param name="generator">element generator function</param>
    public Vec(int size, Func<T> generator) {
        size = Math.Max(0, size);
        var values = new T[size];
        for (var i = 0; i < size; i++) {
            values[i] = generator();
        }
        this.values = values;
    }

    /// <summary>
    /// Create a vector with the given values
    /// </summary>
    /// <param name="values">values</param>
    public Vec(T[] values) {
        //this.values = values;
        this.values = values;
    }

    /// <summary>
    /// Create a vector with the given values
    /// </summary>
    /// <param name="value">first value</param>
    /// <param name="components">subsequent values</param>
    public Vec(T value, params T[] components) {
        this.values = new T[components.Length + 1];
        this.values[0] = value;
        for (var i = 0; i < components.Length; i++) {
            this.values[i + 1] = components[i];
        }
    }

    /// <summary>
    /// Index an element from the vector
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>element or zero</returns>
    public T this[int index] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values[index];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            if (index >= 0 && index < values.Length)
                values[index] = value;
        }
    }

    /// <summary>
    /// Maximum element value
    /// </summary>a
    public readonly T MaxValue => this.values.Max() ?? T.Zero;
    
    /// <summary>
    /// Minimum element value
    /// </summary>
    public readonly T MinValue => this.values.Min() ?? T.Zero;

    /// <summary>
    /// Number of dimensions in this vector. IE a 2D vector has 2 values. 
    /// </summary>
    public readonly int Dimensionality => this.values.Length;

    /// <summary>
    /// Get the index of the first element that matches the given predicate condition
    /// </summary>
    /// <param name="condition">condition to search for</param>
    /// <returns>element dimension index or null</returns>
    public readonly int? IndexOf(Predicate<T> condition) {
        for (int i = 0; i < this.Dimensionality; i++) {
            if (condition(values[i]))
                return i;
        }
        return null;
    }

    /// <summary>
    /// Returns the index of the maximal element, useful when using the vector as the output of a ML classifier
    /// </summary>
    /// <returns>index</returns>
    public readonly int IndexOfMaxValue() {
        int index = -1;
        T max = T.Zero;

        for (int i = 0; i < this.Dimensionality; i++) {
            if (i == 0 || values[i] > max) {
                max = values[i];
                index = i;
            }
        }

        return index;
    }

    /// <summary>
    /// Returns the index of the minimal element, useful when using the vector as the output of a ML classifier
    /// </summary>
    /// <returns>index</returns>
    public readonly int IndexOfMinValue() {
        int index = 0;
        T min = T.Zero;

        for (int i = 0; i < this.Dimensionality; i++) {
            if (i == 0 || values[i] < min) {
                min = values[i];
                index = i;
            }
        }

        return index;
    }

    /// <summary>
    /// Squared length of the vector
    /// </summary>
    public readonly T SqrLength() => values.Select(x => x * x).Aggregate(T.Zero, (a , b) => a + b);

    /// <summary>
    /// Dot-product between this vector and another
    /// </summary>
    /// <param name="other">other vector</param>
    /// <returns>dot product</returns>
    public readonly T Dot(Vec<T> other) {
        if (other.Dimensionality != Dimensionality)
            throw new ArithmeticException("Incompatible shape for dot product");
        return this.values.Zip(other.values).Select(x => x.First * x.Second).Aggregate(T.Zero, (a, b) => a + b);
    }

    /// <summary>
    /// Hadamard or element-wise multiplication of two vectors
    /// </summary>
    /// <param name="other">other vector</param>
    /// <returns>element-wise product</returns>
    public readonly Vec<T> Hadamard(Vec<T> other) {
        if (other.Dimensionality != Dimensionality)
            throw new ArithmeticException("Incompatible shape for hadamard product");

        int output_size = this.values.Length;
        T[] outs = new T[output_size];
        for (var i = 0; i < output_size; i++) {
            outs[i] = this.values[i] * other[i];
        }
        return Wrap(outs);
    }

    /// <summary>
    /// Transform the values in this vector to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New vector, same size as the existing one but with elements modified by the mapping function</returns>
    public readonly Vec<R> Transform<R>(Func<T, R> mapping) where R:INumber<R> {
        R[] values = new R[this.Dimensionality];

        for (var col = 0; col < values.Length; col++)
            values[col] = mapping(this.values[col]);
        
        return Vec<R>.Wrap(values);
    }

    /// <summary>
    /// Transform the values in this vector to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New vector, same size as the existing one but with elements modified by the mapping function</returns>
    public readonly Vec<R> Transform<R>(Func<Index, T, R> mapping) where R:INumber<R> {
        R[] values = new R[this.Dimensionality];

        for (var col = 0; col < values.Length; col++)
            values[col] = mapping(col, this.values[col]);
        
        return Vec<R>.Wrap(values);
    }

    /// <summary>
    /// Perform an element-wise operation between this vector and another
    /// </summary>
    /// <typeparam name="T2">2nd vector type</typeparam>
    /// <typeparam name="R">result type</typeparam>
    /// <param name="other">2nd vector</param>
    /// <param name="mapping">mapping function</param>
    /// <returns>New vector</returns>
    /// <exception cref="ArithmeticException">Thrown when the vectors are incompatible for addition</exception>
    public readonly Vec<R> ElementWise<T2, R>(Vec<T2> other, Func<T, T2, R> mapping) where T2:INumber<T2> where R:INumber<R> {
        if (this.Dimensionality != other.Dimensionality)
            throw new ArithmeticException("Incompatible dimensions for element-wise operations");

        var selfv = this.values;
        R[] values = new R[this.Dimensionality];
        for (var i = 0; i < values.Length; i++)
            values[i] = mapping(selfv[i], other[i]);
        
        return Vec<R>.Wrap(values);
    }

    /// <summary>
    /// Transform the values in this vector to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New vector, same size as the existing one but with elements modified by the mapping function</returns>
    public void Apply(Func<T, T> mapping) {
        for (var col = 0; col < values.Length; col++)
            values[col] = mapping(this.values[col]);
    }

    /// <summary>
    /// Transform the values in this vector to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New vector, same size as the existing one but with elements modified by the mapping function</returns>
    public void Apply(Func<Index, T, T> mapping) {
        for (var col = 0; col < values.Length; col++)
            values[col] = mapping(col, this.values[col]);
    }

    /// <summary>
    /// Perform an element-wise operation between this vector and another storing the results in this vector
    /// </summary>
    /// <typeparam name="T2">2nd vector type</typeparam>
    /// <typeparam name="R">result type</typeparam>
    /// <param name="other">2nd vector</param>
    /// <param name="mapping">mapping function</param>
    /// <exception cref="ArithmeticException">Thrown when the vectors are incompatible for addition</exception>
    public void ElementWiseInplace<T2>(Vec<T2> other, Func<T, T2, T> mapping) where T2:INumber<T2> {
        if (this.Dimensionality != other.Dimensionality)
            throw new ArithmeticException("Incompatible dimensions for element-wise operations");

        T[] values = this.values;
        for (var i = 0; i < values.Length; i++)
            values[i] = mapping(values[i], other[i]);
    }

    /// <summary>
    /// Create a vector whose values are the average of all vectors. Vectors should be the same size.
    /// </summary>
    /// <param name="vectors">list of vectors</param>
    /// <returns>vector with averaged values</returns>
    public static Vec<T> Average(IEnumerable<Vec<T>> vectors) {
        T[]? values = null;

        // Sum across all elements
        int size = 0;
        T count = T.Zero;
        foreach (var matrix in vectors) {
            if (values == null) {
                size = matrix.Dimensionality;
                values = new T[size];
            }

            for (var c = 0; c < size; c++) {
                values[c] += matrix[c];
            }

            count = count + T.One;
        }

        if (values is null || count == T.Zero) {
            throw new DivideByZeroException();
        }

        // Divide by count to average it
        for (var c = 0; c < size; c++) {
            values[c] = values[c] / count;
        }

        return Vec<T>.Wrap(values);
    }

    /// <summary>
    /// Deep clone the vector
    /// </summary>
    /// <returns>vector</returns>
    public readonly Vec<T> Clone() {
        T[] values = new T[this.Dimensionality];
        for (var i = 0; i < values.Length; i++) {
            values[i] = this.values[i];
        }
        return Vec<T>.Wrap(values);
    }

    /// <summary>
    /// Implicitly convert an array to a vector
    /// </summary>
    /// <param name="values">array</param>
    public static implicit operator Vec<T> (T[] values) => Vec<T>.Wrap(values);

    /// <summary>
    /// Explicitly convert a vector back to an array
    /// </summary>
    /// <param name="vec">vector</param>
    public static explicit operator T[] (Vec<T> vec) => vec.values;

    public static Vec<T> operator * (T lhs, Vec<T> rhs) => rhs.ScaledBy(lhs);

    public static Vec<T> operator * (Vec<T> lhs, T rhs) => lhs.ScaledBy(rhs);

    public static Vec<T> operator / (Vec<T> lhs, T rhs) => lhs.ScaledBy(T.One / rhs);

    /// <summary>
    /// Return a new vector that is this vector scaled by the given scalar value
    /// </summary>
    /// <param name="value">scaling value</param>
    /// <returns>scaled vector</returns>
    public readonly Vec<T> ScaledBy(T value) {
        var self = this.values;
        var result = new T[this.Dimensionality];

        for (var i = 0; i < result.Length; i++) {
            result[i] = self[i] * value;
        }

        return Wrap(result);
    }

    public static Vec<T> operator + (Vec<T> lhs, Vec<T> rhs) => lhs.AddedWith(rhs);

    /// <summary>
    /// Return a new vector that this the result of adding the vector with another
    /// </summary>
    /// <param name="rhs">other vector</param>
    /// <returns>this + rhs</returns>
    /// <exception cref="ArithmeticException">Thrown when the vectors are incompatible for addition</exception>
    public readonly Vec<T> AddedWith(Vec<T> rhs) {
        if (this.Dimensionality != rhs.Dimensionality)
            throw new ArithmeticException("Incompatible dimensions for vector addition");
        var result = new T[this.Dimensionality];

        var self = this.values;
        var amount = result.Length;
        for (var i = 0; i < amount; i++) {
            result[i] = self[i] + rhs[i];
        }

        return Wrap(result);
    }

    public static Vec<T> operator - (Vec<T> lhs, Vec<T> rhs) => lhs.SubtractWith(rhs);

    /// <summary>
    /// Return a new vector that this the result of subtracting the vector with another
    /// </summary>
    /// <param name="rhs">other vector</param>
    /// <returns>this - rhs</returns>
    /// <exception cref="ArithmeticException">Thrown when the vectors are incompatible for subtraction</exception>
    public readonly Vec<T> SubtractWith(Vec<T> rhs) {
        if (this.Dimensionality != rhs.Dimensionality)
            throw new ArithmeticException("Incompatible dimensions for vector subtraction");
        var result = new T[this.Dimensionality];

        var self = this.values;
        var amount = result.Length;
        for (var i = 0; i < amount; i++) {
            result[i] = self[i] - rhs[i];
        }

        return Wrap(result);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TransformInplace<R>(Vec<R> target, Vec<T> src, Func<T, R> mapping) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        var result = (R[])target;
        var dims = target.Dimensionality;
        if (src.Dimensionality != dims) {
            throw new ArithmeticException("Incompatible dimensions for storing vector transformation result");
        }
        
        for (var r = 0; r < dims; r++) {
            result[r] =  mapping(src[r]);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TransformInplace<R>(Vec<R> target, Vec<T> src, Func<Index, T, R> mapping) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        var result = (R[])target;
        var dims = target.Dimensionality;
        if (src.Dimensionality != dims) {
            throw new ArithmeticException("Incompatible dimensions for storing vector transformation result");
        }
        
        for (var r = 0; r < dims; r++) {
            result[r] =  mapping(r, src[r]);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInplace(Vec<T> target, Vec<T> a, Vec<T> b) {
        if (a.Dimensionality != b.Dimensionality || a.Dimensionality != b.Dimensionality)
            throw new ArithmeticException("Incompatible dimensions for vector addition");

        int dims = target.Dimensionality;
        var result = (T[])target;

        if (dims != a.Dimensionality) {
            throw new ArithmeticException("Incompatible dimensions for storing vector addition result");
        }

        for (int i = 0; i < dims; i++)
            result[i] = a[i] + b[i];   
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SubInplace(Vec<T> target, Vec<T> a, Vec<T> b) {
        if (a.Dimensionality != b.Dimensionality || a.Dimensionality != b.Dimensionality)
            throw new ArithmeticException("Incompatible dimensions for vector subtraction");

        int dims = target.Dimensionality;
        var result = (T[])target;

        if (dims != a.Dimensionality) {
            throw new ArithmeticException("Incompatible dimensions for storing vector subtraction result");
        }

        for (int i = 0; i < dims; i++)
            result[i] = a[i] - b[i];   
    }

    public override string ToString() {
        return "[" + string.Join(", ", values) + "]";
    }

    /// <summary>
    /// Create a span over the entire vector
    /// </summary>
    /// <returns>span over the vector elements</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan() {
        return this.values.AsSpan();
    }

    /// <summary>
    /// Get the underlying array of values
    /// </summary>
    /// <returns>underlying array of elements</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T[] AsArray() {
        return this.values;
    }

    /// <summary>
    /// Convert the vector to a column-matrix representation
    /// </summary>
    /// <returns>matrix</returns>
    public Matrix<T> ToColumnMatrix() {
        var mat = new Matrix<T>(this.Dimensionality,1);
        for(var i = 0; i < values.Length; i++) {
            mat[i, 0] = values[i];
        }
        return mat;
    }

    /// <summary>
    /// Convert the vector to a row-matrix representation
    /// </summary>
    /// <returns>matrix</returns>
    public Matrix<T> ToRowMatrix() {
        var mat = new Matrix<T>(1,this.Dimensionality);
        for(var i = 0; i < values.Length; i++) {
            mat[0, i] = values[i];
        }
        return mat;
    }


    /// <summary>
    /// Shape the vector into multiple 2D matrices of the given sizes.
    /// </summary>
    /// <param name="shapes">Matrix sizes</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Shape(params Shape2D[] shapes) => Shape((IEnumerable<Shape2D>)shapes);

    /// <summary>
    /// Shape the vector into multiple 2D matrices of the given size.
    /// </summary>
    /// <param name="shape">3D Matrix shape</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Shape(Shape3D shape) => Shape(shape.EnumerateSubshapes());

    /// <summary>
    /// Shape the vector into multiple 3D matrices of the given sizes.
    /// </summary>
    /// <param name="shapes">Matrix sizes</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Shape(IEnumerable<Shape2D> shapes) {
        var index = 0;
        foreach (var shape in shapes) {
            var values = new Matrix<T>(shape.Rows, shape.Columns);
             for (var row = 0; row < shape.Rows; row++) {
                for (var col = 0; col < shape.Columns; col++) {
                    if (index < this.Dimensionality)
                        values[row, col] = this[index++];
                    else 
                        values[row, col] = T.Zero;
                }
            }
            yield return values;
        }
    }

    /// <summary>
    /// Shape the vector into multiple 3D matrices of the given size.
    /// </summary>
    /// <param name="rows">Number of rows per matrix</param>
    /// <param name="columns">Number of columns per matrix</param>
    /// <param name="channels">Max number of channels (matrices) to produce. Use -1 for no limit</param>
    /// <returns>Shaped matrices</returns>
    public IEnumerable<Matrix<T>> Shape(Shape2D shape, int channels = -1) {
        var index = 0;
        var channel = 0;
        while (index < this.Dimensionality && (channels < 0 || channel < channels)) {
            var values = new Matrix<T>(shape.Rows, shape.Columns);
            for (var row = 0; row < shape.Rows; row++) {
                for (var col = 0; col < shape.Columns; col++) {
                    values[row, col] = this[index];
                    index++;
                }
            }
            yield return values;
            channel++;
        }
    }

    /// <summary>
    /// Wrap an existing array as a vector without copying it's elements
    /// </summary>
    /// <param name="values">vector elements</param>
    /// <returns>vector</returns>
    public static Vec<T> Wrap(T[] values) {
        return new Vec<T>(values);
    }

    /// <summary>
    /// Clone the values of the given array into the vector copying each element
    /// </summary>
    /// <param name="values">vector elements</param>
    /// <returns>vector</returns>
    public static Vec<T> FromCopy(T[] values) {
        var new_values = new T[values.Length];
        Array.Copy(values, new_values, values.Length);
        return new Vec<T>(new_values);
    }

    public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)values).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => values.GetEnumerator();

    public override bool Equals([NotNullWhen(true)] object? obj) {
        if (obj is not Vec<T> other)
            return false;
        return Equals(other);
    }

    public override int GetHashCode() {
        var amt = this.Dimensionality;
        return HashCode.Combine(
            amt, 
            amt == 0 ? T.Zero : this[0], 
            amt == 0 ? T.Zero : this[this.Dimensionality - 1]
        );
    }

    public bool Equals(Vec<T> other) {
        if (this.Dimensionality != other.Dimensionality)
            return false;
        return this.SequenceEqual(other);
    }
}