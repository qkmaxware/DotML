using System.Collections;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace DotML;

/// <summary>
/// Wrapper struct around value array providing vector like functionality. Behaves like pass-by-reference rather than pass-by-value while minimizing heap allocations and dereferences. 
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct Vec<T> 
    : IDistanceable<Vec<T>,T>,
    IEnumerable<T>
where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T>
{
    private static T[] NONE = new T[0];
    private T[] values; // Literally just a pointer to an array... so size of struct is just int or nint.

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
    /// Create a vector with the given values
    /// </summary>
    /// <param name="values">values</param>
    public Vec(T[] values) {
        //this.values = values;
        this.values = values;
    }

    private Vec(T[] values, bool shared) {
        if (!shared) {
            this.values = new T[values.Length];
            Array.Copy(values, this.values, values.Length);
        } else {
            this.values = values;
        }
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
        get => index >= 0 && index < values.Length ? values[index] : T.Zero;
        //set => values[index] = value;
    }

    /// <summary>
    /// Maximum element value
    /// </summary>a
    public T MaxValue => this.values.Max() ?? T.Zero;
    
    /// <summary>
    /// Minimum element value
    /// </summary>
    public T MinValue => this.values.Min() ?? T.Zero;

    /// <summary>
    /// Number of dimensions in this vector. IE a 2D vector has 2 values. 
    /// </summary>
    public int Dimensionality => this.values.Length;

    /// <summary>
    /// Get the index of the first element that matches the given predicate condition
    /// </summary>
    /// <param name="condition">condition to search for</param>
    /// <returns>element dimension index or null</returns>
    public int? IndexOf(Predicate<T> condition) {
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
    public int IndexOfMaxValue() {
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
    public int IndexOfMinValue() {
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
    /// Length of the vector
    /// </summary>
    public T Length => T.Sqrt(SqrLength);

    /// <summary>
    /// Squared length of the vector
    /// </summary>
    public T SqrLength => values.Select(x => x * x).Aggregate(T.Zero, (a , b) => a + b);

    /// <summary>
    /// Distance from one vector to another
    /// </summary>
    /// <param name="instance">other vector</param>
    /// <returns>distance</returns>
    public T DistanceTo(Vec<T> instance) {
        // Distance from A to B = len(B - A)
        // ...
        // len(B - A)
        // sqrt(sqrLen(B - A))
        // sqrt((B-A).x^2 + (B-A).y^2 + ... (B-A).n^2)
        T sqrDistance = T.Zero;
        for (var dim = 0; dim < Math.Max(this.Dimensionality, instance.Dimensionality); dim++) {
            var subtraction = instance[dim] - this[dim];
            sqrDistance += subtraction * subtraction;
        }
        return T.Sqrt(sqrDistance);
    }

    /// <summary>
    /// Dot-product between this vector and another
    /// </summary>
    /// <param name="other">other vector</param>
    /// <returns>dot product</returns>
    public readonly T Dot(Vec<T> other) {
        return this.values.Zip(other.values).Select(x => x.First * x.Second).Aggregate(T.Zero, (a, b) => a + b);
    }

    /// <summary>
    /// Hadamard or element-wise multiplication of two vectors
    /// </summary>
    /// <param name="other">other vector</param>
    /// <returns>element-wise product</returns>
    public readonly Vec<T> Hadamard(Vec<T> other) {
        int output_size = Math.Max(this.values.Length, other.values.Length);
        T[] outs = new T[output_size];
        for (var i = 0; i < output_size; i++) {
            outs[i] = this.values[i] * other[i];
        }
        return Wrap(outs);
    }

    /// <summary>
    /// Normalize the vector using the softmax function which converts the vector into a probability distribution with values between 0 and 1.
    /// </summary>
    /// <returns>normalized vector</returns>
    public Vec<T> SoftmaxNormalized(){
        var sum = T.Zero;
        T[] values = new T[this.Dimensionality];
        for (var i = 0; i < this.Dimensionality; i++) {
            var exp_i = T.Exp(this.values[i]);
            values[i] = exp_i;
            sum += exp_i;
        }
        for (var i = 0; i < this.Dimensionality; i++) {
            values[i] = values[i] / sum;
        }
        return Vec<T>.Wrap(values);
    }

    /// <summary>
    /// Deep clone the vector
    /// </summary>
    /// <returns>vector</returns>
    public Vec<T> Clone() {
        T[] values = new T[this.Dimensionality];
        for (var i = 0; i < this.Dimensionality; i++) {
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

    public static Vec<T> operator * (T lhs, Vec<T> rhs) {
        var result = new T[rhs.Dimensionality];

        for (var i = 0; i < result.Length; i++) {
            result[i] = lhs * rhs[i];
        }

        return Wrap(result);
    }

    public static Vec<T> operator * (Vec<T> lhs, T rhs) {
        var result = new T[lhs.Dimensionality];

        for (var i = 0; i < result.Length; i++) {
            result[i] = lhs[i] * rhs;
        }

        return Wrap(result);
    }

    public static Vec<T> operator + (Vec<T> lhs, Vec<T> rhs) {
        if (lhs.Dimensionality != rhs.Dimensionality)
            throw new ArgumentException("Incompatible dimensions for vector addition");
        var result = new T[Math.Max(lhs.Dimensionality, rhs.Dimensionality)];

        var amount = Math.Min(lhs.Dimensionality, rhs.Dimensionality);
        for (var i = 0; i < amount; i++) {
            result[i] = lhs[i] + rhs[i];
        }

        return Wrap(result);
    }

    public static Vec<T> operator - (Vec<T> lhs, Vec<T> rhs) {
        if (lhs.Dimensionality != rhs.Dimensionality)
            throw new ArgumentException("Incompatible dimensions for vector addition");
        var result = new T[Math.Max(lhs.Dimensionality, rhs.Dimensionality)];

        var amount = Math.Min(lhs.Dimensionality, rhs.Dimensionality);
        for (var i = 0; i < amount; i++) {
            result[i] = lhs[i] - rhs[i];
        }

        return Wrap(result);
    }

    public override string ToString() {
        return "[" + string.Join(", ", values) + "]";
    }

    /// <summary>
    /// Convert the vector to a column-matrix representation
    /// </summary>
    /// <returns>matrix</returns>
    public Matrix<T> ToColumnMatrix() {
        var mat = new T[this.Dimensionality,1];
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
        var mat = new T[1,this.Dimensionality];
        for(var i = 0; i < values.Length; i++) {
            mat[0, i] = values[i];
        }
        return mat;
    }

    /// <summary>
    /// Shape the vector into multiple 3D matrices of the given sizes.
    /// </summary>
    /// <param name="shapes">Matrix sizes</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Shape(params Shape[] shapes) {
        var index = 0;
        foreach (var shape in shapes) {
            T[,] values = new T[shape.Rows, shape.Columns];
             for (var row = 0; row < shape.Rows; row++) {
                for (var col = 0; col < shape.Columns; col++) {
                    if (index < this.Dimensionality)
                        values[row, col] = this[index++];
                    else 
                        values[row, col] = T.Zero;
                }
            }
            yield return Matrix<T>.Wrap(values);
        }
    }

    /// <summary>
    /// Shape the vector into multiple 3D matrices of the given size.
    /// </summary>
    /// <param name="rows">Number of rows per matrix</param>
    /// <param name="columns">Number of columns per matrix</param>
    /// <param name="channels">Max number of channels (matrices) to produce. Use -1 for no limit</param>
    /// <returns>Shaped matrices</returns>
    public IEnumerable<Matrix<T>> Shape(Shape shape, int channels = -1) {
        var index = 0;
        var channel = 0;
        while (index < this.Dimensionality && (channels < 0 || channel < channels)) {
            T[,] values = new T[shape.Rows, shape.Columns];
            for (var row = 0; row < shape.Rows; row++) {
                for (var col = 0; col < shape.Columns; col++) {
                    values[row, col] = this[index];
                    index++;
                }
            }
            yield return Matrix<T>.Wrap(values);
            channel++;
        }
    }

    /// <summary>
    /// Wrap an existing array as a vector without copying it's elements
    /// </summary>
    /// <param name="values">vector elements</param>
    /// <returns>vector</returns>
    public static Vec<T> Wrap(T[] values) {
        return new Vec<T>(values, shared: true);
    }

    /// <summary>
    /// Clone the values of the given array into the vector copying each element
    /// </summary>
    /// <param name="values">vector elements</param>
    /// <returns>vector</returns>
    public static Vec<T> FromCopy(T[] values) {
        return new Vec<T>(values, shared: false);
    }

    /// <summary>
    /// Get the associated label corresponding to the element with the largest value
    /// </summary>
    /// <param name="labels">list of labels to use</param>
    /// <returns>label or null if no labels match</returns>
    public string? GetLabel(IList<string> labels) {
        var i = this.IndexOfMaxValue();
        if (i < 0 || i >= labels.Count)
            return null;
        return labels[i];
    }

    public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)values).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => values.GetEnumerator();
}