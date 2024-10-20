using System.Collections;
using System.Numerics;
using System.Runtime.InteropServices;

namespace DotML;

/// <summary>
/// Wrapper struct around value array providing vector like functionality. Behaves like pass-by-reference rather than pass-by-value while minimizing heap allocations and dereferences. 
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct Vec<T> 
    : IDistanceable<Vec<T>,T>,
    IEnumerable<T>
where T:INumber<T> 
{
    private T[] values; // Literally just a pointer to an array... so size of struct is just int or nint.

    public Vec(): this(0) {}

    public Vec(int size) {
        this.values = new T[Math.Max(0, size)];
    }

    public Vec(T[] values) {
        this.values = values;
    }

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

    public int IndexOfMaxValue() {
        int index = 0;
        T max = T.Zero;

        for (int i = 0; i < this.Dimensionality; i++) {
            if (i == 0 || values[i] > max) {
                max = values[i];
                index = i;
            }
        }

        return index;
    }

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

    private static T Sqrt(T value) {
        if (value < T.Zero) 
            throw new ArgumentException("Cannot compute the square root of a negative number.");

        T x = T.One;
        T two = T.One + T.One;      // Represents the number 2
        T epsilon = T.One / T.One;  // Small number to define precision (adjust if needed)

        const int max_iterations = 10_000;
        var iter = 0;

        // Iterate to approximate the square root
        while (T.Abs(x * x - value) > epsilon && (iter++ < max_iterations)) {
            x = (x + value / x) / two;
        }

        return x;
    }

    /// <summary>
    /// Length of the vector
    /// </summary>
    public T Length => Sqrt(SqrLength);

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
        return Sqrt(sqrDistance);
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
    /// Implicitly convert an array to a vector
    /// </summary>
    /// <param name="values">array</param>
    public static implicit operator Vec<T> (T[] values) => new Vec<T>(values);

    /// <summary>
    /// Explicitly convert a vector back to an array
    /// </summary>
    /// <param name="vec">vector</param>
    public static explicit operator T[] (Vec<T> vec) => vec.values;

    public static Vec<T> operator + (Vec<T> lhs, Vec<T> rhs) {
        var result = new T[Math.Max(lhs.Dimensionality, rhs.Dimensionality)];

        var amount = Math.Min(lhs.Dimensionality, rhs.Dimensionality);
        for (var i = 0; i < amount; i++) {
            result[i] = lhs[i] + rhs[i];
        }

        return result;
    }

    public static Vec<T> operator - (Vec<T> lhs, Vec<T> rhs) {
        var result = new T[Math.Max(lhs.Dimensionality, rhs.Dimensionality)];

        var amount = Math.Min(lhs.Dimensionality, rhs.Dimensionality);
        for (var i = 0; i < amount; i++) {
            result[i] = lhs[i] - rhs[i];
        }

        return result;
    }

    public override string ToString() {
        return "[" + string.Join(", ", values) + "]";
    }

    public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)values).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => values.GetEnumerator();
}