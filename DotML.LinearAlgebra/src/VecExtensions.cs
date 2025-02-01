using System.Numerics;

namespace DotML;

/// <summary>
/// Extension methods for type specific additional functionality for vectors
/// </summary>
public static class VecExtensions {

    /// <summary>
    /// Length of the vector
    /// </summary>
    public static T Length<T>(this Vec<T> vec) where T:INumber<T>, IRootFunctions<T> {
        return T.Sqrt(vec.SqrLength());
    } 

    /// <summary>
    /// Distance from one vector to another
    /// </summary>
    /// <param name="other">other vector</param>
    /// <returns>distance</returns>
    public static T DistanceTo<T>(this Vec<T> vec, Vec<T> other) where T:INumber<T>, IRootFunctions<T> {
        T sqrDistance = T.Zero;
        for (var dim = 0; dim < Math.Min(vec.Dimensionality, other.Dimensionality); dim++) {
            var subtraction = other[dim] - vec[dim];
            sqrDistance += subtraction * subtraction;
        }
        return T.Sqrt(sqrDistance);
    }

    /// <summary>
    /// Normalize this vector
    /// </summary>
    /// <returns>normalized vector</returns>
    public static Vec<T> Normalized<T>(this Vec<T> vec) where T:INumber<T>, IRootFunctions<T> {
        T[] values = new T[vec.Dimensionality];
        var len = vec.Length();

        for (var i = 0; i < values.Length; i++)
            values[i] = vec[i] / len;

        return Vec<T>.Wrap(values);
    }

    /// <summary>
    /// Normalize the vector using the softmax function which converts the vector into a probability distribution with values between 0 and 1.
    /// </summary>
    /// <returns>normalized vector</returns>
    public static Vec<T> SoftmaxNormalized<T>(this Vec<T> vec) where T:INumber<T>, IExponentialFunctions<T>  {
        var sum = T.Zero;
        T[] values = new T[vec.Dimensionality];
        for (var i = 0; i < vec.Dimensionality; i++) {
            var exp_i = T.Exp(vec[i]);
            values[i] = exp_i;
            sum += exp_i;
        }
        for (var i = 0; i < vec.Dimensionality; i++) {
            values[i] = values[i] / sum;
        }
        return Vec<T>.Wrap(values);
    } 

}