using System.Numerics;

namespace DotML;

/// <summary>
/// Interface to convert a vector from one type to another
/// </summary>
/// <typeparam name="T">input type</typeparam>
public interface IFeatureExtractor<T, TVector> where TVector:INumber<TVector> {
    /// <summary>
    /// Convert a value to a vector
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>vector representation of the value</returns>
    public Vec<TVector> ToVector(T value);
}