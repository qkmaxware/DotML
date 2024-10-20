namespace DotML;

/// <summary>
/// Interface to convert a vector from one type to another
/// </summary>
/// <typeparam name="T">input type</typeparam>
public interface IFeatureExtractor<T> {
    /// <summary>
    /// Convert a value to a vector
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>vector representation of the value</returns>
    public Vec<double> ToVector(T value);
}