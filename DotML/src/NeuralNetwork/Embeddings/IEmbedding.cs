using System.Numerics;

namespace DotML.Network.Embedding;

/// <summary>
/// Interface to convert a vector from one type to another
/// </summary>
/// <typeparam name="T">input type</typeparam>
public interface IEmbedding<TObject, TScalar> where TScalar:INumber<TScalar> {
    /// <summary>
    /// Convert a value to a tensor embedding
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>tensor representation of the value</returns>
    public Tensor<TScalar> ToTensor(TObject value);

    /// <summary>
    /// Convert a value to a matrix embedding
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>2D matrix representation of the value</returns>
    public Matrix<TScalar> ToMatrix(TObject value) {
        var tensor = ToTensor(value);
        var shape = tensor.Shape.NormalizeRank(2);
        return new Matrix<TScalar>(shape[0], shape[1], tensor.AsArray());
    }

    /// <summary>
    /// Convert a value to a vector embedding
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>flattened vector representation of the value</returns>
    public Vec<TScalar> ToVector(TObject value) {
        var tensor = ToTensor(value);
        return new Vec<TScalar>(tensor.AsArray());
    }
}