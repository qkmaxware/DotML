namespace DotML;

/// <summary>
/// Generic interface for anything that is tensor-like 
/// </summary>
/// <typeparam name="T">stored element type</typeparam>
public interface ITensorLike<T> {
    /// <summary>
    /// Number of dimensions in tensor
    /// </summary>
    public int Rank {get;}
    /// <summary>
    /// Length/size of a particular dimension
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>dimension length</returns>
    public int GetDimension(int index);
    /// <summary>
    /// Get a particular element from the tensor by index
    /// </summary>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public T GetElementAt(params int[] indices);
}

/// <summary>
/// Extension methods for ITensorLike for common operations
/// </summary>
public static class ITensorLikeExtension {
    /// <summary>
    /// Enumerate over all elements in a tensor
    /// </summary>
    /// <typeparam name="T">tensor element type</typeparam>
    /// <param name="tensor">tensor to enumerate ove</param>
    /// <returns>enumerable of tensor elements</returns>
    public static IEnumerable<T> EnumerateElements<T>(this ITensorLike<T> tensor) {
        var shape = Enumerable.Range(0, tensor.Rank).Select(x => tensor.GetDimension(x)).ToArray();

        foreach (var indices in Safetensors.iterate_over_dimensions(shape)) {
            var index1d = Safetensors.create_1d_index(shape, indices);
            yield return tensor.GetElementAt(indices);
        }
    }

    /// <summary>
    /// Copy the contents of one tensor to another
    /// </summary>
    /// <typeparam name="T">tensor element type</typeparam>
    /// <param name="source">tensor to copy from</param>
    /// <param name="target">tensor to copy to</param>
    /// <exception cref="ArgumentException">thrown when tensors have different shapes</exception>
    public static void CopyTo<T>(this ITensorLike<T> source, IMutableTensorLike<T> target) {
        if (source.Rank != target.Rank)
            throw new ArgumentException("Source and target tensors must have the same rank");

        for (var i = 0; i < source.Rank; i++) {
            if (source.GetDimension(i) != target.GetDimension(i))
                throw new ArgumentException($"Source and target tensors must have the same dimension length for dimension {i}");
        }

        var shape = Enumerable.Range(0, source.Rank).Select(x => source.GetDimension(x)).ToArray();
        foreach (var indices in Safetensors.iterate_over_dimensions(shape)) {
            var value = source.GetElementAt(indices);
            target.SetElementAt(value, indices);
        }
    }
}

/// <summary>
/// Generic interface for anything that is a modifiable tensor-like 
/// </summary>
/// <typeparam name="T">stored element type</typeparam>
public interface IMutableTensorLike<T> : ITensorLike<T> {
    /// <summary>
    /// Set a particular element in the tensor by index
    /// </summary>
    /// <param name="value">value to store</param>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public void SetElementAt(T value, params int[] indices);
}