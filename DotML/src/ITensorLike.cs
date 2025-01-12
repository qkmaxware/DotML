namespace DotML;

/// <summary>
/// Generic interface for anything that is tensor-like 
/// </summary>
/// <typeparam name="T">stored element type</typeparam>
public interface ITensorLike<T> {
    /// <summary>
    /// Number of dimensions in tensor
    /// </summary>
    public int Dimensions {get;}
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