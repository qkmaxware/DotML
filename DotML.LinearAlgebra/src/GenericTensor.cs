namespace DotML;

/// <summary>
/// A generic tensor storage container
/// </summary>
/// <typeparam name="T">Element type</typeparam>
public class GenericTensor<T> : IMutableTensorLike<T> {
    private int[] shape;
    private T[] values;

    /// <summary>
    /// Create a generic tensor of the given shape
    /// </summary>
    /// <param name="shape">shape</param>
    public GenericTensor(params int[] shape) {
        this.shape = shape;
        this.values = new T[shape.Aggregate(1, (a, b) => a * b)];
    }

    /// <summary>
    /// Number of dimensions in tensor
    /// </summary>
    public int Dimensions => shape.Length;

    /// <summary>
    /// Length/size of a particular dimension
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>dimension length</returns>    
    public int GetDimension(int index) {
        if (index >= 0 && index < shape.Length)
            return shape[index];
        else
            return 1;
    }   

    /// <summary>
    /// Get or set a particular element from the tensor by index
    /// </summary>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public T this[params int[] indices] {
        get => GetElementAt(indices);
        set => SetElementAt(value, indices);
    }

    /// <summary>
    /// Get a particular element from the tensor by index
    /// </summary>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public T GetElementAt(params int[] indices) {
        var index_1d = Safetensors.create_1d_index(shape, indices);
        return values[index_1d];
    }

    /// <summary>
    /// Set a particular element in the tensor by index
    /// </summary>
    /// <param name="value">value to store</param>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public void SetElementAt(T value, params int[] indices) {
        var index_1d = Safetensors.create_1d_index(shape, indices);
        values[index_1d] = value;
    }
}