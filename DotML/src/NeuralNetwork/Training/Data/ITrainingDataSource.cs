using System.Numerics;

namespace DotML.Network.Training;

/// <summary>
/// A generic training data source
/// </summary>
/// <typeparam name="TType">tensor element type</typeparam>
public interface ITrainingDataSource<TType>
: IEnumerable<(Tensor<TType> Input, Tensor<TType> Output)>
where TType : INumber<TType>
{
    /// <summary>
    /// Number of training pairs in the data source
    /// </summary>
    public int Count { get; }
    /// <summary>
    /// Shape of the input training tensors
    /// </summary>
    public TensorShape InputShape { get; }
    /// <summary>
    /// Shape of the output training tensors
    /// </summary>
    public TensorShape OutputShape { get; }

    /// <summary>
    /// Get a particular IO pair in the data source
    /// </summary>
    /// <param name="index">index</param>
    /// <returns>IO pair</returns>
    public (Tensor<TType> Input, Tensor<TType> Output) this[int index] { get; }

    public ITrainingDataSampler<TType> CreateSequentialSampler()
    {
        return new SequentialSampler<TType>(this);
    }

    public ITrainingDataSampler<TType> CreateRandomSampler(bool allowDuplicates = true)
    {
        return new RandomSampler<TType>(this, allowDuplicates);
    }
}

/// <summary>
/// A training data source which is backed by a list of tensors
/// </summary>
/// <typeparam name="TType">tensor element type</typeparam>
public class ListTrainingDataSource<TType>
: List<(Tensor<TType> Input, Tensor<TType> Output)>, ITrainingDataSource<TType>
where TType : INumber<TType>
{
    public TensorShape InputShape { get; init; }
    public TensorShape OutputShape { get; init; }

    public ListTrainingDataSource(TensorShape ishape, TensorShape oshape)
    {
        this.InputShape = ishape;
        this.OutputShape = oshape;
    }

    public ListTrainingDataSource(TensorShape ishape, TensorShape oshape, IEnumerable<(Tensor<TType> Input, Tensor<TType> Output)> data)
    : this(ishape, oshape)
    {
        this.AddRange(data);
    }
}