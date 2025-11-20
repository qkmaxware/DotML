using System.Collections;
using System.ComponentModel.DataAnnotations;
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

    public SubsetTrainingDataSource<TType> Subset(Range range)
    {
        var (offset, length) = range.GetOffsetAndLength(Count);
        return new SubsetTrainingDataSource<TType>(offset, length, this);
    }

    public SubsetTrainingDataSource<TType> RandomSubset(int length)
    {
        length = Math.Clamp(length, 0, this.Count);
        var max_cap = this.Count - length;
        var random = Random.Shared;
        var offset = random.Next(0, max_cap);
        return new SubsetTrainingDataSource<TType>(offset, length, this);
    }

    public ITrainingDataSampler<TType> CreateSequentialSampler()
    {
        return new SequentialSampler<TType>(this);
    }

    public ITrainingDataSampler<TType> CreateRandomSampler(bool allowDuplicates = true)
    {
        return new RandomSampler<TType>(this, allowDuplicates: allowDuplicates);
    }

    public ITrainingDataSampler<TType> CreateRandomSampler(int count, bool allowDuplicates = true)
    {
        return new RandomSampler<TType>(this, allowDuplicates: allowDuplicates, sampleSize: count);
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

public class CombinationTrainingDataSource<TType>
: ITrainingDataSource<TType>
where TType : INumber<TType>
{
    private List<ITrainingDataSource<TType>> srcs = new List<ITrainingDataSource<TType>>();

    public CombinationTrainingDataSource() { }

    public CombinationTrainingDataSource(params IEnumerable<ITrainingDataSource<TType>> values)
    {
        AddRange(values);
    }

    public void Add(ITrainingDataSource<TType> src)
    {
        if (srcs.Count == 0 || (src.InputShape.Equals(this.InputShape) && src.OutputShape.Equals(this.OutputShape)))
        {
            srcs.Add(src);
            return;
        }
        throw new ArgumentException("Cannot add to combined source. The incoming source has an incompatible shape with the combined shape.");
    }

    public void AddRange(IEnumerable<ITrainingDataSource<TType>> values)
    {
        foreach (var i in values)
            Add(i);
    }

    public (Tensor<TType> Input, Tensor<TType> Output) this[int index]
    {
        get
        {
            if (index < 0)
                throw new IndexOutOfRangeException();

            // Determine which source we fall in
            var offset = index;
            ITrainingDataSource<TType>? selected = null;
            foreach (var src in srcs)
            {
                if (offset < src.Count)
                {
                    selected = src;
                    break;
                }
                else
                {
                    offset -= src.Count;
                    continue;
                }
            }
            if (selected is null)
            {
                throw new IndexOutOfRangeException();
            }

            // Get the element from that source
            return selected[offset];
        }
    }

    public int Count => srcs.Sum(src => src.Count);

    public TensorShape InputShape => srcs.FirstOrDefault()?.InputShape ?? TensorShape.Scalar;

    public TensorShape OutputShape => srcs.FirstOrDefault()?.OutputShape ?? TensorShape.Scalar;

    public IEnumerator<(Tensor<TType> Input, Tensor<TType> Output)> GetEnumerator()
    {
        foreach (var src in srcs)
        {
            foreach (var io in src)
                yield return io;
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

}

public class SubsetTrainingDataSource<TType>
: ITrainingDataSource<TType>
where TType : INumber<TType>
{
    private int offset;
    private int length;
    ITrainingDataSource<TType> data;

    public SubsetTrainingDataSource(int offset, int length, ITrainingDataSource<TType> underlying)
    {
        this.offset = offset;
        this.length = length;
        this.data = underlying;
    }

    public (Tensor<TType> Input, Tensor<TType> Output) this[int index] => data[offset + index];

    public int Count => length;

    public TensorShape InputShape => data.InputShape;

    public TensorShape OutputShape => data.OutputShape;

    public IEnumerator<(Tensor<TType> Input, Tensor<TType> Output)> GetEnumerator()
    {
        for (var i = 0; i < length; i++)
        {
            yield return data[offset + i];
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

}