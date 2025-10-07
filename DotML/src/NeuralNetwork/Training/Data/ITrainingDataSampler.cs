using System.Numerics;

namespace DotML.Network.Training;

public interface ITrainingDataSampler<TType> 
where TType:INumber<TType>
{
    /// <summary>
    /// The shape of the input tensors
    /// </summary>
    public TensorShape InputShape { get; }
    /// <summary>
    /// The shape of the output tensors
    /// </summary>
    public TensorShape OutputShape { get; }
    /// <summary>
    /// Try to fetch a batch of samples from the training data up to the given batch size.
    /// </summary>
    /// <param name="batchSize">requested batch size, actual batch size may be less than this depending on then number of samples in the training data</param>
    /// <returns>batch of samples</returns>
    public IEnumerable<(Tensor<TType> Input, Tensor<TType> Ouput)> Sample(int batchSize = 1);
}

public class SequentialSampler<TType>: ITrainingDataSampler<TType> 
where TType:INumber<TType>
{
    private ITrainingDataSource<TType> src;

    public TensorShape InputShape => src.InputShape;
    public TensorShape OutputShape => src.OutputShape;

    public SequentialSampler(ITrainingDataSource<TType> src)
    {
        this.src = src;
    }

    private (Tensor<TType> Input, Tensor<TType> Ouput) Next(int index) {
        return src[index];
    }

    public IEnumerable<(Tensor<TType> Input, Tensor<TType> Ouput)> Sample(int batchSize = 1) {
        batchSize = Math.Max(batchSize, 1);
        int startIndex = 0;
        var count = src.Count;
        var ishape = src.InputShape;
        var oshape = src.OutputShape;

        while (startIndex < count) {
            // Compute "real" batch size
            var length = Math.Min(batchSize, count - startIndex);
            var endIndex = startIndex + length;

            // Create input shape. Prepend batch dimension
            var shape = new int[ishape.Rank + 1];
            shape[0] = length;
            ishape.AsDimensionSpan().CopyTo(shape.AsSpan().Slice(1));

            var input = Tensor<TType>.Defaults(new TensorShape(shape));
            var ispan = input.AsSpan();

            // Create output shape. Prepend batch dimension
            shape = new int[oshape.Rank + 1];
            shape[0] = length;
            oshape.AsDimensionSpan().CopyTo(shape.AsSpan().Slice(1));
            
            var output = Tensor<TType>.Defaults(new TensorShape(shape));
            var ospan = output.AsSpan();

            // Populate batched tensors
            for (var i = 0; i < length; i++) {
                var pair = Next(startIndex + i);
                // Populate input tensor. Copy elements from each input into single batched tensor
                {
                    var target = ispan.Slice(i * ishape.Stride(0), ishape.Stride(0));
                    var src = pair.Input.AsSpan();
                    src.CopyTo(target);
                }
                // Populate output tensor. Copy elements from each output into single batched tensor
                {
                    var target = ospan.Slice(i * oshape.Stride(0), oshape.Stride(0));
                    var src = pair.Ouput.AsSpan();
                    src.CopyTo(target);
                }
            }

            // Return batched tensors
            yield return (input, output);
            startIndex += length;
        }
    }
}

public class RandomSampler<TType>: ITrainingDataSampler<TType> 
where TType:INumber<TType>
{
    private ITrainingDataSource<TType> src;
    private bool allowDuplicates;
    private Random random;

    public TensorShape InputShape => src.InputShape;
    public TensorShape OutputShape => src.OutputShape;

    public RandomSampler(ITrainingDataSource<TType> src, bool allowDuplicates = true)
    {
        this.src = src;
        this.allowDuplicates = allowDuplicates;
        this.random = new Random();
    }

    private (Tensor<TType> Input, Tensor<TType> Ouput) Next(int index, List<int>? working) {
        if (allowDuplicates)
            return src[random.Next(0, src.Count)];

        if (working is null || working.Count == 0)
            throw new IndexOutOfRangeException("All items have been sampled");
            
        var rng_idx = random.Next(0, working.Count);
        var item_idx = working[rng_idx];
        var item = src[item_idx];
        working.RemoveAt(rng_idx);
        return item;
    }

    public IEnumerable<(Tensor<TType> Input, Tensor<TType> Ouput)> Sample(int batchSize = 1) {
        batchSize = Math.Max(batchSize, 1);
        int startIndex = 0;
        var count = src.Count;
        var ishape = src.InputShape;
        var oshape = src.OutputShape;

        List<int>? working = allowDuplicates ? null : new List<int>(Enumerable.Range(0, count).ToArray());

        while (startIndex < count) {
            // Compute "real" batch size
            var length = Math.Min(batchSize, count - startIndex);
            var endIndex = startIndex + length;

            // Create input shape. Prepend batch dimension
            var shape = new int[ishape.Rank + 1];
            shape[0] = length;
            ishape.AsDimensionSpan().CopyTo(shape.AsSpan().Slice(1));

            var input = Tensor<TType>.Defaults(new TensorShape(shape));
            var ispan = input.AsSpan();

            // Create output shape. Prepend batch dimension
            shape = new int[oshape.Rank + 1];
            shape[0] = length;
            oshape.AsDimensionSpan().CopyTo(shape.AsSpan().Slice(1));
            
            var output = Tensor<TType>.Defaults(new TensorShape(shape));
            var ospan = output.AsSpan();

            // Populate batched tensors
            for (var i = 0; i < length; i++) {
                var pair = Next(startIndex + i, working);
                // Populate input tensor. Copy elements from each input into single batched tensor
                {
                    var target = ispan.Slice(i * ishape.Stride(0), ishape.Stride(0));
                    var src = pair.Input.AsSpan();
                    src.CopyTo(target);
                }
                // Populate output tensor. Copy elements from each output into single batched tensor
                {
                    var target = ospan.Slice(i * oshape.Stride(0), oshape.Stride(0));
                    var src = pair.Ouput.AsSpan();
                    src.CopyTo(target);
                }
            }

            // Return batched tensors
            yield return (input, output);
            startIndex += length;
        }
    }
}