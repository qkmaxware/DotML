using System.Data;
using System.Numerics;

namespace DotML.Network.Training;

public interface ITrainingDataSampler<TType> 
where TType:INumber<TType>
{
    /// <summary>
    /// Number of samples
    /// </summary>
    public int Count { get; }
    /// <summary>
    /// The shape of the input tensors
    /// </summary>
    public Shape InputShape { get; }
    /// <summary>
    /// The shape of the output tensors
    /// </summary>
    public Shape OutputShape { get; }
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

    public int Count => src.Count;
    public Shape InputShape => src.InputShape;
    public Shape OutputShape => src.OutputShape;

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
            var batchLength = Math.Min(batchSize, count - startIndex);
            var endIndex = startIndex + batchLength;
            if (batchLength == 0)
                break;

            // Create input shape. Prepend batch dimension
            var shape = new int[ishape.Rank + 1];
            shape[0] = batchLength;
            ishape.AsDimensionSpan().CopyTo(shape.AsSpan().Slice(1));

            var input = Tensor<TType>.Defaults(new Shape(shape));
            var ispan = input.AsSpan();

            // Create output shape. Prepend batch dimension
            shape = new int[oshape.Rank + 1];
            shape[0] = batchLength;
            oshape.AsDimensionSpan().CopyTo(shape.AsSpan().Slice(1));
            
            var output = Tensor<TType>.Defaults(new Shape(shape));
            var ospan = output.AsSpan();

            // Populate batched tensors
            for (var i = 0; i < batchLength; i++) {
                var pair = Next(startIndex + i);
                // Populate input tensor. Copy elements from each input into single batched tensor
                {
                    var target = ispan.Slice(i * input.Shape.Stride(0), input.Shape.Stride(0));
                    var src = pair.Input.AsSpan();
                    src.CopyTo(target);
                }
                // Populate output tensor. Copy elements from each output into single batched tensor
                {
                    var target = ospan.Slice(i * output.Shape.Stride(0), output.Shape.Stride(0));
                    var src = pair.Ouput.AsSpan();
                    src.CopyTo(target);
                }
            }

            // Return batched tensors
            yield return (input, output);
            startIndex += batchLength;
        }
    }
}

public class RandomSampler<TType>: ITrainingDataSampler<TType> 
where TType:INumber<TType>
{
    private ITrainingDataSource<TType> src;
    private bool allowDuplicates;
    private Random random;

    private int? sampleSize;
    public int Count => sampleSize.HasValue ? sampleSize.Value : src.Count;
    public Shape InputShape => src.InputShape;
    public Shape OutputShape => src.OutputShape;

    public RandomSampler(ITrainingDataSource<TType> src, bool allowDuplicates = true, int? sampleSize = null)
    {
        this.src = src;
        this.sampleSize = sampleSize;
        this.allowDuplicates = allowDuplicates;
        this.random = new Random();

        if (!allowDuplicates && this.sampleSize > src.Count)
            throw new ArgumentException("Sample size cannot be larger than dataset when duplicates are not allowed.");
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

        List<int>? working = allowDuplicates ? null : new List<int>(Enumerable.Range(0, src.Count).OrderBy(_ => random.Next()).Take(count).ToArray());

        while (startIndex < count) {
            // Compute "real" batch size
            var batchLength = Math.Max(0, Math.Min(batchSize, count - startIndex));
            var endIndex = startIndex + batchLength;
            if (batchLength == 0)
                break;

            // Create input shape. Prepend batch dimension
            var shape = new int[ishape.Rank + 1];
            shape[0] = batchLength;
            ishape.AsDimensionSpan().CopyTo(shape.AsSpan(1));

            var input = Tensor<TType>.Defaults(new Shape(shape));
            var ispan = input.AsSpan();

            // Create output shape. Prepend batch dimension
            shape = new int[oshape.Rank + 1];
            shape[0] = batchLength;
            oshape.AsDimensionSpan().CopyTo(shape.AsSpan(1));
            
            var output = Tensor<TType>.Defaults(new Shape(shape));
            var ospan = output.AsSpan();

            // Populate batched tensors
            for (var i = 0; i < batchLength; i++)
            {
                var pair = Next(startIndex + i, working);
                // Populate input tensor. Copy elements from each input into single batched tensor
                {
                    var target = ispan.Slice(i * input.Shape.Stride(0), input.Shape.Stride(0));
                    var src = pair.Input.AsSpan();
                    src.CopyTo(target);
                }
                // Populate output tensor. Copy elements from each output into single batched tensor
                {
                    var target = ospan.Slice(i * output.Shape.Stride(0), output.Shape.Stride(0));
                    var src = pair.Ouput.AsSpan();
                    src.CopyTo(target);
                }
            }

            // Return batched tensors
            yield return (input, output);
            startIndex += batchLength;
        }
    }
}