using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Apply pooling to reduce the size of the image data
/// <see href="https://en.wikipedia.org/wiki/Pooling_layer"/>
/// </summary>
public abstract class LocalPooling : Pooling
{

}

/// <summary>
/// Apply pooling to reduce the size of the image data across the last 2 dimensions of the input (Height and Width)
/// <see href="https://en.wikipedia.org/wiki/Pooling_layer"/>
/// </summary>
public abstract class LocalPooling2D : LocalPooling
{
    /// <summary>
    /// Size of the filter horizontally
    /// </summary>
    public int FilterWidth { get; private set; }

    /// <summary>
    /// Size of the filter vertically
    /// </summary>
    public int FilterHeight { get; private set; }

    /// <summary>
    /// Horizontal movement stride (minimum 1)
    /// </summary>
    public int StrideX { get; private set; }

    /// <summary>
    /// Vertical movement stride (minimum 1)
    /// </summary>
    public int StrideY { get; private set; }

    /// <summary>
    /// Horizontal padding of the input (min 0)
    /// </summary>
    public int PaddingX { get; private set; } = 0;

    /// <summary>
    /// Vertical padding of the input (min 0)
    /// </summary>
    public int PaddingY { get; private set; } = 0;

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    /// <param name="padding">input padding</param>
    public LocalPooling2D(int size, int stride, int padding) : this(size, size, stride, stride, padding, padding) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    /// <param name="paddingX">horizontal input padding</param>
    /// <param name="paddingY">vertical input stride</param>
    public LocalPooling2D(int width, int height, int strideX, int strideY, int paddingX, int paddingY)
    {
        this.FilterWidth = width;
        this.FilterHeight = height;
        this.StrideX = strideX;
        this.StrideY = strideY;
        this.PaddingX = paddingX;
        this.PaddingY = paddingY;
    }

    public override void Initialize(IInitializer initializer) { }

    // Accumulate value over entire kernel
    protected abstract float Accumulate(float current, float delta, int count);
    // Final value aggregation (when combined with accumulate it should cover most use cases)
    protected abstract float Aggregate(float current, int count);

    public override TensorShape ForwardShape(TensorShape input)
    {
        var inputWidth = input.Length(^1);
        var inputHeight = input.Length(^2);

        var padded_input_width = inputWidth + 2 * PaddingX;
        var padded_input_height = inputHeight + 2 * PaddingY;
        var outputWidth = ((padded_input_width - this.FilterWidth) / this.StrideX) + 1;
        var outputHeight = ((padded_input_height - this.FilterHeight) / this.StrideY) + 1;

        var outDimensions = input.AsDimensionSpan().ToArray();
        outDimensions[^2] = outputHeight;
        outDimensions[^1] = outputWidth;
        return new TensorShape(outDimensions);
    }

    public override Tensor<float> Forward(Tensor<float> inputs)
    {
        // Each channel generates exactly 1 output
        var originalRank = inputs.Shape.Rank;
        inputs = inputs.ReshapeShared(inputs.Shape.EnsureRank(2)); // Minimum of [N,C,H,W], but can have more batch dims [D1, D2, ..., C, H, W]

        var filterWidth = this.FilterWidth;
        var filterHeight = this.FilterHeight;

        var stridex = this.StrideX;
        var stridey = this.StrideY;

        var inputWidth = inputs.Shape.Length(^1);
        var inputHeight = inputs.Shape.Length(^2);
        var inputSliceLength = inputWidth * inputHeight;

        var padded_input_width = inputWidth + 2 * PaddingX;
        var padded_input_height = inputHeight + 2 * PaddingY;
        var outputWidth = ((padded_input_width - this.FilterWidth) / this.StrideX) + 1;
        var outputHeight = ((padded_input_height - this.FilterHeight) / this.StrideY) + 1;
        var outputSliceLength = outputWidth * outputHeight;

        var outDimensions = inputs.Shape.AsDimensionSpan().ToArray();
        outDimensions[^2] = outputHeight;
        outDimensions[^1] = outputWidth;
        var outShape = new TensorShape(outDimensions);
        var output = Tensor<float>.Defaults(outShape);

        Span<float> inputSpan = inputs.AsSpan();
        Span<float> outputSpan = output.AsSpan();
        var batches = inputSpan.Length / inputSliceLength;

        // Iterate in groups of size inputSliceLength
        Parallel.For(0, batches, ParallelOptions, (batch) =>
        //for (var batch = 0; batch < batches; batch++)
        {
            var inputMatrix = inputs.AsSpan(batch * inputSliceLength, inputSliceLength);
            var ouputMatrix = output.AsSpan(batch * outputSliceLength, outputSliceLength);

            for (var row = 0; row < outputHeight; row++)
            {
                var StartY = row * stridey;
                var EndY = row * stridey + filterHeight;

                var row_base = row * outputWidth;

                for (var col = 0; col < outputWidth; col++)
                {
                    var StartX = col * stridex;
                    var EndX = col * stridex + filterWidth;

                    var accumulator = 0.0f;
                    var count = 0;
                    for (var irow = StartY; irow < EndY; irow++)
                    {
                        var real_irow = irow - PaddingY;

                        if (real_irow < 0 || real_irow >= inputHeight)
                        {
                            count += filterWidth;
                            continue;
                        }

                        var real_irow_base = real_irow * inputWidth;
                        for (var icol = StartX; icol < EndX; icol++)
                        {
                            var real_icol = icol - PaddingX;

                            if (real_icol < 0 || real_icol >= inputWidth)
                            {
                                count += 1; // In padding, add 1 to count but do nothing else
                                continue;
                            }

                            var x = inputMatrix[real_irow_base + real_icol];
                            accumulator = Accumulate(accumulator, x, ++count);
                        }
                    }

                    ouputMatrix[row_base + col] = Aggregate(accumulator, count);
                }
            }
            //}
        });

        return output;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var filterWidth = FilterWidth;
        var filterHeight = FilterHeight;
        var filterElementCount = filterWidth * filterHeight;

        var errShape = dy.Shape.EnsureRank(2);
        var errRows = errShape.Length(^2);
        var errColumns = errShape.Length(^1);
        var errMatSize = errRows * errColumns;

        var inputShape = x.Shape.EnsureRank(2);
        var inputRows = inputShape.Length(^2);
        var inputColumns = inputShape.Length(^1);
        var inputMatrixSize = inputRows * inputColumns;

        var dx = Tensor<float>.Defaults(x.Shape);

        var batches = x.ElementCount / inputMatrixSize;

        Parallel.For(0, batches, ParallelOptions, (batch) =>
        //for (var batch = 0; batch < batches; batch++)
        {
            var inputSpan = x.AsSpan();
            var outSpan = dx.AsSpan();
            var errSpan = dy.AsSpan();

            var input = new ReadOnlySpan2D<float>(inputSpan.Slice(batch * inputMatrixSize, inputMatrixSize), inputRows, inputColumns);
            var inputErrors = new Span2D<float>(outSpan.Slice(batch * inputMatrixSize, inputMatrixSize), inputRows, inputColumns); ;
            var errors = new Span2D<float>(errSpan.Slice(batch * errMatSize, errMatSize), errRows, errColumns);

            // Loop over output
            for (int row = 0; row < errRows; row++)
            {
                var StartY = row * StrideY;
                var EndY = row * StrideY + filterHeight;
                for (int col = 0; col < errColumns; col++)
                {
                    var StartX = col * StrideX;
                    var EndX = col * StrideX + filterWidth;

                    // Loop over input values where the filter is applied
                    Backpropagate(
                        inputErrors,            // Where to place the resulting values
                        input,                  // The original input
                        errors[row, col],       // The error dY
                        filterElementCount,     // The number of filters

                        // The region of the input that produced the output/output error
                        StartX - PaddingX,
                        EndX - PaddingX,
                        StartY - PaddingY,
                        EndY - PaddingY
                    );
                }
            }
            //}
        });

        // Pass errors along for next layer
        return new Gradient(dx);
    }

    protected abstract void Backpropagate(Span2D<float> dx, ReadOnlySpan2D<float> x, float dy, int items, int startX, int endX, int startY, int endY);

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)  { /* Nothing to do here */ }
}