using System.Drawing;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Apply pooling to reduce the size of the image data
/// </summary>
public abstract class PoolingLayer : ConvolutionalFeedforwardNetworkLayer {
    /// <summary>
    /// Size of the filter horizontally
    /// </summary>
    public int FilterWidth {get; private set;}

    /// <summary>
    /// Size of the filter vertically
    /// </summary>
    public int FilterHeight {get; private set;}

    /// <summary>
    /// Horizontal movement stride (minimum 1)
    /// </summary>
    public int StrideX {get; private set;}
    /// <summary>
    /// Vertical movement stride (minimum 1)
    /// </summary>
    public int StrideY {get; private set;}

    public override int InputCount => -1;
    public override int OutputCount => -1;
    public override int NeuronCount => -1;

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public PoolingLayer(int size) : this(size, size, size, size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public PoolingLayer(int size, int stride) : this(size, size, stride, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public PoolingLayer(Size size) : this(size, size.Width, size.Height) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public PoolingLayer(Size size, int strideX, int strideY) : this(size.Width, size.Height, strideX, strideY) {}

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public PoolingLayer(int width, int height, int strideX, int strideY) {
        this.FilterWidth = width;
        this.FilterHeight = height;
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);
    }

    public override void Initialize(IInitializer initializer) {}

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public override int TrainableParameterCount() => 0;

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}

public abstract class LocalPoolingLayer : PoolingLayer {
    
    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public LocalPoolingLayer(int size) : base(size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public LocalPoolingLayer(int size, int stride) : base(size, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public LocalPoolingLayer(Size size) : base(size) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalPoolingLayer(Size size, int strideX, int strideY) : base(size, strideX, strideY) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalPoolingLayer(int width, int height, int strideX, int strideY) : base(width, height, strideX, strideY) { }


    protected abstract double Accumulate(double current, double delta, int count);
    protected abstract double Aggregate(double current, int count);

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) {
        // Each channel generates exactly 1 output
        var channels = inputs.Length;
        var pooled = new Matrix<double>[channels];

        var filterWidth = this.FilterWidth;
        var filterHeight = this.FilterHeight;

        for (var channel = 0; channel < channels; channel++) {
            var input = inputs[channel];
            var inputWidth = input.Columns;
            var inputHeight = input.Rows;

            var outputWidth = ((inputWidth - filterWidth) / StrideX) + 1;
            var outputHeight = ((inputHeight - filterHeight) / StrideY) + 1;

            var result = new Matrix<double>(outputWidth, outputHeight);
            var data = (double[,])result;
            pooled[channel] = result;

            for (var row = 0; row < outputHeight; row++) {
                for (var col = 0; col < outputWidth; col++) {
                    (int StartX, int StartY, int EndX, int EndY) region = (
                        col * StrideX, 
                        row * StrideY,
                        col * StrideX + filterWidth,
                        row * StrideY + filterHeight
                    );

                    var accumulator = 0.0;
                    var count = 0;
                    for (var irow = region.StartY; irow < region.EndY; irow++) {
                        for (var icol = region.StartX; icol < region.EndX; icol++) {
                            accumulator = Accumulate(accumulator, input[irow, icol], ++count);
                        }
                    }

                    data[row, col] = Aggregate(accumulator, count);
                }
            }
        }

        return pooled;
    }
}

/// <summary>
/// Apply max pooling to reduce the size of the input data by selecting the max element
/// </summary>
public class LocalMaxPoolingLayer : LocalPoolingLayer {
    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public LocalMaxPoolingLayer(int size) : base(size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public LocalMaxPoolingLayer(int size, int stride) : base(size, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public LocalMaxPoolingLayer(Size size) : base(size) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalMaxPoolingLayer(Size size, int strideX, int strideY) : base(size, strideX, strideY) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalMaxPoolingLayer(int width, int height, int strideX, int strideY) : base(width, height, strideX, strideY) { }

    protected override double Accumulate(double current, double delta, int count) {
        return Math.Max(current, delta);
    }

    protected override double Aggregate(double current, int count){
        return current;
    }
}

/// <summary>
/// Apply average pooling to reduce the size of the input data by selecting the average element
/// </summary>
public class LocalAvgPoolingLayer : LocalPoolingLayer {
    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public LocalAvgPoolingLayer(int size) : base(size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public LocalAvgPoolingLayer(int size, int stride) : base(size, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public LocalAvgPoolingLayer(Size size) : base(size) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalAvgPoolingLayer(Size size, int strideX, int strideY) : base(size, strideX, strideY) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalAvgPoolingLayer(int width, int height, int strideX, int strideY) : base(width, height, strideX, strideY) { }

    protected override double Accumulate(double current, double delta, int count) {
        return current + delta;
    }

    private static readonly double Epsilon = 1e-8;

    protected override double Aggregate(double current, int count){
        return current / (count + Epsilon);
    }
}