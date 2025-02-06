using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Apply pooling to reduce the size of the image data
/// <see href="https://en.wikipedia.org/wiki/Pooling_layer"/>
/// </summary>
public abstract class PoolingLayer : FeedforwardNetworkLayer {
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

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public PoolingLayer(Shape3D input_size, int size) : this(input_size, size, size, size, size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public PoolingLayer(Shape3D input_size, int size, int stride) : this(input_size, size, size, stride, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public PoolingLayer(Shape3D input_size, Size size) : this(input_size, size, size.Width, size.Height) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public PoolingLayer(Shape3D input_size, Size size, int strideX, int strideY) : this(input_size, size.Width, size.Height, strideX, strideY) {}

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public PoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY) {
        this.FilterWidth = width;
        this.FilterHeight = height;
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);

        this.InputShape = input_size;
        var outputWidth = ((input_size.Columns - this.FilterWidth) / this.StrideX) + 1;
        var outputHeight = ((input_size.Rows - this.FilterHeight) / this.StrideY) + 1;
        this.OutputShape = new Shape3D(input_size.Channels, outputHeight, outputWidth);
    }

    public override void Initialize(IInitializer initializer) {}

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public override int TrainableParameterCount() => 0;

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}

public abstract class LocalPoolingLayer : PoolingLayer {
    
    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public LocalPoolingLayer(Shape3D input_size, int size) : base(input_size, size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public LocalPoolingLayer(Shape3D input_size, int size, int stride) : base(input_size, size, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public LocalPoolingLayer(Shape3D input_size, Size size) : base(input_size, size) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalPoolingLayer(Shape3D input_size, Size size, int strideX, int strideY) : base(input_size, size, strideX, strideY) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalPoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY) : base(input_size, width, height, strideX, strideY) { }


    protected abstract double Accumulate(double current, double delta, int count);
    protected abstract double Aggregate(double current, int count);

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) {
        // Each channel generates exactly 1 output
        var channels = inputs.Channels;
        var pooled = new Matrix<double>[channels];

        var filterWidth = this.FilterWidth;
        var filterHeight = this.FilterHeight;

        var stridex = this.StrideX;
        var stridey = this.StrideY;

        var outputWidth = this.OutputShape.Columns;
        var outputHeight = this.OutputShape.Rows;

        for (var channel = 0; channel < channels; channel++) {
            var input = inputs[channel];

            var result = new Matrix<double>(outputWidth, outputHeight);
            pooled[channel] = result;

            for (var row = 0; row < outputHeight; row++) {
                var StartY = row * stridey;
                var EndY = row * stridey + filterHeight;
                for (var col = 0; col < outputWidth; col++) {
                    var StartX = col * stridex;
                    var EndX = col * stridex + filterWidth;

                    var accumulator = 0.0;
                    var count = 0;
                    for (var irow = StartY; irow < EndY; irow++) {
                        for (var icol = StartX; icol < EndX; icol++) {
                            accumulator = Accumulate(accumulator, input[irow, icol], ++count);
                        }
                    }

                    result[row, col] = Aggregate(accumulator, count);
                }
            }
        }

        return (FeatureSet<double>)pooled;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<double>[] input_errors = new FeatureSet<double>[args.OutputBatch.Batches];

        Parallel.For(0, args.OutputBatch.Batches, batchIndex => {
            // Extract inputs, outputs, and errors
            var inputs = args.InputBatch[batchIndex];
            var outputs = args.OutputBatch[batchIndex];
            var errors = args.OutputErrors[batchIndex];

            int featureCount = inputs.Channels;
            var batchErrors = new Matrix<double>[featureCount];

            var filterWidth = FilterWidth;
            var filterHeight = FilterHeight;
            var filterElementCount = filterWidth * filterHeight;

            Parallel.For(0, featureCount, featureIndex => {
                // Get the input and output for this batch item
                var input = inputs[featureIndex];
                var output = outputs[featureIndex];
                var error = errors[featureIndex];

                // Initialize the error matrix for the input
                var inputError = new Matrix<double>(input.Rows, input.Columns);

                // Loop over output
                for (int row = 0; row < output.Rows; row++) {
                    var StartY = row * StrideY;
                    var EndY = Math.Min(row * StrideY + filterHeight, input.Rows);
                    for (int col = 0; col < output.Columns; col++) {
                        var StartX = col * StrideX;
                        var EndX = Math.Min(col * StrideX + filterWidth, input.Columns);

                        // Loop over input values where the filter is applied
                        Backpropagate(inputError, input, error[row, col], filterElementCount, StartX, EndX, StartY, EndY);
                    }
                }

                // Assign the errors for this input features
                batchErrors[featureIndex] = inputError;
            });

            // Assign the errors for the input features into the batch
            input_errors[batchIndex] = new FeatureSet<double>(batchErrors);
        });
        

        // Pass errors along for next layer
        return new BackpropagationReturns (
            new BatchedFeatureSet<double>(input_errors),
            null
        );
    }

    protected abstract void Backpropagate(Matrix<double> inputError, Matrix<double> input, double error, int filterSize, int startX, int endX, int startY, int endY);
}

/// <summary>
/// Apply max pooling to reduce the size of the input data by selecting the max element
/// </summary>
public class LocalMaxPoolingLayer : LocalPoolingLayer {
    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public LocalMaxPoolingLayer(Shape3D input_size, int size) : base(input_size, size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public LocalMaxPoolingLayer(Shape3D input_size, int size, int stride) : base(input_size, size, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public LocalMaxPoolingLayer(Shape3D input_size, Size size) : base(input_size, size) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalMaxPoolingLayer(Shape3D input_size, Size size, int strideX, int strideY) : base(input_size, size, strideX, strideY) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalMaxPoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY) : base(input_size, width, height, strideX, strideY) { }

    protected override double Accumulate(double current, double delta, int count) {
        return Math.Max(current, delta);
    }

    protected override double Aggregate(double current, int count){
        return current;
    }
    
   protected override void Backpropagate(Matrix<double> inputError, Matrix<double> input, double error, int filterSize, int startX, int endX, int startY, int endY) {
        int maxRow = 0, maxCol = 0; double maxVal = double.MinValue; // Values for max pooling
        for (int kr = startY; kr < endY; kr++) {
            for (int kc = startX; kc < endX; kc++) {
                var value = input[kr, kc];

                // Compute; Assume max pooling (avg is different)
                if (value > maxVal) {
                    maxVal = value;
                    maxRow = kr;
                    maxCol = kc;
                }
            }
        }
        inputError[maxRow, maxCol] += error; 
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
    public LocalAvgPoolingLayer(Shape3D input_size, int size) : base(input_size, size) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public LocalAvgPoolingLayer(Shape3D input_size, int size, int stride) : base(input_size, size, stride) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    public LocalAvgPoolingLayer(Shape3D input_size, Size size) : base(input_size, size) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="size">filter size</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalAvgPoolingLayer(Shape3D input_size, Size size, int strideX, int strideY) : base(input_size, size, strideX, strideY) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public LocalAvgPoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY) : base(input_size, width, height, strideX, strideY) { }

    protected override double Accumulate(double current, double delta, int count) {
        return current + delta;
    }

    protected override double Aggregate(double current, int count){
        return current / Math.Max(1, count);
    }

    protected override void Backpropagate(Matrix<double> inputError, Matrix<double> input, double error, int filterSize, int startX, int endX, int startY, int endY) {
        double errorContribution = error / Math.Max(1, filterSize); // Distribute the error
        for (int kr = startY; kr < endY; kr++) {
            for (int kc = startX; kc < endX; kc++) {
                inputError[kr, kc] += errorContribution;            // Assign the error contribution to each element in the pooling region
            }   
        }
    }
}