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
    /// Horizontal padding of the input (min 0)
    /// </summary>
    public int PaddingX {get; private set;} = 0;

    /// <summary>
    /// Vertical padding of the input (min 0)
    /// </summary>
    public int PaddingY {get; private set;} = 0;

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    public PoolingLayer(Shape3D input_size, int size) : this(input_size, size, size, size, size, 0, 0) { }

    /// <summary>
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    public PoolingLayer(Shape3D input_size, int size, int stride) : this(input_size, size, size, stride, stride, 0, 0) { }

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
    public PoolingLayer(Shape3D input_size, Size size, int strideX, int strideY) : this(input_size, size.Width, size.Height, strideX, strideY, 0, 0) {}

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    public PoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY, int paddingX, int paddingY) {
        this.FilterWidth = width;
        this.FilterHeight = height;
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);
        this.PaddingX = Math.Max(0, paddingX);
        this.PaddingY = Math.Max(0, paddingY);

        this.InputShape = input_size;
        var padded_input_width = input_size.Columns + 2 * PaddingX;
        var padded_input_height = input_size.Rows + 2 * PaddingY;
        var outputWidth = ((padded_input_width - this.FilterWidth) / this.StrideX) + 1;
        var outputHeight = ((padded_input_height - this.FilterHeight) / this.StrideY) + 1;
        this.OutputShape = new Shape3D(input_size.Channels, outputHeight, outputWidth);
    }

    public override void Initialize(IInitializer initializer) {}

    public override void SubtractGradients(LayerGradients? gradients) { }

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public override int TrainableParameterCount() => 0;

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);
    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}

/// <summary>
/// Apply local pooling as opposed to global pooling
/// </summary>
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
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    /// <param name="padding">input padding</param>
    public LocalPoolingLayer(Shape3D input_size, int size, int stride, int padding) : base(input_size, size, size, stride, stride, padding, padding) { }

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
    public LocalPoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY) : base(input_size, width, height, strideX, strideY, 0, 0) { }

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    /// <param name="paddingX">horizontal input padding</param>
    /// <param name="paddingY">vertical input stride</param>
    public LocalPoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY, int paddingX, int paddingY) : base(input_size, width, height, strideX, strideY, paddingX, paddingY) { }


    protected abstract float Accumulate(float current, float delta, int count);
    protected abstract float Aggregate(float current, int count);

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> inputs) {
        // Each channel generates exactly 1 output
        var channels = inputs.Channels;
        var pooled = new Matrix<float>[channels];

        var filterWidth = this.FilterWidth;
        var filterHeight = this.FilterHeight;

        var stridex = this.StrideX;
        var stridey = this.StrideY;

        var inputWidth = inputs.Columns;
        var inputHeight = inputs.Rows;

        var outputWidth = this.OutputShape.Columns;
        var outputHeight = this.OutputShape.Rows;

        var kernel_size = filterWidth * filterHeight;

        for (var channel = 0; channel < channels; channel++) {
            var input = inputs[channel];

            var result = new Matrix<float>(outputWidth, outputHeight);
            pooled[channel] = result;

            for (var row = 0; row < outputHeight; row++) {
                var StartY = row * stridey;
                var EndY = row * stridey + filterHeight;
                for (var col = 0; col < outputWidth; col++) {
                    var StartX = col * stridex;
                    var EndX = col * stridex + filterWidth;

                    var accumulator = 0.0f;
                    var count = 0;
                    for (var irow = StartY; irow < EndY; irow++) {
                        var real_irow = irow - PaddingY;

                        if (real_irow < 0 || real_irow >= inputHeight) {
                            count += filterWidth;
                            continue;
                        }

                        for (var icol = StartX; icol < EndX; icol++) {
                            var real_icol = icol - PaddingX;

                            if (real_icol < 0 || real_icol >= inputWidth) {
                                count += 1;
                                continue;
                            }

                            var x = input[real_irow, real_icol];
                            accumulator = Accumulate(accumulator, x, ++count); // Hmm should count also reflect the "padded" 0's?
                        }
                    }

                    result[row, col] = Aggregate(accumulator, count);
                }
            }
        }

        return (FeatureSet<float>)pooled;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<float>[] input_errors = new FeatureSet<float>[args.OutputBatch.Batches];

        Parallel.For(0, args.OutputBatch.Batches, batchIndex => {
            // Extract inputs, outputs, and errors
            var inputs = args.InputBatch[batchIndex];
            var outputs = args.OutputBatch[batchIndex];
            var errors = args.OutputErrors[batchIndex];

            int featureCount = inputs.Channels;
            var batchErrors = new Matrix<float>[featureCount];

            var filterWidth = FilterWidth;
            var filterHeight = FilterHeight;
            var filterElementCount = filterWidth * filterHeight;

            Parallel.For(0, featureCount, featureIndex => {
                // Get the input and output for this batch item
                var input = inputs[featureIndex];
                var output = outputs[featureIndex];
                var error = errors[featureIndex];

                // Initialize the error matrix for the input
                var inputError = new Matrix<float>(input.Rows, input.Columns);

                // Loop over output
                for (int row = 0; row < output.Rows; row++) {
                    var StartY = row * StrideY;
                    var EndY = row * StrideY + filterHeight;
                    for (int col = 0; col < output.Columns; col++) {
                        var StartX = col * StrideX;
                        var EndX = col * StrideX + filterWidth;

                        // Loop over input values where the filter is applied
                        Backpropagate(
                            inputError,             // Where to place the resulting values
                            input,                  // The original input
                            error[row, col],        // The error dY
                            filterElementCount,     // The number of filters

                            // The region of the input that produced the output/output error
                            StartX - PaddingX,                 
                            EndX - PaddingX, 
                            StartY - PaddingY, 
                            EndY - PaddingY
                        );
                    }
                }

                // Assign the errors for this input features
                batchErrors[featureIndex] = inputError;
            });

            // Assign the errors for the input features into the batch
            input_errors[batchIndex] = new FeatureSet<float>(batchErrors);
        });
        

        // Pass errors along for next layer
        return new BackpropagationReturns (
            new BatchedFeatureSet<float>(input_errors),
            null
        );
    }

    protected abstract void Backpropagate(Matrix<float> inputError, Matrix<float> input, float error, int filterSize, int startX, int endX, int startY, int endY);
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
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    /// <param name="padding">input padding</param>
    public LocalMaxPoolingLayer(Shape3D input_size, int size, int stride, int padding) : base(input_size, size, stride, padding) { }

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

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    /// <param name="paddingX">horizontal input padding</param>
    /// <param name="paddingY">vertical input stride</param>
    public LocalMaxPoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY, int paddingX, int paddingY) : base(input_size, width, height, strideX, strideY, paddingX, paddingY) { }

    protected override float Accumulate(float current, float delta, int count) {
        if (count == 1)
            return delta;                   // First accumulated value
        return Math.Max(current, delta);    // Subsequent accumulated values
    }

    protected override float Aggregate(float current, int count){
        return current;
    }
    
   protected override void Backpropagate(Matrix<float> inputError, Matrix<float> input, float error, int filterSize, int startX, int endX, int startY, int endY) {
        var inputHeight = input.Rows;
        var inputWidth = input.Columns;
        int maxRow = startY, maxCol = startX; float maxVal = float.MinValue; // Values for max pooling
        for (int kr = startY; kr < endY; kr++) {
            if (kr < 0 || kr >= inputHeight)
                continue;

            for (int kc = startX; kc < endX; kc++) {
                if (kc < 0 || kc >= inputWidth)
                    continue;
                var value = (kr < 0 || kr >= inputHeight || kc < 0 || kc >= inputWidth) ? 0.0f :  input[kr, kc];

                // Compute; Assume max pooling (avg is different)
                if (value > maxVal) {
                    maxVal = value;
                    maxRow = kr;
                    maxCol = kc;
                }
            }
        }
        if (maxRow < 0 || maxRow >= inputHeight || maxCol < 0 || maxCol >= inputWidth)
            return;
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
    /// Create a pooling layer with a square filter
    /// </summary>
    /// <param name="size">width and height</param>
    /// <param name="stride">stride to apply the filter</param>
    /// <param name="padding">input padding</param>
    public LocalAvgPoolingLayer(Shape3D input_size, int size, int stride, int padding) : base(input_size, size, stride, padding) { }

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

    /// <summary>
    /// Create a pooling layer with a rectangular filter
    /// </summary>
    /// <param name="width">filter width</param>
    /// <param name="height">filter height</param>
    /// <param name="strideX">horizontal stride</param>
    /// <param name="strideY">vertical stride</param>
    /// <param name="paddingX">horizontal input padding</param>
    /// <param name="paddingY">vertical input stride</param>
    public LocalAvgPoolingLayer(Shape3D input_size, int width, int height, int strideX, int strideY, int paddingX, int paddingY) : base(input_size, width, height, strideX, strideY, paddingX, paddingY) { }


    protected override float Accumulate(float current, float delta, int count) {
        return current + delta;
    }

    protected override float Aggregate(float current, int count){
        return current / Math.Max(1, count);
    }

    protected override void Backpropagate(Matrix<float> inputError, Matrix<float> input, float error, int filterSize, int startX, int endX, int startY, int endY) {
        var inputHeight = input.Rows;
        var inputWidth = input.Columns;
        float errorContribution = error / Math.Max(1, filterSize); // Distribute the error
        for (int kr = startY; kr < endY; kr++) {
            if (kr < 0 || kr >= inputHeight)
                continue;

            for (int kc = startX; kc < endX; kc++) {
                if (kc < 0 || kc >= inputWidth)
                    continue;

                inputError[kr, kc] += errorContribution;            // Assign the error contribution to each element in the pooling region
            }   
        }
    }
}