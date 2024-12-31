using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Apply a convolution using the given kernel/filter
/// <see href="https://towardsdatascience.com/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503"/>
/// </summary>
[Untested()]
public class DepthwiseConvolutionLayer : ConvolutionalFeedforwardNetworkLayer {
    
    /// <summary>
    /// The filter to apply to the input channels
    /// </summary>
    public ConvolutionFilter Filter {get; set;}
    public Padding Padding {get; init;}
    public int StrideX {get; init;}
    public int StrideY {get; init;}

    private int filterRows;
    private int filterColumns;
    public int RowsPadding {get; init;}
    public int ColumnsPadding {get; init;}

    public DepthwiseConvolutionLayer(Shape3D input_size) : this(input_size, Padding.Same, 1, 1, new ConvolutionFilter(Kernels.RandomKernel(3))) { }

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding) : this(input_size, padding, 1, 1, new ConvolutionFilter(Kernels.RandomKernel(3)) ) { }

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding, ConvolutionFilter filter) : this(input_size, padding, 1, 1, filter) { }

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding, int stride, ConvolutionFilter filter) : this(input_size, padding, stride, stride, filter) {}

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding, int strideX, int strideY, ConvolutionFilter filter) {
        this.Filter = filter;
        this.Padding = padding;
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);

        if (input_size.Channels != Filter.Count)
            throw new ArgumentException($"Expecting {input_size.Channels} kernels (1 per input channel) but was only given {Filter.Count}.");

        // Note, this only works if FILTERS is FIXED!! which may not be true
        this.InputShape         = input_size;
        var inputRows           = InputShape.Rows;                                                      // 32
        var inputColumns        = InputShape.Columns;                                                   // 32
        this.filterRows         = filter.Height;                                                        // 3
        this.filterColumns      = filter.Width;                                                         // 3
        this.RowsPadding        = Padding == Padding.Same ? (filterRows - 1) / 2 : 0;                   // 1 
        this.ColumnsPadding     = Padding == Padding.Same ? (filterColumns - 1) / 2 : 0;                // 1

        OutputShape             = new Shape3D(
            channel: input_size.Channels,
            rows: (inputRows - filterRows + 2 * RowsPadding) / StrideY + 1,
            columns: (inputColumns - filterColumns + 2 * ColumnsPadding) / StrideX + 1
        );
    }

    public override void Initialize(IInitializer initializer) {
        var parameters = this.TrainableParameterCount();

        Filter.Bias = 0; // Unused //initializer.RandomBias(this.InputShape.Count, this.OutputShape.Count, parameters);
        foreach (var kernel in Filter) {
            var values = (double[,])kernel;

            for (var i = 0; i < values.GetLength(0); i++) {
                for (var j = 0; j < values.GetLength(1); j++) {
                    values[i, j] = initializer.RandomWeight(this.InputShape.Count, this.OutputShape.Count, parameters);
                }
            }
        }
    }

    public override int TrainableParameterCount() => Filter.Select(kernel => kernel.Rows * kernel.Columns).Sum(); // Hmm what about biases?

    public Matrix<double> Convolve(Matrix<double> input, Matrix<double> kernel) {
        // Compute output size taking into account padding & stride                                     // Same
        var filterRows          = this.filterRows;                                                        // 3
        var filterColumns       = this.filterColumns;                                                         // 3
        var paddingRows         = this.RowsPadding;                   // 1 
        var paddingColumns      = this.ColumnsPadding;                // 1
        var outputRows          = this.OutputShape.Rows;             // 32 (good)
        var outputColumns       = this.OutputShape.Columns;    // 32 (good)
        var stridex             = this.StrideX;
        var stridey             = this.StrideY;

        // Allocate output
        var output = new double[outputRows, outputColumns];

        // Slide over output
        for (var outY = 0; outY < outputRows; outY++) {
            var startY = outY * stridey - paddingRows;
            for (var outX = 0; outX < outputColumns; outX++) {
                var total_sum = 0.0;
                var startX = outX * stridex - paddingColumns;

                // Compute value by applying the kernel to the input region associated with this output
                for (int ky = 0; ky < filterRows; ky++) {
                    var inY = startY + ky;
                    for (int kx = 0; kx < filterColumns; kx++) {
                        var inX = startX + kx;
                        
                        total_sum += input[inY, inX] * kernel[ky, kx];
                    }
                }

                // Set the output position's value
                output[outY, outX] = total_sum;
            }
        }

        // Exit
        return Matrix<double>.Wrap(output);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var len = channels.Channels;
        var outputs = new Matrix<double>[len];

        Parallel.For(0, len, i => {
            var channel = channels[i];
            var kernel = Filter[i];

            outputs[i] = Convolve(channel, kernel);
        });

        return (FeatureSet<double>)outputs;
    }

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}