using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Apply a convolution using the given kernel/filter
/// </summary>
public class ConvolutionLayer : ConvolutionalFeedforwardNetworkLayer {
    private ConvolutionFilter[] filters;
    public ReadOnlyCollection<ConvolutionFilter> Filters {get; init;}
    public Padding Padding {get; init;}
    public ActivationFunction ActivationFunction {get; set;} = Identity.Instance;
    public int StrideX {get; set;}
    public int StrideY {get; set;}

    public int FilterCount => filters.Length;
    public override int InputCount => filters.Select(kernels => kernels.Count).Max();
    public override int OutputCount => FilterCount;
    public override int NeuronCount => filters.SelectMany(kernels => kernels.Select(kernel => kernel.Size)).Sum();

    public ConvolutionLayer() {
        this.Padding = Padding.Same;
        this.StrideX = 1;
        this.StrideY = 1;
        this.filters = new ConvolutionFilter[] { new ConvolutionFilter(Kernels.RandomKernel(3)) };
        this.Filters = Array.AsReadOnly(this.filters);
    }

    public ConvolutionLayer(Padding padding) {
        this.Padding = padding;
        this.StrideX = 1;
        this.StrideY = 1;
        this.filters = new ConvolutionFilter[] { new ConvolutionFilter(Kernels.RandomKernel(3)) };
        this.Filters = Array.AsReadOnly(this.filters);
    }

    public ConvolutionLayer(Padding padding, params ConvolutionFilter[] filters) {
        this.Padding = padding;
        this.filters = filters;
        this.Filters = Array.AsReadOnly(this.filters);
        this.StrideX = 1;
        this.StrideY = 1;
    }

    public ConvolutionLayer(Padding padding, int stride, params ConvolutionFilter[] filters) {
        this.Padding = padding;
        this.filters = filters;
        this.Filters = Array.AsReadOnly(this.filters);
        this.StrideX = Math.Max(1, stride);
        this.StrideY = Math.Max(1, stride);
    }

    public ConvolutionLayer(Padding padding, int strideX, int strideY, params ConvolutionFilter[] filters) {
        this.Padding = padding;
        this.filters = filters;
        this.Filters = Array.AsReadOnly(this.filters);
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);
    }

    public override void Initialize(IInitializer initializer) {
        foreach (var filter in filters) {
            filter.Bias = initializer.RandomBias(filters.Length);
            foreach (var kernel in filter) {
                var values = (double[,])kernel;
                var parameters = Math.Max(values.GetLength(0), values.GetLength(1));

                for (var i = 0; i < values.GetLength(0); i++) {
                    for (var j = 0; j < values.GetLength(1); j++) {
                        values[i, j] = initializer.RandomWeight(parameters);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public override int TrainableParameterCount() => Filters.Select(filter => filter.Select(kernel => kernel.Rows * kernel.Columns).Sum()).Sum() + FilterCount;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool even(int i) {
        return (i & 1) == 0;
    }

    public Matrix<double> Convolve(Matrix<double>[] inputs, ConvolutionFilter filter) {
        // Compute output size taking into account padding & stride                                     // Same
        var inputRows           = inputs.Select(x => x.Rows).Max();                                     // 32
        var inputColumns        = inputs.Select(x => x.Columns).Max();                                  // 32
        var filterRows          = filter.Height;                                                        // 3
        var filterColumns       = filter.Width;                                                         // 3
        var paddingRows         = Padding == Padding.Same ? (filterRows - 1) / 2 : 0;                   // 1 
        var paddingColumns      = Padding == Padding.Same ? (filterColumns - 1) / 2 : 0;                // 1
        var outputRows          = (inputRows - filterRows + 2 * paddingRows) / StrideY + 1;             // 32 (good)
        var outputColumns       = (inputColumns - filterColumns + 2 * paddingColumns) / StrideX + 1;    // 32 (good)

        // Allocate output
        var output = new double[outputRows, outputColumns];

        // Slide over output
        for (var outY = 0; outY < outputRows; outY++) {
            for (var outX = 0; outX < outputColumns; outX++) {
                double total_sum = 0.0;

                for (var inputIndex = 0; inputIndex < inputs.Length; inputIndex++) {
                    var input   = inputs[inputIndex];
                    var kernel  = filter[inputIndex];

                    // Compute value by applying the kernel to the input region associated with this output
                    double sum = 0.0;
                    for (int ky = 0; ky < filterRows; ky++) {
                        for (int kx = 0; kx < filterColumns; kx++) {
                            var inY = outY * StrideY - paddingRows + ky;
                            var inX = outX * StrideX - paddingColumns + kx;
                            
                            if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                sum += input[inY, inX] * kernel[ky, kx];
                            }
                        }
                    }
                    
                    total_sum += sum;
                }

                // Set the ouput position's value
                output[outY, outX] = total_sum;
            }
        }

        // Exit
        return Matrix<double>.Wrap(output);
    }

    public Matrix<double>[] Convolve(Matrix<double>[] inputs) {
        var output_list         = new Matrix<double>[filters.Length];

        for (var filterIndex = 0; filterIndex < filters.Length; filterIndex++) {
            var filter = filters[filterIndex];
            var output = Convolve(inputs, filter);
            output_list[filterIndex] = output;
        }

        return output_list;
    }

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) {
        var z = this.Convolve(inputs);

        var a = new Matrix<double>[z.Length];
        for (var i = 0; i < z.Length; i++) {
            a[i] = this.ActivationFunction.Invoke(z[i]);
        }
        return a;
    }

    /*public Matrix<double>[] OldEvaluateSync(Matrix<double>[] inputs) {
        // Each filter generates 1 output, each filter kernel is applied to a different input channel
        var output_list         = new Matrix<double>[filters.Length];
        var channels            = inputs.Length;
        var inputRows           = inputs.Select(x => x.Rows).Max();
        var inputColumns        = inputs.Select(x => x.Columns).Max();

        for (var filterIndex = 0; filterIndex < filters.Length; filterIndex++) {
            var filter = filters[filterIndex];

            var maxFilterWidth      = filter.Width;
            var maxFilterHeight     = filter.Height;
            var maxFilterCenterX    = (maxFilterWidth - 1) >> 1; 
            var maxFilterCenterY    = (maxFilterHeight - 1) >> 1;
            // TODO update output size to account for Stride.... double check this since it seems like SAME isn't actually doing the same size
            var outputRows          = (inputRows - maxFilterHeight + (Padding == Padding.Same ? maxFilterHeight - 1 : 0)) / StrideY + 1;
            var outputColumns       = (inputColumns - maxFilterWidth + (Padding == Padding.Same ? maxFilterWidth - 1 : 0)) / StrideX + 1;

            var output = new Matrix<double>(outputRows, outputColumns);
            var values = (double[,])output;

            // Slide over input (centered about the middle of the kernel)
            var ioffsetX        = Padding == Padding.Same ? 0 : maxFilterCenterX;
            var ioffsetY        = Padding == Padding.Same ? 0 : maxFilterCenterY;
            for (var cy = 0; cy < outputRows; cy++) {
                for (var cx = 0; cx < outputColumns; cx++) {
                    // Compute the dot product of the kernel over the input
                    double total_sum = 0.0;
                    for (var channel = 0; channel < channels; channel++) {
                        var input = inputs[channel];
                        var kernel = filter[channel];

                        double sum = 0.0;
                        for (int ky = 0; ky < maxFilterHeight; ky++) {
                            for (int kx = 0; kx < maxFilterWidth; kx++) {
                                var inX = cx * StrideX + ioffsetX + kx - maxFilterCenterX;
                                var inY = cy * StrideY + ioffsetY + ky - maxFilterCenterY;
                                sum += input[inY, inX] * kernel[ky, kx];
                            }
                        }
                        total_sum += sum;
                    }
                    values[cy, cx] = total_sum + filter.Bias;
                }
            }
            
            output_list[filterIndex] = ActivationFunction.Invoke(output);
        }

        return output_list;
    }*/

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}