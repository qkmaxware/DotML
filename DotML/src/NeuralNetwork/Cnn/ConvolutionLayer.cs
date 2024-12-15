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
    public int StrideX {get; init;}
    public int StrideY {get; init;}

    public int FilterCount => filters.Length;
    
    private int filterRows;
    private int filterColumns;
    public int RowsPadding {get; init;}
    public int ColumnsPadding {get; init;}

    public ConvolutionLayer(Shape3D input_size) : this(input_size, Padding.Same, 1, 1, new ConvolutionFilter[] { new ConvolutionFilter(Kernels.RandomKernel(3)) }) { }

    public ConvolutionLayer(Shape3D input_size, Padding padding) : this(input_size, padding, 1, 1, new ConvolutionFilter[] { new ConvolutionFilter(Kernels.RandomKernel(3)) }) { }

    public ConvolutionLayer(Shape3D input_size, Padding padding, params ConvolutionFilter[] filters) : this(input_size, padding, 1, 1, filters) { }

    public ConvolutionLayer(Shape3D input_size, Padding padding, int stride, params ConvolutionFilter[] filters) : this(input_size, padding, stride, stride, filters) {}

    public ConvolutionLayer(Shape3D input_size, Padding padding, int strideX, int strideY, params ConvolutionFilter[] filters) {
        this.Padding = padding;
        this.filters = filters;
        this.Filters = Array.AsReadOnly(this.filters);
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);

        // Note, this only works if FILTERS is FIXED!! which may not be true
        this.InputShape = input_size;
        var inputRows           = InputShape.Rows;                                                      // 32
        var inputColumns        = InputShape.Columns;                                                   // 32
        this.filterRows          = filters.Select(f => f.Height).Max();                                                        // 3
        this.filterColumns       = filters.Select(f => f.Width).Max();                                                         // 3
        this.RowsPadding         = Padding == Padding.Same ? (filterRows - 1) / 2 : 0;                   // 1 
        this.ColumnsPadding      = Padding == Padding.Same ? (filterColumns - 1) / 2 : 0;                // 1


        OutputShape             = new Shape3D(
            channel: filters.Length, 
            rows: (inputRows - filterRows + 2 * RowsPadding) / StrideY + 1,
            columns: (inputColumns - filterColumns + 2 * ColumnsPadding) / StrideX + 1
        );
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

    public Matrix<double> ConvolveParallel(Matrix<double>[] inputs, ConvolutionFilter filter) {
        // Compute output size taking into account padding & stride                                     // Same
        var filterRows          = this.filterRows;                                                        // 3
        var filterColumns       = this.filterColumns;                                                         // 3
        var paddingRows         = this.RowsPadding;                   // 1 
        var paddingColumns      = this.ColumnsPadding;                // 1
        var outputRows          = this.OutputShape.Rows;             // 32 (good)
        var outputColumns       = this.OutputShape.Columns;    // 32 (good)
        var inputLength         = inputs.Length;
        var stridex             = this.StrideX;
        var stridey             = this.StrideY;

        // Allocate output
        var output = new double[outputRows, outputColumns];

        // Slide over output
        Parallel.For(0, outputRows * outputColumns, outIndex => {
            var outY = outIndex / outputColumns;
            var outX = outIndex % outputColumns;
            var startY = outY * stridey - paddingRows;

            var total_sum = 0.0;
            var startX = outX * stridex - paddingColumns;

            for (var inputIndex = 0; inputIndex < inputLength; inputIndex++) {
                var input   = inputs[inputIndex];
                var kernel  = filter[inputIndex];

                // Compute value by applying the kernel to the input region associated with this output
                for (int ky = 0; ky < filterRows; ky++) {
                    var inY = startY + ky;
                    for (int kx = 0; kx < filterColumns; kx++) {
                        var inX = startX + kx;
                        
                        total_sum += input[inY, inX] * kernel[ky, kx];
                    }
                }
            }

            // Set the ouput position's value
            output[outY, outX] = total_sum;
        });

        // Exit
        return Matrix<double>.Wrap(output);
    }

    public Matrix<double> Convolve(Matrix<double>[] inputs, ConvolutionFilter filter) {
        // Compute output size taking into account padding & stride                                     // Same
        var filterRows          = this.filterRows;                                                        // 3
        var filterColumns       = this.filterColumns;                                                         // 3
        var paddingRows         = this.RowsPadding;                   // 1 
        var paddingColumns      = this.ColumnsPadding;                // 1
        var outputRows          = this.OutputShape.Rows;             // 32 (good)
        var outputColumns       = this.OutputShape.Columns;    // 32 (good)
        var inputLength         = inputs.Length;
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

                for (var inputIndex = 0; inputIndex < inputLength; inputIndex++) {
                    var input   = inputs[inputIndex];
                    var kernel  = filter[inputIndex];

                    // Compute value by applying the kernel to the input region associated with this output
                    for (int ky = 0; ky < filterRows; ky++) {
                        var inY = startY + ky;
                        for (int kx = 0; kx < filterColumns; kx++) {
                            var inX = startX + kx;
                            
                            total_sum += input[inY, inX] * kernel[ky, kx];
                        }
                    }
                }

                // Set the ouput position's value
                output[outY, outX] = total_sum;
            }
        }

        // Exit
        return Matrix<double>.Wrap(output);
    }

    public Matrix<double>[] Convolve(Matrix<double>[] inputs) {
        var filtersLength       = filters.Length;
        var output_list         = new Matrix<double>[filtersLength];

        for (var filterIndex = 0; filterIndex < filtersLength; filterIndex++) {
            var filter = filters[filterIndex];
            var output = ConvolveParallel(inputs, filter);
            output_list[filterIndex] = output;
        }

        return output_list;
    }

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) {
        var z = this.Convolve(inputs);
        return z;
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