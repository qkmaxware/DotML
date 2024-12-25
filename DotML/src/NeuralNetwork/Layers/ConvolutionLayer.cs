using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Apply a convolution using the given kernel/filter
/// <see href="https://en.wikipedia.org/wiki/Convolutional_layer"/>
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
        var parameters = this.TrainableParameterCount();
        foreach (var filter in filters) {
            filter.Bias = initializer.RandomBias(this.InputShape.Count, this.OutputShape.Count, parameters);
            foreach (var kernel in filter) {
                var values = (double[,])kernel;

                for (var i = 0; i < values.GetLength(0); i++) {
                    for (var j = 0; j < values.GetLength(1); j++) {
                        values[i, j] = initializer.RandomWeight(this.InputShape.Count, this.OutputShape.Count, parameters);
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
            output[outY, outX] = total_sum + filter.Bias; //Missed this?
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

                // Set the output position's value
                output[outY, outX] = total_sum + filter.Bias; //Missed this?
            }
        }

        // Exit
        return Matrix<double>.Wrap(output);
    }

    public Matrix<double>[] Convolve(Matrix<double>[] inputs) {
        var filtersLength       = filters.Length;
        var output_list         = new Matrix<double>[filtersLength];

        Parallel.For(0, filtersLength, filterIndex => {
            var filter = filters[filterIndex];
            var output = Convolve(inputs, filter);
            output_list[filterIndex] = output;
        });

        return output_list;
    }

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) {
        var z = this.Convolve(inputs);
        return z;
    }

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}