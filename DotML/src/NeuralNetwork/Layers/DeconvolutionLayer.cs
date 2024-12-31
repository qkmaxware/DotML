using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network;

[Untested]
[WorkInProgress]
public class DeconvolutionLayer : ConvolutionalFeedforwardNetworkLayer {
    private ConvolutionFilter[] filters;
    public ReadOnlyCollection<ConvolutionFilter> Filters {get; init;}
    public Expansion Padding {get; init;} // Indicates if the output matrix size will be larger than the input, or the same size (kinda like inverse padding)
    public int StrideX {get; init;}
    public int StrideY {get; init;}

    public int FilterCount => filters.Length;

    private int filterRows;
    private int filterColumns;
    public int RowsPadding {get; init;}
    public int ColumnsPadding {get; init;}

    public DeconvolutionLayer(Shape3D input_size) : this(input_size, Expansion.Same, 1, 1, new ConvolutionFilter[] { new ConvolutionFilter(Kernels.RandomKernel(3)) }) { }

    public DeconvolutionLayer(Shape3D input_size, Expansion padding) : this(input_size, padding, 1, 1, new ConvolutionFilter[] { new ConvolutionFilter(Kernels.RandomKernel(3)) }) { }

    public DeconvolutionLayer(Shape3D input_size, Expansion padding, params ConvolutionFilter[] filters) : this(input_size, padding, 1, 1, filters) { }

    public DeconvolutionLayer(Shape3D input_size, Expansion padding, int stride, params ConvolutionFilter[] filters) : this(input_size, padding, stride, stride, filters) {}

    public DeconvolutionLayer(Shape3D input_size, Expansion padding, int strideX, int strideY, params ConvolutionFilter[] filters) {
        this.Padding            = padding;
        this.filters            = filters;
        this.Filters            = Array.AsReadOnly(this.filters);
        this.StrideX            = Math.Max(1, strideX);
        this.StrideY            = Math.Max(1, strideY);

        this.InputShape         = input_size;
        var inputRows           = InputShape.Rows;                                                    
        var inputColumns        = InputShape.Columns;                                                   
        this.filterRows         = filters.Select(f => f.Height).Max();                                                        // 3
        this.filterColumns      = filters.Select(f => f.Width).Max();                                                         // 3
        this.RowsPadding        = Padding == Expansion.Expand ? (filterRows - 1) / 2 : 0;       // Expand vs keep Same
        this.ColumnsPadding     = Padding == Expansion.Expand ? (filterColumns - 1) / 2 : 0;    // Expand vs keep Same


        OutputShape             = new Shape3D(
            channel:            filters.Length,
            rows:               (inputRows - 1) * StrideY + filterRows - 2 * RowsPadding,
            columns:            (inputColumns - 1) * StrideX + filterColumns - 2 * ColumnsPadding
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

    public override int TrainableParameterCount() {
        return Filters.Select(filter => filter.Select(kernel => kernel.Rows * kernel.Columns).Sum()).Sum() + FilterCount;
    }

    public Matrix<double> TransposeConvolve(Matrix<double> input, int channel) {
        var output = new double[input.Rows, input.Columns];

        // Calculate the input errors for each filter
        for (var filterIndex = 0; filterIndex < Filters.Count; filterIndex++) {
            var filter = filters[filterIndex];
            var filterRows = filter.Height;
            var filterColumns = filter.Width;
            var rows = input.Rows;
            var cols = input.Columns;
            var kernel = filter[channel];

            // Iterate over output errors to compute gradient
            for (int outY = 0; outY < rows; outY++) {
                var startY = outY * StrideY - RowsPadding;
                for (int outX = 0; outX < cols; outX++) {
                    var startX = outX * StrideX - ColumnsPadding;
                    // Place the error at the corresponding position in the input space
                    for (int ky = 0; ky < filterRows; ky++) {
                        var inY = startY + ky;
                        for (int kx = 0; kx < filterColumns; kx++) {
                            var inX = startX + kx;

                            if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                output[inY, inX] += input[outY, outX] * kernel[ky, kx];
                            }
                        }
                    }
                }
            }
        }

        return Matrix<double>.Wrap(output);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var channel_count = OutputShape.Channels;
        var outputs = new Matrix<double>[channel_count];

        Parallel.For(0, channel_count, channel => {
            outputs[channel] = TransposeConvolve(channels[channel], channel);
        });

        return new FeatureSet<double>(outputs);
    }

    public override void Visit(IConvolutionalLayerVisitor visitor) {
        throw new NotImplementedException();
    }

    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) {
        throw new NotImplementedException();
    }

    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) {
        throw new NotImplementedException();
    }
}