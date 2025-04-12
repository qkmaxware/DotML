using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Apply a convolution using the given kernel/filter
/// <see href="https://towardsdatascience.com/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503"/>
/// </summary>
[Untested()]
public class DepthwiseConvolutionLayer : FeedforwardNetworkLayer {
    
    /// <summary>
    /// The filter to apply to the input channels
    /// </summary>
    private ConvolutionFilter[] filters;
    public ReadOnlyCollection<ConvolutionFilter> Filters {get; init;}
    public Padding Padding {get; init;}
    public int StrideX {get; init;}
    public int StrideY {get; init;}

    private int filterRows;
    private int filterColumns;
    public int RowsPadding {get; init;}
    public int ColumnsPadding {get; init;}

    public DepthwiseConvolutionLayer(Shape3D input_size) : this(input_size, Padding.Same, 1, 1, [new ConvolutionFilter(Kernels.RandomKernel(3))]) { }

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding) : this(input_size, padding, 1, 1, [new ConvolutionFilter(Kernels.RandomKernel(3))] ) { }

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding, ConvolutionFilter[] filter) : this(input_size, padding, 1, 1, filter) { }

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding, int stride, ConvolutionFilter[] filter) : this(input_size, padding, stride, stride, filter) {}

    public DepthwiseConvolutionLayer(Shape3D input_size, Padding padding, int strideX, int strideY, ConvolutionFilter[] filter) {
        this.filters = filter;
        this.Filters = this.filters.AsReadOnly();
        this.Padding = padding;
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);

        if (input_size.Channels != filters.Length)
            throw new ArgumentException($"Expecting {input_size.Channels} filters but was given {filters.Length}.");
        foreach (var f in filters) {
            if (f.Count != 1)
                throw new ArgumentException($"Only a single kernel allowed per filter for depthwise convolution, {f.Count} kernels found.");
        }

        // Note, this only works if FILTERS is FIXED!! which may not be true
        this.InputShape         = input_size;
        var inputRows           = InputShape.Rows;                                                      // 32
        var inputColumns        = InputShape.Columns;                                                   // 32
        this.filterRows         = filter.Select(x => x.Height).Max();                                                        // 3
        this.filterColumns      = filter.Select(x => x.Width).Max();                                                         // 3
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

        foreach (var f in filters) {
            f.Bias = initializer.RandomBias(this.InputShape.Count, this.OutputShape.Count, parameters);

            foreach (var kernel in f) {
                var self = kernel;
                for (var i = 0; i < self.Rows; i++) {
                    for (var j = 0; j < self.Columns; j++) {
                        self[i, j] = initializer.RandomWeight(this.InputShape.Count, this.OutputShape.Count, parameters);
                    }
                }
            }
        }
    }

    public override int TrainableParameterCount() => filters.Select(filter => filter.Select(kernel => kernel.Rows * kernel.Columns).Sum()).Sum() + filters.Length; 

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var len = channels.Channels;
        var outputs = new Matrix<double>[len];

        for (var i = 0; i < len; i++) {
            var channel = channels[i];
            var filter = filters[i];
            var kernel = filter[0];

            var result = channel.Convolve(kernel, StrideX, StrideY, ColumnsPadding, RowsPadding, bias: filter.Bias);
            outputs[i] = result;
        }

        return (FeatureSet<double>)outputs;
    }

    public class Gradients : LayerGradients {
        private ConvolutionFilter[] Filters;
        public BatchedFeatureSet<double> FilterKernelGradients;
        public Vec<double> BiasGradients;

        public Gradients(ConvolutionFilter[] filters, BatchedFeatureSet<double> filter, Vec<double> bias) {
            this.Filters = filters;
            this.FilterKernelGradients = filter;
            this.BiasGradients = bias;
        }

        public override void Clip(double weight_threshold, double bias_threshold) {
            ClipBatch(FilterKernelGradients, weight_threshold);
            ClipVector(BiasGradients, bias_threshold);
        }

        public override void Apply(GradientTransformationHandler handler) {
            int parameter_index = 0; // Keep track of the parameter index
            for (var b = 0; b < FilterKernelGradients.Batches; b++) { // Filter
                for (var f = 0; f < FilterKernelGradients.Channels; f++) { // Kernel
                    var kmatrix = FilterKernelGradients[b][f];
                    var paramk = Filters[b][f];
                    for (var i = 0; i < kmatrix.Size; i++) { // Kernel value
                        kmatrix[i] = handler(parameter_index++, paramk[i], kmatrix[i]);
                    }
                }
            }

            for (var f = 0; f < BiasGradients.Dimensionality; f++) {
                BiasGradients[f] = handler(parameter_index++, Filters[f].Bias, BiasGradients[f]);
            } 
        } 
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var filter_shape = new Shape4D(filters.Length, 1, filterRows, filterColumns);
        var dW = BackpropagateWrtWeights(args.InputBatch, args.OutputErrors, filter_shape);
        var dB = BackpropagateWrtBias   (args.OutputErrors);
        var dX = BackpropagateWrtInput  (args.InputBatch, args.OutputBatch, args.OutputErrors);

        return new BackpropagationReturns(
            dX,
            new Gradients(
                this.filters,
                dW,
                dB
            )
        );
    }

    public override void SubtractGradients(LayerGradients? gradients) {
        if (gradients is null || gradients is not Gradients grads)
            throw new ArgumentException(nameof(gradients));

        for (var f = 0; f < filters.Length; f++) {
            var filter = filters[f];
            for (var k = 0; k < filter.Count; k++) {
                var kernel = filter[k];
                kernel.SubtractWithInplace(grads.FilterKernelGradients[f, k]);
            }
        } 

        for (var f = 0; f < filters.Length; f++) {
            var filter = filters[f];
            filter.Bias -= grads.BiasGradients[f];
        }  
    }

    private BatchedFeatureSet<double> BackpropagateWrtWeights(BatchedFeatureSet<double> X, BatchedFeatureSet<double> dY, Shape4D filter_shape) {
        var (batch_size, in_channels, height, width) = X.Shape;
        var (_, _, output_height, output_width) = dY.Shape;
        var (filter_count, _, filter_height, filter_width) = filter_shape;

        var dW = new BatchedFeatureSet<double>(filter_shape); // (filter_count, 1, filter_height, filter_width)

        for (var oY = 0; oY < output_height; oY++) {
            for (var oX = 0; oX < output_width; oX++) {
                // Region on the input which was used to compute this value on the output
                var x_start = oX * StrideX - ColumnsPadding;
                var x_end = x_start + filter_width;
                var y_start = oY * StrideY - RowsPadding;
                var y_end = y_start + filter_height;

                for (var batchIndex = 0; batchIndex < batch_size; batchIndex++) {
                    for (var filterIndex = 0; filterIndex < filter_count; filterIndex++) {
                        var grad = dY[batchIndex, filterIndex, oY, oX];

                        var dk = dW[filterIndex, 0];
                        var x = X[batchIndex, filterIndex];

                        for (int iY = y_start, ky = 0; iY < y_end; iY++, ky++) {
                            if (iY < 0 || iY >= height)
                                continue;

                            for (int iX = x_start, kx = 0; iX < x_end; iX++, kx++) {
                                if (iX < 0 || iX >= width)
                                    continue;

                                dk[ky, kx] += grad * x[iY, iX];
                            }
                        }
                    }
                }
            }
        }

        return dW;
    }

    private Vec<double> BackpropagateWrtBias(BatchedFeatureSet<double> dY) {
        var (batches, in_channels, _, _) = dY.Shape;
        var filterCount = this.filters.Length;

        Vec<double> results = new Vec<double>(filterCount);

        for (var filterIndex = 0; filterIndex < filterCount; filterIndex++) {
            // Sum of all elements over all spatial dimensions in dY
            results[filterIndex] = dY.Select(batch => batch[filterIndex].Sum()).Sum(); // Sum over batches, rows, columns of output
        }

        return results;
    }

    private BatchedFeatureSet<double> BackpropagateWrtInput(BatchedFeatureSet<double> X, BatchedFeatureSet<double> Y, BatchedFeatureSet<double> dY) {
        // Input Gradient
        // To compute the gradients w.r.t. the input (dinput), you perform a convolution of dY with the filter weights, flipping them. 
        // This is the same process used to calculate the forward pass convolution but with flipped weights
        // ---------------------------------------------------------------
        var (batch_size, in_channels, out_rows, out_columns) = X.Shape; // In and out rows/columns flipped here since the "input" is dY and the output is "dX"
        var (_, out_channels, in_rows, in_columns) = dY.Shape;
        var (filter_count, kernel_height, kernel_width) = (this.filters.Length, this.filterRows, this.filterColumns);
        var kernel_rows_m1 = kernel_height - 1;
        var kernel_cols_m1 = kernel_width - 1;

        var dX = new BatchedFeatureSet<double>(X.Shape);
        const bool flip_kernel = false;
        
        // How it worked.
        // Each filter was an output channel
        // Each kernel applied to a single input
        // So to go backwards we need to take the output from each filter and distribute it with each kernel back to the associated input
        for (var batch = 0; batch < batch_size; batch++) {
            for (var output_index = 0; output_index < out_channels; output_index++) {
                var filter = this.filters[output_index];
                var output = dY[batch, output_index];

                var kernel = filter[0];
                var result = dX[batch, output_index];

                // --------------------------------------
                // COPIED FROM Matrix<double>.TransposeConvolve();
                // --------------------------------------
                for (var r = 0; r < in_rows; r++) {
                    var region_start_y = r * StrideY - RowsPadding;
                    var region_end_y = region_start_y + kernel_height;

                    for (var c = 0; c < in_columns; c++) {
                        var region_start_x = c * StrideX - ColumnsPadding;
                        var region_end_x = region_start_x + kernel_width;

                        var i = output[r, c];

                        for (int out_y = region_start_y, ky = 0; out_y < region_end_y; out_y++, ky++) {
                            if (out_y < 0 || out_y >= out_rows)
                                continue;

                            for (int out_x = region_start_x, kx = 0; out_x < region_end_x; out_x++, kx++) {
                                if (out_x < 0 || out_x >= out_columns)
                                    continue;

                                if (flip_kernel) {
                                    result[out_y, out_x] += i * kernel[kernel_rows_m1 - ky, kernel_cols_m1 - kx];
                                } else {
                                    result[out_y, out_x] += i * kernel[ky, kx];
                                }
                            }
                        }
                    }
                }
                // --------------------------------------
            }
        }

        return new BatchedFeatureSet<double>(dX);
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);
    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}