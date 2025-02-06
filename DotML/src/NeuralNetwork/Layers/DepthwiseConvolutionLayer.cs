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

    public Matrix<double> Convolve(Matrix<double> input, Matrix<double> kernel, double bias) {
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
        var output = new Matrix<double>(outputRows, outputColumns, bias);

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
        return output;
    }

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

        public BatchedFeatureSet<double> FilterKernelGradients;
        public Vec<double> BiasGradients;

        public Gradients(BatchedFeatureSet<double> filter, Vec<double> bias) {
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
                    for (var i = 0; i < kmatrix.Size; i++) { // Kernel value
                        kmatrix[i] = handler(parameter_index++, kmatrix[i]);
                    }
                }
            }

            for (var f = 0; f < BiasGradients.Dimensionality; f++) {
                BiasGradients[f] = handler(parameter_index++, BiasGradients[f]);
            } 
        } 
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var dW = BackpropagateWrtWeights(args.InputBatch, args.OutputErrors);
        var dB = BackpropagateWrtBias   (args.OutputErrors);
        //var dX = BackpropagateWrtInput  (args.InputBatch, args.OutputErrors);
        var dX = DepthwiseTransposeConvolve2(args.InputBatch, args.OutputBatch, args.OutputErrors);

        return new BackpropagationReturns(
            dX,
            new Gradients(
                dW,
                dB
            )
        );
    }

    private BatchedFeatureSet<double> BackpropagateWrtWeights(BatchedFeatureSet<double> X, BatchedFeatureSet<double> dY) {
        // This is essentially the convolution of the input region X with the error term from the output Y for each filter.
        var (batches, in_channels, in_rows, in_columns) = X.Shape;

        var results = new BatchedFeatureSet<double>(new Shape4D(in_channels, 1, filterRows, filterColumns)); // output channels = input channels, kernels, kernel rows, kernel columns

        for (var channel = 0; channel < in_channels; channel++) {
            // For each channel, convolve X with Y and sum over all batches
            var channel_matrix = results[channel, 0];

            for(var batch = 0; batch < batches; batch++) {
                channel_matrix.AddWithInplace(
                    X[batch, channel].Convolve(dY[batch, channel], strideX: StrideX, strideY: StrideY, paddingX: 0, paddingY: 0)
                );
            }
        }

        return results;
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

    private BatchedFeatureSet<double> BackpropagateWrtInput(BatchedFeatureSet<double> X, BatchedFeatureSet<double> dY) {
        var (batches, in_channels, in_rows, in_columns) = X.Shape;
        var (_, _, out_rows, out_columns) = dY.Shape;

        var results = new BatchedFeatureSet<double>(X.Shape);

        for (var batch_index = 0; batch_index < batches; batch_index++) {
            for (var channel = 0; channel < in_channels; channel++) {
                var dX_matrix = results[batch_index, channel];
                var dY_matrix = dY[batch_index, channel];
                var kernel = filters[channel][0];

                for (var row = 0; row < out_rows; row++) {
                    for (var col = 0; col < out_columns; col++) {
                        for (var krow = 0; krow < kernel.Rows; krow++) {
                            for (var kcol = 0; kcol < kernel.Columns; kcol++) {
                                dX_matrix[row + krow, col + kcol] += dY_matrix[row, col] * kernel[krow, kcol];
                            }
                        }
                    }
                }
            }
        }

        return results;
    }

    private BatchedFeatureSet<double> DepthwiseTransposeConvolve2(BatchedFeatureSet<double> X, BatchedFeatureSet<double> Y , BatchedFeatureSet<double> dY) {
        // dx = dy_0 * w'
        var (batch_count, channel_count, input_height, input_width) = X.Shape;
        var (_, _, output_height, output_width) = Y.Shape;
        var stride_x = this.StrideX;
        var stride_y = this.StrideY;

        var padding_rows = this.RowsPadding;
        var padding_cols = this.ColumnsPadding;

        var input_to_output_padding_rows = (input_height - output_height) / 2;
        var input_to_output_padding_cols = (input_width - output_width) / 2;

        var result_features = new FeatureSet<double>[batch_count];

        Parallel.For(0, batch_count, batchIndex => {
            var batch_inputs = X[batchIndex];
            var batch_outputs = Y[batchIndex];
            var batch_errors = dY[batchIndex];

            var batch_features = new Matrix<double>[channel_count];

            var filters = this.filters;
            var filter_width = filterColumns;
            var filter_height = filterRows;

            var filter_height_m1 = filter_height - 1;
            var filter_width_m1 = filter_width - 1;

            var input_padding_rows = (filter_height_m1) / 2;
            var input_padding_cols = (filter_width_m1) / 2;

            var input_width_padded = input_width + 2 * input_padding_rows;
            var input_height_padded = input_height + 2 * input_padding_rows;

            for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                var input_error = new Matrix<double>(input_height, input_width);
                    
                var kernel = filters[channelIndex][0];
                var error = batch_errors[channelIndex];
                
                // Slide kernel over input
                for (var inputY = 0; inputY < input_height_padded; inputY++) {
                    var outY = (inputY - input_padding_rows - input_to_output_padding_rows) / stride_y;  // This assumes the output is "centered" in the middle of the input

                    for (var inputX = 0; inputX < input_width_padded; inputX++) {
                        var outX = (inputX - input_padding_cols - input_to_output_padding_cols) / stride_x; // This assumes the output is "centered" in the middle of the input

                        var sum = 0.0;
                        for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                            //var inv_kernelY = filter_height_m1 - kernelY;
                            var outY_plus_kernel = outY + kernelY;
                            if (outY_plus_kernel < 0 || outY_plus_kernel >= output_height) continue; // Skip out-of-bounds rows
                            
                            for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                //var inv_kernelX = filter_width_m1 - kernelX;
                                var outX_plus_kernel = outX + kernelX;
                                if (outX_plus_kernel < 0 || outX_plus_kernel >= output_width) continue; // Skip out-of-bounds columns

                                var kernel_value = kernel[kernelY, kernelX]; // Kernel is NOT inverted here according to chatgpt
                                var output_value = error[outY_plus_kernel, outX_plus_kernel];
                                var result = kernel_value * output_value;

                                sum += result;      
                            }
                        }

                        if (inputX >= 0 && inputX < input_width && inputY >= 0 && inputY < input_height)
                            input_error[inputY, inputX] += sum;
                    }
                }

                batch_features[channelIndex] = input_error;
            } 
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        });

        return new BatchedFeatureSet<double>(result_features);
    }


    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}