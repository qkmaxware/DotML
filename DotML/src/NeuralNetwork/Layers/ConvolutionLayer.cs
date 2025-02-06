using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Apply a convolution using the given kernel/filter
/// <see href="https://en.wikipedia.org/wiki/Convolutional_layer"/>
/// </summary>
public class ConvolutionLayer : FeedforwardNetworkLayer {
    private ConvolutionFilter[] filters;
    public ReadOnlyCollection<ConvolutionFilter> Filters {get; init;}
    public Padding Padding {get; init;}
    public int StrideX {get; init;}
    public int StrideY {get; init;}

    public int FilterCount => filters.Length;

    public Shape4D FilterShape => new Shape4D(FilterCount, this.InputShape.Channels, filters[0].Height, filters[0].Width);
    
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
                var self = kernel;
                
                for (var i = 0; i < self.Rows; i++) {
                    for (var j = 0; j < self.Columns; j++) {
                        self[i, j] = initializer.RandomWeight(this.InputShape.Count, this.OutputShape.Count, parameters);
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
    private Matrix<double>[] Convolve(Matrix<double>[] inputs) {
        var filtersLength       = filters.Length;
        var output_list         = new Matrix<double>[filtersLength];

        for (var filterIndex = 0; filterIndex < filtersLength; filterIndex++) {
            var filter = filters[filterIndex];
            var output = Matrix<double>.ConvolveEach(
                inputs, filter, 
                strideX: StrideX, strideY: StrideY, 
                paddingX: ColumnsPadding, paddingY: RowsPadding,
                bias: filter.Bias
            );
            output_list[filterIndex] = output;
        }

        return output_list;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) {
        return new FeatureSet<double>(this.Convolve((Matrix<double>[])inputs));
    }

    public class Gradients : LayerGradients {
        public BatchedFeatureSet<double> FilterKernelGradients;
        public Vec<double> BiasGradients;

        public Gradients(BatchedFeatureSet<double> weights, Vec<double> bias) {
            this.FilterKernelGradients = weights;
            this.BiasGradients = bias;
        }

        public override void Clip(double weight_threshold, double bias_threshold) {
            base.ClipBatch(FilterKernelGradients, weight_threshold);
            base.ClipVector(BiasGradients, bias_threshold);
        }

        public override void Apply(GradientTransformationHandler handler) {
            int index = 0;
            for (var m = 0; m < FilterKernelGradients.Batches; m++) {
                var filter = FilterKernelGradients[m];
                for (var k = 0; k < filter.Channels; k++) {
                    var matrix = filter[k];
                    for (var r = 0; r < matrix.Rows; r++) {
                        for (var c = 0; c < matrix.Columns; c++) {
                            matrix[r,c] = handler(index++, matrix[r,c]);
                        }
                    }
                }
            }

            for (var i = 0; i < BiasGradients.Dimensionality; i++) {
                BiasGradients[i] = handler(index++, BiasGradients[i]);
            }
        }
    }
    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var filter_shape = FilterShape;
        var dW = BackpropagateWrtWeights(args.InputBatch, args.OutputErrors, filter_shape);
        var dB = BackpropagateWrtBias   (args.OutputErrors);
        var dX = TransposeConvolve2(this, args);
        // TODO fix this
        //var dX = BackpropagateWrtInput  (args.InputBatch, args.OutputErrors, filter_shape);

        return new BackpropagationReturns(
            dX,
            new Gradients (
                dW,
                dB
            )
        );
    }

    private BatchedFeatureSet<double> BackpropagateWrtWeights(BatchedFeatureSet<double> X, BatchedFeatureSet<double> dY, Shape4D filter_shape) {
        // Kernel/Weight Gradients
        // dW(filter, kernel, row, col) = dy(filter, i, j) * input(c, i+k-1, j+l-1)
        // ----------------------------------------------------------------------------
        var (batch_size, in_channels, height, width) = X.Shape;
        var (out_channels, _, filter_height, filter_width) = filter_shape;

        var dW = new BatchedFeatureSet<double>(filter_shape); // (out_channels, kernel_count, filter_height, filter_width)

        // Iterate through each batch and output channel
        for (var batch = 0; batch < batch_size; batch++) {
            for (var out_channel = 0; out_channel < out_channels; out_channel++) {
                // Apply a valid convolution of the input image and dY
                for (var in_channel = 0; in_channel < in_channels; in_channel++) {
                    dW[out_channel, in_channel].AddWithInplace(
                        X[batch, in_channel].Convolve(dY[batch, out_channel], strideX: StrideX, strideY: StrideY, paddingX: 0, paddingY: 0) // Maybe this is correct? This is a "valid" convolution as valid means no padding
                    );
                }
            }
        }

        return dW;
    }

    private Vec<double> BackpropagateWrtBias(BatchedFeatureSet<double> dY) {
        // Bias Gradients
        // dL/dB = dL/dY * dY/dB = dY * dY/dB
        // dY/dB = [1; ... ; 1] because b is constant wrt y
        // dL/dB = dL/dY = Sum over x,y of dY(x,y) given the above statement
        // -----------------------------------------------------------------------
        // Compute dL/dB by summing over batches, rows, and columns
        var featureCount = dY.Channels; // Should be equal to FilterCount
        var dB = new double[featureCount];
        for (var featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            // Compute sum 
            var sum = 0.0;
            for (var batchIndex = 0; batchIndex < dY.Batches; batchIndex++) {
                sum += dY[batchIndex, featureIndex].Sum();
            }

            // Apply biases
            dB[featureIndex] = sum;
        }

        return Vec<double>.Wrap(dB);
    }

    private BatchedFeatureSet<double> BackpropagateWrtInput(BatchedFeatureSet<double> X, BatchedFeatureSet<double> dY, Shape4D filter_shape) {
        // Input Gradient
        // To compute the gradients w.r.t. the input (dinput), you perform a convolution of dY with the filter weights, flipping them. 
        // This is the same process used to calculate the forward pass convolution but with flipped weights
        // ---------------------------------------------------------------
        var (batch_size, in_channels, in_rows, in_columns) = X.Shape;
        var (_, out_channels, out_rows, out_columns) = dY.Shape;
        var (_, _, kernel_height, kernel_width) = filter_shape;

        var dX = new FeatureSet<double>[batch_size];
        var flipped_filters = this.filters.Select(filter => new ConvolutionFilter(
            filter.Select(
                kernel => kernel.Mirror(x: true, y: true) // Flip the kernel
            ).ToArray()
        )).ToArray();
        var filtersLength = flipped_filters.Length;

        // DY should be zero-padded to match the shape of X
        var Sy = StrideY;
        var Sx = StrideX;

        for (var batch = 0; batch < batch_size; batch++) {
            var features = new Matrix<double>[in_channels];
            for (var feature = 0; feature < in_channels; feature++) {
                var matrix = new Matrix<double>(in_rows, in_columns);
                for (var i = 0; i < in_rows; i++) {
                    for (var j = 0; j < in_columns; j++) {
                        // Loop over Cout 
                        for (var channel = 0; channel < out_channels; channel++) {
                            var flipped_kernel = flipped_filters[channel][feature];
                            // Loop over Kh
                            for (var kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                                // Loop over Kw
                                for (var kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                                    var dX_n_cin_i_j = dY[batch, channel, (i + kernel_y) / Sy, (j + kernel_x) / Sx] * flipped_kernel[kernel_y, kernel_x];
                                    matrix[i,j] = dX_n_cin_i_j;
                                }
                            }
                        }
                    }
                }
                features[feature] = matrix;
            }

            dX[batch] = new FeatureSet<double>(features);
        }

        return new BatchedFeatureSet<double>(dX);
    }

    private BatchedFeatureSet<double> TransposeConvolve2(ConvolutionLayer layer, BackpropagationArgs args) {
        // dx = dy_0 * w'
        var (batch_count, channel_count, input_height, input_width) = args.InputBatch.Shape;
        var filter_count = layer.FilterCount;
        var (_, _, output_height, output_width) = args.OutputBatch.Shape;
        var stride_x = layer.StrideX;
        var stride_y = layer.StrideY;

        var padding_rows = layer.RowsPadding;
        var padding_cols = layer.ColumnsPadding;

        var input_to_output_padding_rows = (input_height - output_height) / 2;
        var input_to_output_padding_cols = (input_width - output_width) / 2;

        var result_features = new FeatureSet<double>[batch_count];

        Parallel.For(0, batch_count, batchIndex => {
            var batch_inputs = args.InputBatch[batchIndex];
            var batch_outputs = args.OutputBatch[batchIndex];
            var batch_errors = args.OutputErrors[batchIndex];

            var batch_features = new Matrix<double>[channel_count];
            for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                var input_error = new Matrix<double>(input_height, input_width);

                for (var filterIndex = 0; filterIndex < filter_count; filterIndex++) {
                    var filter = layer.Filters[filterIndex];
                    var filter_width = filter.Width;
                    var filter_height = filter.Height;

                    var filter_height_m1 = filter_height - 1;
                    var filter_width_m1 = filter_width - 1;

                    var input_padding_rows = (filter_height_m1) / 2;
                    var input_padding_cols = (filter_width_m1) / 2;

                    var input_width_padded = input_width + 2 * input_padding_rows;
                    var input_height_padded = input_height + 2 * input_padding_rows;
                    
                    var kernel = filter[channelIndex];
                    var error = batch_errors[filterIndex];
                    
                    // Slide kernel over input
                    for (var inputY = 0; inputY < input_height_padded; inputY++) {
                        var outY = (inputY - input_padding_rows - input_to_output_padding_rows) / stride_y;  // This assumes the output is "centered" in the middle of the input

                        for (var inputX = 0; inputX < input_width_padded; inputX++) {
                            var outX = (inputX - input_padding_cols - input_to_output_padding_cols) / stride_x; // This assumes the output is "centered" in the middle of the input

                            var sum = 0.0;
                            for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                                var inv_kernelY = filter_height_m1 - kernelY;
                                var outY_plus_kernel = outY + kernelY;
                                if (outY_plus_kernel < 0 || outY_plus_kernel >= output_height) continue; // Skip out-of-bounds rows
                                
                                for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                    var inv_kernelX = filter_width_m1 - kernelX;
                                    var outX_plus_kernel = outX + kernelX;
                                    if (outX_plus_kernel < 0 || outX_plus_kernel >= output_width) continue; // Skip out-of-bounds columns

                                    var kernel_value = kernel[inv_kernelY, inv_kernelX];
                                    var output_value = error[outY_plus_kernel, outX_plus_kernel];
                                    var result = kernel_value * output_value;

                                    sum += result;      
                                }
                            }

                            if (inputX >= 0 && inputX < input_width && inputY >= 0 && inputY < input_height)
                                input_error[inputY, inputX] += sum;
                        }
                    }
                }

                batch_features[channelIndex] = input_error;
            } 
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        });

        // TOD gradient clipping

        return new BatchedFeatureSet<double>(result_features);
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}