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
        private ConvolutionFilter[] Filters;
        public BatchedFeatureSet<double> FilterKernelGradients;
        public Vec<double> BiasGradients;

        public Gradients(ConvolutionFilter[] filters, BatchedFeatureSet<double> weights, Vec<double> bias) {
            this.Filters = filters;
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
                var param_filter = Filters[m];
                for (var k = 0; k < filter.Channels; k++) {
                    var matrix = filter[k];
                    var param_kernel = param_filter[k];
                    for (var r = 0; r < matrix.Rows; r++) {
                        for (var c = 0; c < matrix.Columns; c++) {
                            matrix[r,c] = handler(index++, param_kernel[r,c], matrix[r,c]);
                        }
                    }
                }
            }

            for (var i = 0; i < BiasGradients.Dimensionality; i++) {
                BiasGradients[i] = handler(index++, Filters[i].Bias, BiasGradients[i]);
            }
        }
    }
    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var filter_shape = FilterShape;
        var dW = BackpropagateWrtWeights(args.InputBatch, args.OutputErrors, filter_shape);
        var dB = BackpropagateWrtBias   (args.OutputErrors);
        //var dX = TransposeConvolve2(this, args);
        // TODO fix this
        var dX = BackpropagateWrtInput  (args.InputBatch, args.OutputErrors, filter_shape);

        return new BackpropagationReturns(
            dX,
            new Gradients (
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

    /*
Real
[
    -10.99089527130127,  -14.076019287109375,  -7.06515645980835;
    -3.2001407146453857, -4.924467086791992,   8.84919548034668;
    3.4817280769348145,  -0.5803017616271973,  -3.469475746154785
]
Mine
[ 
    -4.924466511838364,  8.849195123358085,   2.8818279499852455;
    -0.5803017808134274, -3.4694756084658263, -5.9314796635580285;
    -4.323853240887876,  1.008588256967232,   1.2048606242936302
]
    */
    private BatchedFeatureSet<double> BackpropagateWrtWeights(BatchedFeatureSet<double> X, BatchedFeatureSet<double> dY, Shape4D filter_shape) {
        // Kernel/Weight Gradients
        // dW(filter, kernel, row, col) = dy(filter, i, j) * input(c, i+k-1, j+l-1)
        // ----------------------------------------------------------------------------
        var (batch_size, in_channels, height, width) = X.Shape;
        var (_, _, output_height, output_width) = dY.Shape;
        var (out_channels, _, filter_height, filter_width) = filter_shape;

        var dW = new BatchedFeatureSet<double>(filter_shape); // (out_channels, kernel_count, filter_height, filter_width)

        for (var oY = 0; oY < output_height; oY++) {
            for (var oX = 0; oX < output_width; oX++) {
                // Region on the input which was used to compute this value on the output
                var x_start = oX * StrideX - ColumnsPadding;
                var x_end = x_start + filter_width;
                var y_start = oY * StrideY - RowsPadding;
                var y_end = y_start + filter_height;

                for (var batchIndex = 0; batchIndex < batch_size; batchIndex++) {
                    for (var filterIndex = 0; filterIndex < out_channels; filterIndex++) {
                        var grad = dY[batchIndex, filterIndex, oY, oX];

                        for (var kernelIndex = 0; kernelIndex < in_channels; kernelIndex++) {
                            var dk = dW[filterIndex, kernelIndex];
                            var x = X[batchIndex, kernelIndex];

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
        var (batch_size, in_channels, out_rows, out_columns) = X.Shape; // In and out rows/columns flipped here since the "input" is dY and the output is "dX"
        var (_, out_channels, in_rows, in_columns) = dY.Shape;
        var (filter_count, kernel_count, kernel_height, kernel_width) = filter_shape;
        var kernel_rows_m1 = kernel_height - 1;
        var kernel_cols_m1 = kernel_width - 1;

        var dX = new BatchedFeatureSet<double>(X.Shape);
        
        // How it worked.
        // Each filter was an output channel
        // Each kernel applied to a single input
        // So to go backwards we need to take the output from each filter and distribute it with each kernel back to the associated input
        for (var batch = 0; batch < batch_size; batch++) {
            for (var output_index = 0; output_index < out_channels; output_index++) {
                var filter = this.filters[output_index];
                var output = dY[batch, output_index];

                for (var kernel_index = 0; kernel_index < in_channels; kernel_index++) {
                    var kernel = filter[kernel_index];
                    var result = dX[batch, kernel_index];

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

                                    result[out_y, out_x] += i * kernel[ky, kx];
                                }
                            }
                        }
                    }
                    // --------------------------------------
                }
            }
        }

        return new BatchedFeatureSet<double>(dX);
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}