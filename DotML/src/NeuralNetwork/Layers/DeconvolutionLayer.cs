using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Transpose convolution layer. Uses convolution filters to perform an upscaling operation on the input data.
/// </summary>
public class TransposeConvolutionLayer : FeedforwardNetworkLayer {
    private ConvolutionFilter[] filters;
    public ReadOnlyCollection<ConvolutionFilter> Filters {get; init;}
    public int StrideX {get; init;}
    public int StrideY {get; init;}

    public int FilterCount => filters.Length;
    public Shape4D FilterShape => new Shape4D(FilterCount, this.InputShape.Channels, filters[0].Height, filters[0].Width);

    private int filterRows;
    private int filterColumns;
    public int InputRowsPadding {get; init;}
    public int InputColumnsPadding {get; init;}
    public int OutputRowsPadding {get; init;}
    public int OutputColumnsPadding {get; init;}

    public TransposeConvolutionLayer(Shape3D input_size, Padding padding, Expansion expansion, int strideX, int strideY, params ConvolutionFilter[] filters) {
        this.filters                  = filters;
        this.Filters                  = Array.AsReadOnly(this.filters);
        this.StrideX                  = Math.Max(1, strideX);
        this.StrideY                  = Math.Max(1, strideY);
      
        this.InputShape               = input_size;
        var inputRows                 = InputShape.Rows;                                                    
        var inputColumns              = InputShape.Columns;                                                   
        this.filterRows               = filters.Select(f => f.Height).Max();                                                        // 3
        this.filterColumns            = filters.Select(f => f.Width).Max();  

        this.InputRowsPadding         = padding == Padding.Same ? (filterRows - 1) / 2 : 0;
        this.InputColumnsPadding      = padding == Padding.Same ? (filterColumns - 1) / 2 : 0;
        this.OutputRowsPadding        = expansion == Expansion.Expand ? (filterRows - 1) / 2 : 0;
        this.OutputColumnsPadding     = expansion == Expansion.Expand ? (filterColumns - 1) / 2 : 0;

        var false_rows = inputRows + 2 * InputRowsPadding;
        var false_cols = inputColumns + 2 * InputColumnsPadding;

        // Copied from TransposeConvolveEach in Mat.cs
        var out_cols                  = (inputColumns - 1) * StrideX + filterColumns - 2 * InputColumnsPadding + OutputColumnsPadding; 
        var out_rows                  = (inputRows - 1) * StrideY + filterRows - 2 * InputRowsPadding + OutputRowsPadding;
        //var out_cols                  = (false_cols - 1) * StrideX + false_cols - 2 * OutputColumnsPadding; 
        //var out_rows                  = (false_rows - 1) * StrideY + false_rows - 2 * OutputRowsPadding;

        OutputShape = new Shape3D(
            channel:            filters.Length,
            rows:               out_rows,
            columns:            out_cols
        );

        Weights = new WeightTensor(filters, FilterCount, filterRows, filterColumns);
        Biases = new BiasTensor(filters);
    }

    public TransposeConvolutionLayer(Shape3D input_size, int inputPaddingX, int inputPaddingY, int outputPaddingX, int outputPaddingY, int strideX, int strideY, params ConvolutionFilter[] filters) {
        this.filters                  = filters;
        this.Filters                  = Array.AsReadOnly(this.filters);
        this.StrideX                  = Math.Max(1, strideX);
        this.StrideY                  = Math.Max(1, strideY);
      
        this.InputShape               = input_size;
        var inputRows                 = InputShape.Rows;                                                    
        var inputColumns              = InputShape.Columns;                                                   
        this.filterRows               = filters.Select(f => f.Height).Max();                                                        // 3
        this.filterColumns            = filters.Select(f => f.Width).Max();  

        this.InputRowsPadding         = Math.Max(0, inputPaddingY);
        this.InputColumnsPadding      = Math.Max(0, inputPaddingX); 
        this.OutputRowsPadding        = Math.Max(0, outputPaddingY);
        this.OutputColumnsPadding     = Math.Max(0, outputPaddingX);

        var padded_input_rows = inputRows + 2 * InputRowsPadding;
        var padded_input_columns = inputColumns + 2 * InputColumnsPadding;

        // Copied from TransposeConvolveEach in Mat.cs
        //                            = (in_cols - 1) * outputStrideX + kernel.Columns - 2 * outputPaddingX; 
        var out_cols = (inputColumns - 1) * StrideX + filterColumns - 2 * InputColumnsPadding + OutputColumnsPadding; 
        var out_rows = (inputRows - 1) * StrideY + filterRows - 2 * InputRowsPadding + OutputRowsPadding;
        //var out_cols                  = (padded_input_columns - 1) * StrideX + filterColumns - 2 * OutputColumnsPadding; 
        //var out_rows                  = (padded_input_rows - 1) * StrideY + filterRows - 2 * OutputRowsPadding;

        OutputShape = new Shape3D(
            channel:            filters.Length,
            rows:               out_rows,
            columns:            out_cols
        );

        Weights = new WeightTensor(filters, FilterCount, filterRows, filterColumns);
        Biases = new BiasTensor(filters);
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

    public override int TrainableParameterCount() {
        return Filters.Select(filter => filter.Select(kernel => kernel.Rows * kernel.Columns).Sum()).Sum() + FilterCount;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var channel_count = OutputShape.Channels;
        var outputs = new Matrix<double>[channel_count];

        for (var channel = 0; channel < channel_count; channel++) {
            var filter = filters[channel];
            var output = Matrix<double>.TransposeConvolveEach(
                channels,
                filter,
                inputPaddingX: InputColumnsPadding, inputPaddingY: InputRowsPadding,
                outputStrideX: StrideX, outputStrideY: StrideY,
                outputPaddingX: OutputColumnsPadding, outputPaddingY: OutputRowsPadding,
                bias: filter.Bias
            );
            outputs[channel] = output;
        }

        return new FeatureSet<double>(outputs);
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var filter_shape = FilterShape;
        var dW = BackpropagateWrtWeights(args.InputBatch, args.OutputErrors, filter_shape);
        var dB = BackpropagateWrtBias(args.OutputErrors);
        var dX = BackpropagateWrtInput(args.InputBatch, args.OutputErrors, filter_shape);

        return new BackpropagationReturns(
            error: dX,
            gradients: new Gradients(
                filters: this.filters, 
                weights: dW,
                bias: dB
            )
        );
    }

    private BatchedFeatureSet<double> BackpropagateWrtWeights(BatchedFeatureSet<double> X, BatchedFeatureSet<double> dY, Shape4D filter_shape) {
        // Kernel/Weight Gradients
        // dW(filter, kernel, row, col) = dy(filter, i, j) * input(c, i+k-1, j+l-1)
        // ----------------------------------------------------------------------------
        var (batch_size, in_channels, input_height, input_width) = X.Shape; // batches, inputs, rows, columns
        var (_, _, output_height, output_width) = dY.Shape;                 // batches, outputs, rows, columns
        var (out_channels, _, filter_height, filter_width) = filter_shape;  // outputs, inputs, kernel rows, kernel columns

        var dW = new BatchedFeatureSet<double>(filter_shape);               // outputs, kernels, kernel rows, kernel columns

        for (var batch_index = 0; batch_index < batch_size; batch_index++) {
            var xbatch = X[batch_index];
            var ybatch = dY[batch_index];

            for (var feature_index = 0; feature_index < in_channels; feature_index++) {
                var xfeats = xbatch[feature_index];

                for (var r = 0; r < input_height; r++) {
                    // Y-Region on the output that this input position contributed to
                    var region_start_y = r * StrideX - InputRowsPadding;
                    var region_end_y = region_start_y + filter_height;


                    for (var c = 0; c < input_width; c++) {
                        // X-Region on the output that this input position contributed to
                        var region_start_x = c * StrideX - InputColumnsPadding;
                        var region_end_x = region_start_x + filter_width;

                        var i = xfeats[r, c];

                        for (int out_y = region_start_y, ky = 0; out_y < region_end_y; out_y++, ky++) {
                            if (out_y < 0 || out_y >= output_height)
                                continue;

                            for (int out_x = region_start_x, kx = 0; out_x < region_end_x; out_x++, kx++) {
                                if (out_x < 0 || out_x >= output_width)
                                    continue;

                                for (var channel_index = 0; channel_index < out_channels; channel_index++) {
                                    dW[channel_index, feature_index, ky, kx] += i * ybatch[channel_index, out_y, out_x];
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
        var (batch_size, in_channels, out_rows, out_columns) = X.Shape; // In and out rows/columns flipped here since the "input" is dY and the output is "dX"
        //var padded_out_rows = out_rows + 2 * InputRowsPadding;
        //var padded_out_columns = out_columns + 2 * InputColumnsPadding;

        var (_, out_channels, in_rows, in_columns) = dY.Shape;
        var (filter_count, kernel_count, kernel_height, kernel_width) = filter_shape;

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
                    // COPIED FROM Matrix<double>.Convolve();
                    // --------------------------------------
                    for (var y = 0; y < out_rows; y++) {
                        var startY = y * StrideY - InputRowsPadding;

                        for (var x = 0; x < out_columns; x++) {
                            var startX = x * StrideX - InputColumnsPadding;

                            var total_sum = 0.0;
                            for (int ky = 0; ky < filterRows; ky++) {
                                var inY = startY + ky;
                                if (inY < 0 || inY >= in_rows) continue; // Skip out-of-bounds rows

                                for (int kx = 0; kx < filterColumns; kx++) {
                                    var inX = startX + kx;
                                    if (inX < 0 || inX >= in_columns) continue; // Skip out-of-bounds columns
                                    
                                    total_sum += output[inY, inX] * kernel[ky, kx];
                                }
                            }
                            
                            result[y, x] += total_sum;
                        }
                    }
                    // --------------------------------------
                }
            }
        }

        return new BatchedFeatureSet<double>(dX);
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

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);
    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

    /// <summary>
    /// Weight tensor for the convolution layer comprised of all kernels of all filters
    /// </summary>
    public WeightTensor Weights {get; init;}

    /// <summary>
    /// Bias tensor for the convolution layer comprised of all biases of all filters
    /// </summary>
    public BiasTensor Biases {get; init;}

    public class WeightTensor : IMutableTensorLike<double> {
        private ConvolutionFilter[] filters;
        private int kernel_count;
        private int kernel_columns;
        private int kernel_rows;

        public WeightTensor(ConvolutionFilter[] filters, int kernel_count, int kernel_rows, int kernel_columns) {
            this.filters = filters;
            this.kernel_count = kernel_count;
            this.kernel_rows = kernel_rows;
            this.kernel_columns = kernel_columns;
        }

        public int Filters => filters.Length;

        public int Kernels => kernel_count;

        public int Rows => kernel_rows;

        public int Columns => kernel_columns;

        public int Rank => 4;

        // Kernels and Filters are flipped here to match the way PyTorch stores tensors for transpose convolutions

        public int GetDimension(int index) => index switch {
            0 => kernel_count,
            1 => filters.Length,
            2 => kernel_rows,
            3 => kernel_columns,
            _ => throw new ArgumentOutOfRangeException(nameof(index))
        };

        public double GetElementAt(params int[] indices) {
            return filters[indices[1]][indices[0]][indices[2], indices[3]];
        }

        public void SetElementAt(double value, params int[] indices) {
            var mtx = filters[indices[1]][indices[0]];
            mtx[indices[2], indices[3]] = value;
        }
    }

    public class BiasTensor : IMutableTensorLike<double> {
        private ConvolutionFilter[] filters;

        public BiasTensor(ConvolutionFilter[] filters) {
            this.filters = filters;
        }

        public int Filters => filters.Length;

        public int Rank => 1;

        public int GetDimension(int index) => index switch {
            0 => filters.Length,
            _ => throw new ArgumentOutOfRangeException(nameof(index))
        };

        public double GetElementAt(params int[] indices) {
            return filters[indices[0]].Bias;
        }

        public void SetElementAt(double value, params int[] indices) {
            filters[indices[0]].Bias = value;
        }
    }
}