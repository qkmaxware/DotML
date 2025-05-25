using System.Collections.ObjectModel;
using System.Numerics;
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
        this.filters = filters;
        this.Filters = Array.AsReadOnly(this.filters);
        this.StrideX = Math.Max(1, strideX);
        this.StrideY = Math.Max(1, strideY);

        // Note, this only works if FILTERS is FIXED!! which may not be true
        this.InputShape         = input_size;
        var inputRows           = InputShape.Rows;                                                      // 32
        var inputColumns        = InputShape.Columns;                                                   // 32
        this.filterRows         = filters.Select(f => f.Height).Max();                                                        // 3
        this.filterColumns      = filters.Select(f => f.Width).Max();                                                         // 3
        this.RowsPadding        = padding == Padding.Same ? (filterRows - 1) / 2 : 0;                   // 1 
        this.ColumnsPadding     = padding == Padding.Same ? (filterColumns - 1) / 2 : 0;                // 1


        OutputShape             = new Shape3D(
            channel: filters.Length, 
            rows: (inputRows - filterRows + 2 * RowsPadding) / StrideY + 1,
            columns: (inputColumns - filterColumns + 2 * ColumnsPadding) / StrideX + 1
        );

        this.Weights = new WeightTensor(filters, filters.Select(f => f.Count).Max(), filterRows, filterColumns);
        this.Biases = new BiasTensor(filters);
    }

    public ConvolutionLayer(Shape3D input_size, int rowsPadding, int columnsPadding, int strideX, int strideY, params ConvolutionFilter[] filters) {
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
        this.RowsPadding         = Math.Max(0, rowsPadding);                   // 1 
        this.ColumnsPadding      = Math.Max(0, columnsPadding);                // 1


        OutputShape             = new Shape3D(
            channel: filters.Length, 
            rows: (inputRows - filterRows + 2 * RowsPadding) / StrideY + 1,
            columns: (inputColumns - filterColumns + 2 * ColumnsPadding) / StrideX + 1
        );
        
        this.Weights = new WeightTensor(filters, filters.Select(f => f.Count).Max(), filterRows, filterColumns);
        this.Biases = new BiasTensor(filters);
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
    private Matrix<float>[] Convolve(Matrix<float>[] inputs) {
        var filtersLength       = filters.Length;
        var output_list         = new Matrix<float>[filtersLength];

        for (var filterIndex = 0; filterIndex < filtersLength; filterIndex++) {
            var filter = filters[filterIndex];
            var output = Matrix<float>.ConvolveEach(
                inputs, filter, 
                strideX: StrideX, strideY: StrideY, 
                paddingX: ColumnsPadding, paddingY: RowsPadding,
                bias: filter.Bias
            );
            output_list[filterIndex] = output;
        }

        return output_list;
    }

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> inputs) {
        return new FeatureSet<float>(this.Convolve((Matrix<float>[])inputs));
    }

    public class Gradients : LayerGradients {
        private ConvolutionFilter[] Filters;
        public BatchedFeatureSet<float> FilterKernelGradients;
        public Vec<float> BiasGradients;

        public Gradients(ConvolutionFilter[] filters, BatchedFeatureSet<float> weights, Vec<float> bias) {
            this.Filters = filters;
            this.FilterKernelGradients = weights;
            this.BiasGradients = bias;
        }

        public override void Clip(float weight_threshold, float bias_threshold) {
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
    private BatchedFeatureSet<float> BackpropagateWrtWeights(BatchedFeatureSet<float> X, BatchedFeatureSet<float> dY, Shape4D filter_shape) {
        // Kernel/Weight Gradients
        // dW(filter, kernel, row, col) = dy(filter, i, j) * input(c, i+k-1, j+l-1)
        // ----------------------------------------------------------------------------
        var (batch_size, in_channels, height, width) = X.Shape;
        var (_, _, output_height, output_width) = dY.Shape;
        var (out_channels, _, filter_height, filter_width) = filter_shape;

        var dW = new BatchedFeatureSet<float>(filter_shape); // (out_channels, kernel_count, filter_height, filter_width)

        for (var oY = 0; oY < output_height; oY++) {
            // Region on the input which was used to compute this value on the output
            var y_start = oY * StrideY - RowsPadding;
            var y_end = y_start + filter_height;

            for (var oX = 0; oX < output_width; oX++) {
                // Region on the input which was used to compute this value on the output
                var x_start = oX * StrideX - ColumnsPadding;
                var x_end = x_start + filter_width;

                for (var batchIndex = 0; batchIndex < batch_size; batchIndex++) {
                    var feats = dY[batchIndex];
                    for (var filterIndex = 0; filterIndex < out_channels; filterIndex++) {
                        var feat = feats[filterIndex];
                        var grad = feat[oY, oX];

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

    private Vec<float> BackpropagateWrtBias(BatchedFeatureSet<float> dY) {
        // Bias Gradients
        // dL/dB = dL/dY * dY/dB = dY * dY/dB
        // dY/dB = [1; ... ; 1] because b is constant wrt y
        // dL/dB = dL/dY = Sum over x,y of dY(x,y) given the above statement
        // -----------------------------------------------------------------------
        // Compute dL/dB by summing over batches, rows, and columns
        var featureCount = dY.Channels; // Should be equal to FilterCount
        var batchCount = dY.Batches;
        var dB = new float[featureCount];
        var vector_size = Vector<float>.Count;  // TODO check Vector<double>.IsHardwareAccelerated tp determine SIMD vs non SIMD path

        if (Vector.IsHardwareAccelerated)
        {
            for (var featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                // Compute sum
                var sum = 0.0f;
                Vector<float> vsum = Vector<float>.Zero;
                for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
                {
                    var span = dY[batchIndex, featureIndex].AsReadOnlySpan();

                    int k = 0;
                    int lengthMinusBuffer = span.Length - vector_size;
                    for (; k <= lengthMinusBuffer; k += vector_size)
                    {
                        vsum += new Vector<float>(span.Slice(k, vector_size));
                    }

                    for (; k < span.Length; k++)
                    {
                        sum += span[k];
                    }

                }

                // Apply biases
                sum += Vector.Sum(vsum);
                dB[featureIndex] = sum;
            }
        }
        else
        {
            for (var featureIndex = 0; featureIndex < featureCount; featureIndex++) {
                // Compute sum
                var sum = 0.0f;
                for (var batchIndex = 0; batchIndex < batchCount; batchIndex++) {
                    var span = dY[batchIndex, featureIndex].AsReadOnlySpan();
                    for (var k = 0; k < span.Length; k++)
                        sum += span[k];
                }
    
                // Apply biases
                dB[featureIndex] = sum;
            }
        }

        return Vec<float>.Wrap(dB);
    }

    private BatchedFeatureSet<float> BackpropagateWrtInput(BatchedFeatureSet<float> X, BatchedFeatureSet<float> dY, Shape4D filter_shape) {
        // Input Gradient
        // To compute the gradients w.r.t. the input (dinput), you perform a convolution of dY with the filter weights, flipping them. 
        // This is the same process used to calculate the forward pass convolution but with flipped weights
        // ---------------------------------------------------------------
        var (batch_size, in_channels, out_rows, out_columns) = X.Shape; // In and out rows/columns flipped here since the "input" is dY and the output is "dX"
        var (_, out_channels, in_rows, in_columns) = dY.Shape;
        var (_, _, kernel_height, kernel_width) = filter_shape;

        var dX = new BatchedFeatureSet<float>(X.Shape);
        
        // How it worked.
        // Each filter was an output channel
        // Each kernel applied to a single input
        // So to go backwards we need to take the output from each filter and distribute it with each kernel back to the associated input
        for (var batch = 0; batch < batch_size; batch++) {
            var ofeatures = dY[batch];
            var ifeatures = dX[batch];
            for (var output_index = 0; output_index < out_channels; output_index++) {
                var filter = this.filters[output_index];
                var output = ofeatures[output_index];

                for (var kernel_index = 0; kernel_index < in_channels; kernel_index++) {
                    var kernel = filter[kernel_index];
                    var result = ifeatures[kernel_index];

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

        return new BatchedFeatureSet<float>(dX);
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

    public class WeightTensor : IMutableTensorLike<float> {
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

        public int GetDimension(int index) => index switch {
            0 => filters.Length,
            1 => kernel_count,
            2 => kernel_rows,
            3 => kernel_columns,
            _ => throw new ArgumentOutOfRangeException(nameof(index))
        };

        public float GetElementAt(params int[] indices) {
            return filters[indices[0]][indices[1]][indices[2], indices[3]];
        }

        public void SetElementAt(float value, params int[] indices) {
            var mtx = filters[indices[0]][indices[1]];
            mtx[indices[2], indices[3]] = value;
        }
    }

    public class BiasTensor : IMutableTensorLike<float> {
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

        public float GetElementAt(params int[] indices) {
            return filters[indices[0]].Bias;
        }

        public void SetElementAt(float value, params int[] indices) {
            filters[indices[0]].Bias = value;
        }
    }
}