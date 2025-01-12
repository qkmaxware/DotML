using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public partial class BatchTrainerEnumerator<TNetwork> {

public struct BackpropagationArgs {
    public Vec<double>[] BatchTrueLabels;
    public int LayerIndex;
    public BatchedFeatureSet<double> InputBatch;
    public BatchedFeatureSet<double> OutputBatch;
    public BatchedFeatureSet<double> OutputErrors;
}

public abstract class Gradients {
    public abstract void AverageOf(IEnumerable<Gradients?> batchGradients);
}

public class FullyConnectedGradients : Gradients {
    public Matrix<double> WeightGradients;
    public Vec<double> BiasGradients;

    public override void AverageOf(IEnumerable<Gradients?> batchGradients) {
        var all_fc_batches = batchGradients.OfType<FullyConnectedGradients>();
        var WeightGradients = Matrix<double>.Average(all_fc_batches.Select(fcg => fcg.WeightGradients));
        var BiasGradients = Vec<double>.Average(all_fc_batches.Select(fcg => fcg.BiasGradients));
        
        this.WeightGradients = WeightGradients;
        this.BiasGradients = BiasGradients;
    }
}

public class ConvolutionGradients : Gradients {
    public Matrix<double>[][]? FilterKernelGradients;
    public double[]? BiasGradients;

    public override void AverageOf(IEnumerable<Gradients?> batchGradients) {
        var filters = FilterKernelGradients?.Length ?? 0;
        var all_c_batches = batchGradients.OfType<ConvolutionGradients>();
        var new_filter_kernel_gradients = new Matrix<double>[filters][];
        for (var filterIndex = 0; filterIndex < filters; filterIndex++) {
            var kernels = FilterKernelGradients?[filterIndex]?.Length ?? 0;
            var kernel_grads = new Matrix<double>[kernels];

            for (var kernelIndex = 0; kernelIndex < kernels; kernelIndex++) {
                kernel_grads[kernelIndex] = Matrix<double>.Average(all_c_batches.Select(c => c.FilterKernelGradients is null ? new Matrix<double>() : c.FilterKernelGradients[filterIndex][kernelIndex]));
            }

            new_filter_kernel_gradients[filterIndex] = kernel_grads;
        }

        this.FilterKernelGradients = new_filter_kernel_gradients;
        this.BiasGradients = (double[])Vec<double>.Average(all_c_batches.Select(c => c.BiasGradients is null ? new Vec<double>() : Vec<double>.Wrap(c.BiasGradients)));   
    }
}

public class DepthwiseConvolutionGradients : Gradients {
    public Matrix<double>[]? KernelGradients;

    public override void AverageOf(IEnumerable<Gradients?> batchGradients) {
        var kernels = KernelGradients?.Length ?? 0;
        var kernel_grads = new Matrix<double>[kernels];
        var all_batches = batchGradients.OfType<DepthwiseConvolutionGradients>();
        for (var i = 0; i < kernels; i++) {
            kernel_grads[i] = Matrix<double>.Average(all_batches.Select(c => c.KernelGradients is null ? new Matrix<double>() : c.KernelGradients[i]));
        }

        this.KernelGradients = kernel_grads;
    }
}

public class NormalizationGradients : Gradients {
    public Matrix<double>[]? GammaGradients;
    public Matrix<double>[]? BetaGradients;

    public override void AverageOf(IEnumerable<Gradients?> batchGradients) {
        var all_batches = batchGradients.OfType<NormalizationGradients>();

        var gamma_c = GammaGradients?.Length ?? 0;
        var beta_c = BetaGradients?.Length ?? 0;

        var new_gammas = new Matrix<double>[gamma_c];
        for (var i = 0; i < gamma_c; i++) {
            new_gammas[i] = Matrix<double>.Average(all_batches.Select(
                batch_elem => batch_elem.GammaGradients is null ? new Matrix<double>() : batch_elem.GammaGradients[i]
            ));
        }
        
        var new_betas = new Matrix<double>[beta_c];
        for (var i = 0; i < gamma_c; i++) {
            new_betas[i] = Matrix<double>.Average(all_batches.Select(
                batch_elem => batch_elem.BetaGradients is null ? new Matrix<double>() : batch_elem.BetaGradients[i]
            ));
        }

        
        this.GammaGradients = new_gammas;
        this.BetaGradients = new_betas;
    }
}

public struct BackpropagationReturns {
    public BatchedFeatureSet<double> InputErrors;
    public Gradients? Gradient;
}

public bool UseGradientClipping => backpropagationActions.UseGradientClipping;
public double GradientClippingThresholdWeight => backpropagationActions.GradientClippingThresholdWeight;
public double GradientClippingThresholdBias => backpropagationActions.GradientClippingThresholdBias;

private BackpropagationActions backpropagationActions {get; init;}
public class BackpropagationActions : ILayerVisitor<BatchTrainerEnumerator<TNetwork>.BackpropagationArgs, BatchTrainerEnumerator<TNetwork>.BackpropagationReturns> {

    public BackpropagationActions(bool useClipping, double weightThreshold, double biasThreshold) {
        this.UseGradientClipping = useClipping;
        this.GradientClippingThresholdWeight = weightThreshold;
        this.GradientClippingThresholdBias = biasThreshold;
    }

    // Helper to perform tranpose convolution
    private Matrix<double>[] TransposeConvolve(Matrix<double>[] inputs, Matrix<double>[] errors, int strideX, int strideY, int padRows, int padColumns, IList<ConvolutionFilter> filters) {
        var inputChannels = inputs.Length;
        var inputErrors = new Matrix<double>[inputChannels];
        var paddingRows         = padRows;
        var paddingColumns      = padColumns;

        Parallel.For(0, inputChannels, channel => {;
            var input = inputs[channel];
            var inputError = new double[input.Rows, input.Columns];

            // Calculate the input errors for each filter
            for (var filterIndex = 0; filterIndex < filters.Count; filterIndex++) {
                var filter = filters[filterIndex];
                var filterRows = filter.Height;
                var filterColumns = filter.Width;
                var error = errors[filterIndex];
                var rows = error.Rows;
                var cols = error.Columns;
                var kernel = filter[channel];

                // Iterate over output errors to compute gradient
                for (int outY = 0; outY < rows; outY++) {
                    var startY = outY * strideY - paddingRows;
                    for (int outX = 0; outX < cols; outX++) {
                        var startX = outX * strideX - paddingColumns;
                        // Place the error at the corresponding position in the input space
                        for (int ky = 0; ky < filterRows; ky++) {
                            var inY = startY + ky;
                            for (int kx = 0; kx < filterColumns; kx++) {
                                var inX = startX + kx;

                                if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                    inputError[inY, inX] += error[outY, outX] * kernel[ky, kx];
                                }
                            }
                        }
                    }
                }
            }

            inputErrors[channel] = Matrix<double>.Wrap(inputError);
        });

        return inputErrors;
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
        //for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) {
            var batch_inputs = args.InputBatch[batchIndex];
            var batch_outputs = args.OutputBatch[batchIndex];
            var batch_errors = args.OutputErrors[batchIndex];

            var batch_features = new Matrix<double>[channel_count];
            for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                var input_error = new double[input_height, input_width];

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
                                
                                for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                    var inv_kernelX = filter_width_m1 - kernelX;
                                    var outX_plus_kernel = outX + kernelX;

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

                batch_features[channelIndex] = Matrix<double>.Wrap(input_error);
            } 
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        //}
        });

        // TOD gradient clipping

        return new BatchedFeatureSet<double>(result_features);
    }

    public BackpropagationReturns Visit2(ConvolutionLayer layer, BackpropagationArgs args) {
        // https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        // args.InputBatch has shape (Batches, Channels, Rows, Columns)
        // args.OutputBatch has the shape (Batches, Filters, Rows`, Columns`)
        // args.OutpurErrors has the shape (Batches, Filters, Rows`, Columns`)
        // layer.Filters has the shape (Filters, Kernels, Kernel Height, Kernel Width)

        var (batch_count, channel_count, input_height, input_width) = args.InputBatch.Shape;
        var filter_count = layer.FilterCount;
        var (_, _, output_height, output_width) = args.OutputBatch.Shape;

        var paddingRows         = layer.RowsPadding; 
        var paddingColumns      = layer.ColumnsPadding; 

        // Bias Gradients
        // dL/dB = dL/dY * dY/dB = dY * dY/dB
        // dY/dB = [1; ... ; 1] because b is constant wrt y
        // dL/dB = dL/dY = Sum(x)Sum(y) of dY(x,y) given the above statement
        var bias_gradients = new double[filter_count];
        Parallel.For(0, filter_count, filterIndex => {
            //for (var filterIndex = 0; filterIndex < filter_count; filterIndex++) {
                var gradient = 0.0;
                // Sum over all batches
                for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) {
                    gradient += args.OutputErrors[batchIndex][filterIndex].Sum(); // Sum over the rows and columns
                }
                clip(ref gradient, GradientClippingThresholdBias);
                bias_gradients[filterIndex] = gradient;
            //}
        });

        // Kernel Gradients
        // This looks almost identical to what I already have, except summing over batch
        // dW(filter, kernel, row, col) = dy(filter, i, j) * input(c, i+k-1, j+l-1)
        // convolution between the input and the error
        var filter_kernel_gradients = new Matrix<double>[filter_count][];
        for (var f = 0; f < filter_count; f++) {
            var filter = layer.Filters[f];
            var kcount = filter.Count;
            var kernels = new Matrix<double>[kcount];
            for (var i = 0; i < kcount; i++) {
                var kernel = filter[i];
                kernels[i] = Matrix<double>.Wrap(new double[kernel.Rows, kernel.Columns]);
            }
            filter_kernel_gradients[f] = kernels;
        }
        // TODO gradient clipping

        Parallel.For(0, filter_count, filterIndex => {
        //for (var filterIndex = 0; filterIndex < filter_count; filterIndex++) { // This can be parallelized
            var filter = layer.Filters[filterIndex];
            var filter_width = filter.Width;
            var filter_height = filter.Height;
            var kernel_gradients = filter_kernel_gradients[filterIndex];

            for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) { // This can't be as two batches can access the same channel at the same time
                var input_features = args.InputBatch[batchIndex]; // This is the number of channels, not the number of filters
                var output_features = args.OutputBatch[batchIndex][filterIndex]; 
                var output_errors = args.OutputErrors[batchIndex][filterIndex];

                var error_rows = output_errors.Rows;
                var error_cols = output_errors.Columns;

                for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                    var input_channel = input_features[channelIndex]; // Matrix 2D
                    var kernel_gradient = kernel_gradients[channelIndex].AsArray(); // make this a mutable reference

                    for (int outY = 0; outY < error_rows; outY++) {
                        var startY = outY * layer.StrideY - paddingRows;

                        for (int outX = 0; outX < error_cols; outX++) {
                            var startX = outX * layer.StrideX - paddingColumns;
                            var error = output_errors[outY, outX];

                            for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                                var inY = startY + kernelY;

                                for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                    var inX = startX + kernelX;

                                    kernel_gradient[kernelY, kernelX] += input_channel[inY, inX] * error;
                                }
                            }
                        }
                    }
                }
            }
        //}
        });

        /*var input_gradients = new FeatureSet<double>[batch_count];
        Parallel.For(0, batch_count, (batchIndex) => {
            var inputs = args.InputBatch[batchIndex]; // This is the number of channels, not the number of filters
            var errors = args.OutputErrors[batchIndex];

            var inputErrors = TransposeConvolve((Matrix<double>[])inputs, (Matrix<double>[])errors, layer.StrideX, layer.StrideY, paddingRows, paddingColumns, layer.Filters);
            input_gradients[batchIndex] = new FeatureSet<double>(inputErrors);
        });*/

        return new BackpropagationReturns {
            InputErrors = TransposeConvolve2(layer, args),
            Gradient = new ConvolutionGradients {
                FilterKernelGradients = filter_kernel_gradients,
                BiasGradients = bias_gradients,
            }
        };
    }

    public BackpropagationReturns Visit(ConvolutionLayer layer, BackpropagationArgs args) {
        return Visit2(layer, args); // Testing this out :)
        var batch_size = args.InputBatch.Batches;
        var input_gradients = new FeatureSet<double>[batch_size];
        var batch_gradients = new Gradients[batch_size];

        // Compute gradients
        Parallel.For(0, batch_size, (batchIndex) => {
            var filterGradients = new Matrix<double>[layer.FilterCount][];
            var biasGradients = new double[layer.FilterCount];
            var paddingRows         = layer.RowsPadding; 
            var paddingColumns      = layer.ColumnsPadding; 
            var inputs              = args.InputBatch[batchIndex];
            var errors              = args.OutputErrors[batchIndex];
            batch_gradients[batchIndex] = new ConvolutionGradients {
                FilterKernelGradients = filterGradients,
                BiasGradients = biasGradients
            };

            // Loop through each filter and compute gradients
            Parallel.For(0, layer.FilterCount, filterIndex => {
                var filter              = layer.Filters[filterIndex];                                                    

                var gradient            = new Matrix<double>[inputs.Channels];
                var error               = errors[filterIndex];
                var rows                = error.Rows;
                var cols                = error.Columns;
                var output              = args.OutputBatch[batchIndex][filterIndex];
                const int gradOut       = 1;
                var biasGradient        = 0.0;

                for (var inputIndex = 0; inputIndex < inputs.Channels; inputIndex++) {
                    var input           = inputs[inputIndex];
                    var kernel          = filter[inputIndex];
                    var kernelGradient  = new double[kernel.Rows, kernel.Columns];

                    // Slide the kernel over the error map computing the correlation
                    for (int outY = 0; outY < rows; outY++) {
                        var startY = outY * layer.StrideY - paddingRows;
                        for (int outX = 0; outX < cols; outX++) {
                            var startX = outX * layer.StrideX - paddingColumns;
                            var slope = gradOut;
                            var pixel_error = error[outY, outX];
                            var errorContribution = slope * pixel_error;
                            biasGradient += errorContribution;

                            for (var ky = 0; ky < kernel.Rows; ky++) {
                                var inY = startY + ky;
                                for (var kx = 0; kx < kernel.Columns; kx++) {
                                    var inX = startX + kx;

                                    if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                        kernelGradient[ky, kx] += errorContribution * input[inY, inX];
                                    }
                                }
                            }
                        }
                    }

                    // Save the kernel gradient
                    gradient[inputIndex] = Matrix<double>.Wrap(kernelGradient);
                }

                // Store gradients
                clip(gradient, GradientClippingThresholdWeight);
                clip(ref biasGradient, GradientClippingThresholdBias);
                filterGradients[filterIndex] = gradient;
                biasGradients[filterIndex] = biasGradient;
            });

            var inputErrors = TransposeConvolve((Matrix<double>[])inputs, (Matrix<double>[])errors, layer.StrideX, layer.StrideY, paddingRows, paddingColumns, layer.Filters);
            input_gradients[batchIndex] = new FeatureSet<double>(inputErrors);
        });

        // Average gradients
        var grad = batch_gradients.Length > 0 ? batch_gradients[0] : null;
        grad?.AverageOf(batch_gradients);

        // Return results
        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(input_gradients),
            Gradient = grad
        };
        /*// Initialize extra storage
        var filterGradients = new Matrix<double>[layer.FilterCount][];
        var biasGradients = new double[layer.FilterCount];
        var paddingRows         = layer.RowsPadding; 
        var paddingColumns      = layer.ColumnsPadding; 

        // Loop through each filter and compute gradients
        Parallel.For(0, layer.FilterCount, filterIndex => {
            var filter              = layer.Filters[filterIndex];                                                    

            var gradient            = new Matrix<double>[args.Inputs.Channels];
            var error               = args.Errors[filterIndex];
            var rows                = error.Rows;
            var cols                = error.Columns;
            var output              = args.Outputs[filterIndex];
            const int gradOut       = 1;
            var biasGradient        = 0.0;

            for (var inputIndex = 0; inputIndex < args.Inputs.Channels; inputIndex++) {
                var input           = args.Inputs[inputIndex];
                var kernel          = filter[inputIndex];
                var kernelGradient  = new double[kernel.Rows, kernel.Columns];

                // Slide the kernel over the error map computing the correlation
                for (int outY = 0; outY < rows; outY++) {
                    var startY = outY * layer.StrideY - paddingRows;
                    for (int outX = 0; outX < cols; outX++) {
                        var startX = outX * layer.StrideX - paddingColumns;
                        var slope = gradOut;
                        var pixel_error = error[outY, outX];
                        var errorContribution = slope * pixel_error;
                        biasGradient += errorContribution;

                        for (var ky = 0; ky < kernel.Rows; ky++) {
                            var inY = startY + ky;
                            for (var kx = 0; kx < kernel.Columns; kx++) {
                                var inX = startX + kx;

                                if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                    kernelGradient[ky, kx] += errorContribution * input[inY, inX];
                                }
                            }
                        }
                    }
                }

                // Save the kernel gradient
                gradient[inputIndex] = Matrix<double>.Wrap(kernelGradient);
            }

            // Store gradients
            clip(gradient, GradientClippingThresholdWeight);
            clip(ref biasGradient, GradientClippingThresholdBias);
            filterGradients[filterIndex] = gradient;
            biasGradients[filterIndex] = biasGradient;
        });

        var inputErrors = TransposeConvolve((Matrix<double>[])args.Inputs, args.Errors, layer.StrideX, layer.StrideY, paddingRows, paddingColumns, layer.Filters);

        return new BackpropagationReturns {
            InputErrors = inputErrors,
            Gradient = new ConvolutionGradients {
                FilterKernelGradients = filterGradients,
                BiasGradients = biasGradients,
            }
        };*/
    }

    private Matrix<double>[] DepthwiseTransposeConvolve(Matrix<double>[] inputs, Matrix<double>[] errors, int strideX, int strideY, int padRows, int padColumns, ConvolutionFilter filter) {
        var inputChannels = inputs.Length;
        var inputErrors = new Matrix<double>[inputChannels];
        var paddingRows         = padRows;
        var paddingColumns      = padColumns;

        Parallel.For(0, inputChannels, channel => {;
            var input = inputs[channel];
            var inputError = new double[input.Rows, input.Columns];

            // Calculate the input errors for each filter
            var filterRows = filter.Height;
            var filterColumns = filter.Width;
            var error = errors[channel];
            var rows = error.Rows;
            var cols = error.Columns;
            var kernel = filter[channel];

            // Iterate over output errors to compute gradient
            for (int outY = 0; outY < rows; outY++) {
                var startY = outY * strideY - paddingRows;
                for (int outX = 0; outX < cols; outX++) {
                    var startX = outX * strideX - paddingColumns;
                    // Place the error at the corresponding position in the input space
                    for (int ky = 0; ky < filterRows; ky++) {
                        var inY = startY + ky;
                        for (int kx = 0; kx < filterColumns; kx++) {
                            var inX = startX + kx;

                            if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                inputError[inY, inX] += error[outY, outX] * kernel[ky, kx];
                            }
                        }
                    }
                }
            }

            inputErrors[channel] = Matrix<double>.Wrap(inputError);
        });

        return inputErrors;
    }

    private BatchedFeatureSet<double> DepthwiseTransposeConvolve2(DepthwiseConvolutionLayer layer, BackpropagationArgs args) {
        // dx = dy_0 * w'
        var (batch_count, channel_count, input_height, input_width) = args.InputBatch.Shape;
        var (_, _, output_height, output_width) = args.OutputBatch.Shape;
        var stride_x = layer.StrideX;
        var stride_y = layer.StrideY;

        var padding_rows = layer.RowsPadding;
        var padding_cols = layer.ColumnsPadding;

        var input_to_output_padding_rows = (input_height - output_height) / 2;
        var input_to_output_padding_cols = (input_width - output_width) / 2;

        var result_features = new FeatureSet<double>[batch_count];

        Parallel.For(0, batch_count, batchIndex => {
        //for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) {
            var batch_inputs = args.InputBatch[batchIndex];
            var batch_outputs = args.OutputBatch[batchIndex];
            var batch_errors = args.OutputErrors[batchIndex];

            var batch_features = new Matrix<double>[channel_count];

            var filter = layer.Filter;
            var filter_width = filter.Width;
            var filter_height = filter.Height;

            var filter_height_m1 = filter_height - 1;
            var filter_width_m1 = filter_width - 1;

            var input_padding_rows = (filter_height_m1) / 2;
            var input_padding_cols = (filter_width_m1) / 2;

            var input_width_padded = input_width + 2 * input_padding_rows;
            var input_height_padded = input_height + 2 * input_padding_rows;

            for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                var input_error = new double[input_height, input_width];
                    
                var kernel = filter[channelIndex];
                var error = batch_errors[channelIndex];
                
                // Slide kernel over input
                for (var inputY = 0; inputY < input_height_padded; inputY++) {
                    var outY = (inputY - input_padding_rows - input_to_output_padding_rows) / stride_y;  // This assumes the output is "centered" in the middle of the input

                    for (var inputX = 0; inputX < input_width_padded; inputX++) {
                        var outX = (inputX - input_padding_cols - input_to_output_padding_cols) / stride_x; // This assumes the output is "centered" in the middle of the input

                        var sum = 0.0;
                        for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                            var inv_kernelY = filter_height_m1 - kernelY;
                            var outY_plus_kernel = outY + kernelY;
                            
                            for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                var inv_kernelX = filter_width_m1 - kernelX;
                                var outX_plus_kernel = outX + kernelX;

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

                batch_features[channelIndex] = Matrix<double>.Wrap(input_error);
            } 
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        //}
        });

        return new BatchedFeatureSet<double>(result_features);
    }

    public BackpropagationReturns Visit2(DepthwiseConvolutionLayer layer, BackpropagationArgs args) {
        var batch_count = args.InputBatch.Batches;
        Matrix<double>[] kernel_gradients = new Matrix<double>[batch_count];

        // Kernel Gradients
        // This looks almost identical to what I already have, except summing over batch
        // dW(filter, kernel, row, col) = dy(filter, i, j) * input(c, i+k-1, j+l-1)
        // convolution between the input and the error
        var filter = layer.Filter;
        var filter_height = filter.Height;
        var filter_width = filter.Width;
        var paddingRows  = layer.RowsPadding; 
        var paddingColumns  = layer.ColumnsPadding; 

        var kcount = filter.Count;
        Parallel.For(0, kcount, i => {
        //for (var i = 0; i < kcount; i++) {
            var kernel = filter[i];
            var kernel_gradient_matrix = Matrix<double>.Wrap(new double[kernel.Rows, kernel.Columns]);

            for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) { // This can't be as two batches can access the same channel at the same time
                var input_features = args.InputBatch[batchIndex]; // This is the number of channels, not the number of filters
                var output_errors = args.OutputErrors[batchIndex][i];

                var error_rows = output_errors.Rows;
                var error_cols = output_errors.Columns;

                var input_channel = input_features[i]; // Matrix 2D
                var kernel_gradient = kernel_gradient_matrix.AsArray(); // make this a mutable reference

                for (int outY = 0; outY < error_rows; outY++) {
                    var startY = outY * layer.StrideY - paddingRows;

                    for (int outX = 0; outX < error_cols; outX++) {
                        var startX = outX * layer.StrideX - paddingColumns;
                        var error = output_errors[outY, outX];

                        for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                            var inY = startY + kernelY;

                            for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                var inX = startX + kernelX;

                                kernel_gradient[kernelY, kernelX] += input_channel[inY, inX] * error;
                            }
                        }
                    }
                }
            }

            kernel_gradients[i] = kernel_gradient_matrix;
        //}
        });

        clip(kernel_gradients, GradientClippingThresholdWeight);

        return new BackpropagationReturns {
            InputErrors = DepthwiseTransposeConvolve2(layer, args),
            Gradient = new DepthwiseConvolutionGradients {
                KernelGradients = kernel_gradients
            }
        };
    }

    public BackpropagationReturns Visit(DepthwiseConvolutionLayer layer, BackpropagationArgs args) {
        return Visit2(layer, args);
        var batched_gradients = new Gradients[args.InputBatch.Batches];
        var input_errors = new FeatureSet<double>[args.InputBatch.Batches]; 

        Parallel.For(0, args.InputBatch.Batches, batchIndex => {
            var inputs              = args.InputBatch[batchIndex];
            var errors              = args.OutputErrors[batchIndex];
            var length              = layer.Filter.Count;
            var gradients           = new Matrix<double>[length];
            var paddingRows         = layer.RowsPadding; 
            var paddingColumns      = layer.ColumnsPadding; 

            Parallel.For(0, length, inputIndex => {
                var input           = inputs[inputIndex];
                var kernel          = layer.Filter[inputIndex];
                var kernelGradient  = new double[kernel.Rows, kernel.Columns];
                var error           = errors[inputIndex];
                var rows            = error.Rows;
                var cols            = error.Columns;

                // Slide the kernel over the error map computing the correlation
                for (int outY = 0; outY < rows; outY++) {
                    var startY = outY * layer.StrideY - paddingRows;
                    for (int outX = 0; outX < cols; outX++) {
                        var startX = outX * layer.StrideX - paddingColumns;
                        var pixel_error = error[outY, outX];
                        var errorContribution = pixel_error;

                        for (var ky = 0; ky < kernel.Rows; ky++) {
                            var inY = startY + ky;
                            for (var kx = 0; kx < kernel.Columns; kx++) {
                                var inX = startX + kx;

                                if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                    kernelGradient[ky, kx] += errorContribution * input[inY, inX];
                                }
                            }
                        }
                    }
                }

                // Save the kernel gradient
                gradients[inputIndex] = Matrix<double>.Wrap(kernelGradient);
            });

            batched_gradients[batchIndex] = new DepthwiseConvolutionGradients {
                KernelGradients = gradients
            };

            var inputErrors = DepthwiseTransposeConvolve((Matrix<double>[])inputs, (Matrix<double>[])errors, layer.StrideX, layer.StrideY, paddingRows, paddingColumns, layer.Filter);
            input_errors[batchIndex] = new FeatureSet<double>(inputErrors);
        });
        
        var grad = batched_gradients.Length > 0 ? batched_gradients[0] : null;
        grad?.AverageOf(batched_gradients);
        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(input_errors),
            Gradient = grad
        };
    }

    public BackpropagationReturns Visit(PoolingLayer layer, BackpropagationArgs args) {
        FeatureSet<double>[] input_errors = new FeatureSet<double>[args.OutputBatch.Batches];

        Parallel.For(0, args.OutputBatch.Batches, batchIndex => {
            // Extract inputs, outputs, and errors
            var inputs = args.InputBatch[batchIndex];
            var outputs = args.OutputBatch[batchIndex];
            var errors = args.OutputErrors[batchIndex];

            int featureCount = inputs.Channels;
            var batchErrors = new Matrix<double>[featureCount];

            var filterWidth = layer.FilterWidth;
            var filterHeight = layer.FilterHeight;
            var filterElementCount = filterWidth * filterHeight;

            Parallel.For(0, featureCount, featureIndex => {
                // Get the input and output for this batch item
                var input = inputs[featureIndex];
                var output = outputs[featureIndex];
                var error = errors[featureIndex];

                // Initialize the error matrix for the input
                var inputError = new double[input.Rows, input.Columns];

                // Loop over output
                for (int row = 0; row < output.Rows; row++) {
                    var StartY = row * layer.StrideY;
                    var EndY = Math.Min(row * layer.StrideY + filterHeight, input.Rows);
                    for (int col = 0; col < output.Columns; col++) {
                        var StartX = col * layer.StrideX;
                        var EndX = Math.Min(col * layer.StrideX + filterWidth, input.Columns);

                        // Loop over input values where the filter is applied
                        switch (layer) {
                            case LocalMaxPoolingLayer maxPool:
                                int maxRow = 0, maxCol = 0; double maxVal = double.MinValue; // Values for max pooling
                                for (int kr = StartY; kr < EndY; kr++) {
                                    for (int kc = StartX; kc < EndX; kc++) {
                                        var value = input[kr, kc];

                                        // Compute; Assume max pooling (avg is different)
                                        if (value > maxVal) {
                                            maxVal = value;
                                            maxRow = kr;
                                            maxCol = kc;
                                        }
                                    }
                                }
                                inputError[maxRow, maxCol] += error[row, col];              // Set error, assume max pooling assign error to the position of the max input value
                                break;
                            case LocalAvgPoolingLayer avgPool:
                                double errorContribution = error[row, col] / Math.Max(1, filterElementCount); // Distribute the error
                                for (int kr = StartY; kr < EndY; kr++) {
                                    for (int kc = StartX; kc < EndX; kc++) {
                                        inputError[kr, kc] += errorContribution;            // Assign the error contribution to each element in the pooling region
                                    }   
                                }
                                break;
                            default:
                                throw new NotImplementedException($"This trainer doesn't support pooling layers of type {layer.GetType()}.");
                        }
                    }
                }

                // Assign the errors for this input features
                batchErrors[featureIndex] = Matrix<double>.Wrap(inputError);
            });

            // Assign the errors for the input features into the batch
            input_errors[batchIndex] = new FeatureSet<double>(batchErrors);
        });
        

        // Pass errors along for next layer
        return new BackpropagationReturns { 
            InputErrors = new BatchedFeatureSet<double>(input_errors),
            Gradient = null,
        };
    }

    public BackpropagationReturns Visit(DropoutLayer layer, BackpropagationArgs args) {
        Matrix<double>? mask = layer.GetSharedMask();
        if (!mask.HasValue) {
            return new BackpropagationReturns {
                InputErrors = args.OutputErrors, // Just pass the errors to the next layer if no mask was assigned
                Gradient = null
            };
        }

        var mask_matrix = mask.Value;

        FeatureSet<double>[] input_errors = new FeatureSet<double>[args.OutputErrors.Batches];
        for (var batchIndex = 0; batchIndex < args.OutputBatch.Batches; batchIndex++) {
            var batch = args.OutputErrors[batchIndex];

            var matrices = new Matrix<double>[batch.Channels];
            for (var i = 0; i < batch.Channels; i++) {
                matrices[i] = batch[i].Hadamard(mask_matrix);
            }
            input_errors[batchIndex] = new FeatureSet<double>(matrices);
        }

        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(input_errors),
            Gradient = null
        };
    }

    public BackpropagationReturns Visit(LayerNorm layer, BackpropagationArgs args) {
        var batches  = args.OutputErrors.Batches;
        var channels = args.OutputErrors.Channels;
        var rows = args.OutputErrors.Rows;
        var columns = args.OutputErrors.Columns;
        var one_over_features = 1.0 / (rows * columns);

        Matrix<double>[] gradient_betas = new Matrix<double>[channels];
        Matrix<double>[] gradient_gammas = new Matrix<double>[channels];
        var input_gradients = new Matrix<double>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<double>[channels];
        }

        var means_per_batch = new double[batches][];
        var variances_per_batch = new double[batches][];
        for (var b = 0; b < batches; b++) {
            layer.ComputeMeansAndVariances(args.InputBatch[b], out var means, out var variances);
            means_per_batch[b] = means;
            variances_per_batch[b] = variances;
        }

        Parallel.For(0, channels, channelIndex => {
            var gamma = layer.Gammas[channelIndex];
            var beta = layer.Betas[channelIndex];

            // dL/dB = Sum_b( dL/dY )
            var gradient_beta = Matrix<double>.Wrap(new double[rows, columns]);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                Matrix<double>.AddInplace(gradient_beta, gradient_beta, args.OutputErrors[batchIndex][channelIndex]);
            }
            gradient_betas[channelIndex] = gradient_beta;

            // dL/dG = Sum_b ( dL/dY * xHat )
            var gradient_gamma = Matrix<double>.Wrap(new double[rows, columns]);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                // Compute xHat from y
                var xhat_k = args.OutputBatch[batchIndex][channelIndex] - beta; // Sucks that I have to re-compute this
                Matrix<double>.ElementWiseInplace(xhat_k, xhat_k, gamma, (xhat, g) => xhat / (g + epsilon));
                
                var loss_wrt_y_k = args.OutputErrors[batchIndex][channelIndex];
                var loss_wrt_y_times_xHat = xhat_k;
                Matrix<double>.HadamardInplace(loss_wrt_y_times_xHat, loss_wrt_y_k, xhat_k);
                Matrix<double>.AddInplace(gradient_gamma, gradient_gamma, loss_wrt_y_times_xHat);
            }
            gradient_gammas[channelIndex] = gradient_gamma;

            // Gradient of L with respect to xHat
            var loss_wrt_xhats = new Matrix<double>[batches];
            var avg_loss_xhat = new double[batches];
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][channelIndex];
                var loss_wrt_xhat = loss_wrt_y_k.Hadamard(gamma);
                loss_wrt_xhats[batch] = loss_wrt_xhat;

                avg_loss_xhat[batch] = loss_wrt_xhat.Average();
            }

            // Gradient of L with respect to x
            for (var batch = 0; batch < batches; batch++) {
                var mean = means_per_batch[batch][channelIndex];
                var variance = variances_per_batch[batch][channelIndex];
                var std = Math.Sqrt(variance + epsilon);
                var scale = 1.0 / std;

                var x = args.InputBatch[batch][channelIndex];
                var loss_wrt_xHat = loss_wrt_xhats[batch];
                var mean_loss_xHat = avg_loss_xhat[batch];
                var loss_wrt_x = new Matrix<double>(x.Rows, x.Columns);

                // Once we have the gradients with respect to the mean and variance, we can compute the gradient with respect to the input tensor `x`: 
                /*
                \[
                \frac{\partial L}{\partial x_{b,c,h,w}} = \frac{1}{\sqrt{\sigma_c^2 + \epsilon}} \left( \frac{\partial L_{b,c,h,w}}{\partial \hat{x}_{b,c,h,w}} - \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} \frac{\partial L_{b,c,h,w}}{\partial \hat{x}_{b,c,h,w}} \right) - \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} \frac{\partial L_{b,c,h,w}}{\partial \hat{x}_{b,c,h,w}} \cdot (x_{b,c,h,w} - \mu_c)
                \]
                */

                // Add term 1 (1/sqrt(variance + e) * (dL/dxHat - mean(dL/dxHat)))
                Matrix<double>.ElementWiseInplace(loss_wrt_x, loss_wrt_x, loss_wrt_xHat, (_, xHat) => scale * (xHat - mean_loss_xHat));

                // Add term 2 (mean(dL/dxHat) * (x - mean))
                Matrix<double>.ElementWiseInplace(loss_wrt_x, loss_wrt_x, x, (old, x) => old - mean_loss_xHat * (x - mean));

                // Save result
                clip(loss_wrt_x, GradientClippingThresholdWeight);
                input_gradients[batch][channelIndex] = loss_wrt_x;
            }
        });

        clip(gradient_gammas, GradientClippingThresholdWeight);
        clip(gradient_betas, GradientClippingThresholdWeight);

        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(input_gradients.Select(x => new FeatureSet<double>(x)).ToArray()),
            Gradient = new NormalizationGradients {
                GammaGradients = gradient_gammas,
                BetaGradients = gradient_betas
            }
        };

        /*// Getting the inputs, outputs, and errors from the BackpropagationArgs
        Matrix<double>[] inputs = (Matrix<double>[])args.Inputs;
        Matrix<double>[] outputs = (Matrix<double>[])args.Outputs;
        Matrix<double>[] errors = args.Errors;
        var channels = inputs.Length;
    
        // Create gradients for gamma and beta
        var gammaGradients = new Matrix<double>[channels];
        var betaGradients = new Matrix<double>[channels];

        // Output list for backpropagated errors
        var inputErrors = new Matrix<double>[channels];

        // Process each channel
        Parallel.For(0, channels, i => {
            var xHat = outputs[i];
            var input = inputs[i];
            var gamma = layer.Gammas[i];
            var error = errors[i];

            gammaGradients[i] = error.Hadamard(xHat);
            betaGradients[i] = error;

            var mean = input.Average();
            var variance = input.Select(v => Math.Pow(v - mean, 2)).Average();
            var denom = Math.Sqrt(variance + epsilon);
            
            // error * gamma * (x-u)/sqrt(variance^2 + e)
            var inputError = error.Hadamard(gamma);                             // error * gamma
            var imean = input.Transform(x => (x - mean) / denom);               // (x - mean) / sqrt(variance^2 + e)
            Matrix<double>.HadamardInplace(inputError, inputError, imean);      // error * gamma * (x-u)/sqrt(variance^2 + e)
            inputErrors[i] = inputError;
        });

        clip(gammaGradients, GradientClippingThresholdWeight);
        clip(betaGradients, GradientClippingThresholdWeight);

        return new BackpropagationReturns {
            InputErrors = inputErrors,
            Gradient = new NormalizationGradients {
                GammaGradients = gammaGradients,
                BetaGradients = betaGradients
            }
        };*/
    }

    public BackpropagationReturns Visit(BatchNorm layer, BackpropagationArgs args) {

        // Terminology
        // x    is input
        // xHat is output before scaling and shifting
        // y    is output after scaling and shifting
        // g    is a scaling value from the gamma matrix
        // b    is a shifting value from the beta matrix
        // u    is the mean 
        // o2   is the variance

        var batches  = args.OutputErrors.Batches;
        var one_over_batches = 1.0 / batches;
        var channels = args.OutputErrors.Channels;
        var channel_width = args.OutputErrors.Columns;
        var channel_height = args.OutputErrors.Rows;

        // Compute mean and variance across the whole batch per channel
        layer.ComputeMeansAndVariances(args.InputBatch, out var means, out var variances);

        var gamma_gradients = new Matrix<double>[channels];
        var beta_gradients = new Matrix<double>[channels];
        var input_gradients = new Matrix<double>[batches][];
        for (var batch = 0; batch < batches; batch++) {
            input_gradients[batch] = new Matrix<double>[channels];
        }

        // Derivations from https://en.wikipedia.org/wiki/Batch_normalization#:~:text=the%20current%20layer.-,Backpropagation,-%5Bedit%5D
        Parallel.For(0, channels, k => {
            var gamma = layer.Gammas[k];
            var beta = layer.Betas[k];

            // Gradient of L with respect to Beta
            // SUM (dl/dy(k))
            var loss_wrt_b = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                Matrix<double>.AddInplace(loss_wrt_b, loss_wrt_b, loss_wrt_y_k);
            }
            beta_gradients[k] = loss_wrt_b;

            // Gradient of L with respect to Gamma
            // SUM (dl/dy(k) * xHat) where xHat = (y - b) / g
            var loss_wrt_g = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                // Compute xHat from y
                // y = g * xHat + b  => xHat = (y - b) / g
                var xhat_k = args.OutputBatch[batch][k] - beta; // Sucks that I have to re-compute this
                Matrix<double>.ElementWiseInplace(xhat_k, xhat_k, gamma, (xhat, g) => xhat / (g + epsilon));

                var loss_wrt_y_k = args.OutputErrors[batch][k];
                var loss_wrt_y_times_xHat = xhat_k;
                Matrix<double>.HadamardInplace(loss_wrt_y_times_xHat, loss_wrt_y_k, xhat_k);
                Matrix<double>.AddInplace(loss_wrt_g, loss_wrt_g, loss_wrt_y_times_xHat);
            }
            gamma_gradients[k] = loss_wrt_g;

            // Gradient of L with respect to xHat
            var loss_wrt_xhats = new Matrix<double>[batches];
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                loss_wrt_xhats[batch] = loss_wrt_y_k.Hadamard(gamma);
            }

            var mean = means[k]; // Mean of the channel over the whole batch
            var variance = variances[k]; // Variance of the channel over the whole batch
            var std = Math.Sqrt(variance + epsilon);
            var scale = 1.0 / std;

            // Gradient of L with respect to variance
            var loss_wrt_variance = new Matrix<double>(channel_height, channel_width);
            var std_three_halves_times_two = -Math.Pow(variance + epsilon, 3.0 / 2.0) * 2.0;
            var one_over_std_three_halves_times_two = 1.0 / std_three_halves_times_two;
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                var x_k = args.InputBatch[batch][k];

                // Term 1
                var x_minus_mean = x_k.Transform(x => x - mean);
                var x_minus_mean_times_loss = x_minus_mean;
                Matrix<double>.HadamardInplace(x_minus_mean_times_loss, loss_wrt_y_k, x_minus_mean);

                // Term 2
                var term1_times_term2 = x_minus_mean_times_loss;
                Matrix<double>.ElementWiseInplace(term1_times_term2,x_minus_mean_times_loss, gamma, (lhs, g) => {
                    return lhs * (g * one_over_std_three_halves_times_two);
                });
            }

            // Gradient of L with respect to mean
            var loss_wrt_mean = new Matrix<double>(channel_height, channel_width);
            var loss_wrt_mean_term1 = new Matrix<double>(channel_height, channel_width);
            var loss_wrt_mean_term2 = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                var x_k = args.InputBatch[batch][k];
            
                // Add term 1
                Matrix<double>.ElementWiseInplace(loss_wrt_mean_term1, loss_wrt_y_k, gamma, (y, g) => y * -g * scale);
                Matrix<double>.AddInplace(loss_wrt_mean, loss_wrt_mean, loss_wrt_mean_term1);

                // Add term 2
                Matrix<double>.ElementWiseInplace(loss_wrt_mean_term2, loss_wrt_variance, x_k, (v, x) => v * one_over_batches * -2.0 * (x - mean));
                Matrix<double>.AddInplace(loss_wrt_mean, loss_wrt_mean, loss_wrt_mean_term2);
            }

            // Gradient of L with respect to x
            for (var batch = 0; batch < batches; batch++) {
                var x = args.InputBatch[batch][k];
                var loss_wrt_xHat = loss_wrt_xhats[batch];
                var loss_wrt_x = new Matrix<double>(x.Rows, x.Columns);

                // Add term 1 (dL/dxHat * scale)
                Matrix<double>.ElementWiseInplace(loss_wrt_x, loss_wrt_x, loss_wrt_xHat, (_, v) => v * scale);

                // Add term 2 (dl/dV * 2 * (x - mean) / M)
                var temp1 = loss_wrt_variance.ElementWise(x, (v, x) => v * 2 * (x - mean) * one_over_batches);
                Matrix<double>.AddInplace(loss_wrt_x, loss_wrt_x, temp1);

                // Add term 3 (dl/du * 1/M)
                Matrix<double>.ElementWiseInplace(loss_wrt_x, loss_wrt_x, loss_wrt_mean, (old, u) => old + u * one_over_batches);

                // Save result
                clip(loss_wrt_x, GradientClippingThresholdWeight);
                input_gradients[batch][k] = loss_wrt_x;
            }

        });

        clip(gamma_gradients, GradientClippingThresholdWeight);
        clip(beta_gradients, GradientClippingThresholdWeight);

        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(input_gradients.Select(x => new FeatureSet<double>(x)).ToArray()),
            Gradient = new NormalizationGradients {
                GammaGradients = gamma_gradients,
                BetaGradients = beta_gradients
            }
        };

        /*
        // Getting the inputs, outputs, and errors from the BackpropagationArgs
        Matrix<double>[] inputs = (Matrix<double>[])args.Inputs;
        Matrix<double>[] outputs = (Matrix<double>[])args.Outputs;
        Matrix<double>[] errors = args.Errors;
        var channels = inputs.Length;
    
        // Create gradients for gamma and beta
        var gammaGradients = new Matrix<double>[channels];
        var betaGradients = new Matrix<double>[channels];

        // Output list for backpropagated errors
        var inputErrors = new Matrix<double>[channels];

        // Compute mean and variance across the whole batch per channel
        var means = new double[args.InputBatch.Channels];
        var variances = new double[args.InputBatch.Channels];

        if (args.InputBatch.Batches < 2) {
            means = (double[])layer.RunningMean;
            variances = (double[])layer.RunningVariance;
        } 
        else {
            for (var channelIndex = 0; channelIndex < variances.Length; channelIndex++) {
                double sum = 0.0;
                double sumSq = 0.0;
                int count = 0;

                foreach (var featureSet in args.InputBatch) {
                    var matrix = featureSet[channelIndex];
                    var rows = matrix.Rows;
                    var cols = matrix.Columns;
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            double value = matrix[i, j];
                            sum += value;
                            sumSq += value * value;
                            count++;
                        }
                    }
                }

                count = Math.Max(count, 1); // Avoid division by zero
                double mean = sum / count;
                double variance = (sumSq / count) - (mean * mean);

                means[channelIndex] = mean;
                variances[channelIndex] = variance;
            }
        }

        // Process each channel
        Parallel.For(0, channels, channelIndex => {
            //// https://en.wikipedia.org/wiki/Batch_normalization#:~:text=the%20current%20layer.-,Backpropagation,-%5Bedit%5D
            //var output = outputs[channelIndex];
            //var input = inputs[channelIndex];
            //var gamma = layer.Gammas[channelIndex];
            //var beta = layer.Betas[channelIndex];
            //var error = errors[channelIndex];
            //var mean = means[channelIndex];
            //var variance = variances[channelIndex];
            //var xHat = (output - beta).ElementWise(gamma, (y, g) => y / g); // Undo shift/scaling of the output
            //var batch_size = 1; // At this present time, there is no batch (backpropagation is handled solo)
//
            //// {\displaystyle {\frac {\partial l}{\partial {\hat {x}}_{i}^{(k)}}}={\frac {\partial l}{\partial y_{i}^{(k)}}}\gamma ^{(k)}},
            //// dl_dxHat = dl_dy * gamma; 
            //var loss_wrt_xhat = error.Hadamard(gamma);
//
            //var loss_wrt_gamma = //sum over batch error.Hadamard(xHat);
            //var loss_wrt_beta = //sum over batch error;
//
            //var loss_wrt_variance = new Matrix<double>(gamma.Rows, gamma.Columns);
            //for (var i = 0; i < batch_size; i++) {
            //    var temp_term = input.Transform(x => x - mean);
            //    Matrix<double>.HadamardInplace(temp_term, error, temp_term); // error * (x - mean)
            //    Matrix<double>.ElementWiseInplace(temp_term, temp_term, gamma, (lhs, g) => {
            //        return lhs * (-g / (2 * Math.Pow(variance + epsilon, 3.0/2.0)));
            //    });
            //    Matrix<double>.AddInplace(loss_wrt_variance, loss_wrt_variance, temp_term);
            //}
            //var loss_wrt_mean = ;
//
            //Matrix<double>.TransformInplace(loss_wrt_xhat, loss_wrt_xhat, (v) => v / Math.Sqrt(variance + epsilon));
            //var loss_wrt_input = loss_wrt_xhat + loss_wrt_variance + loss_wrt_mean;

            var xHat = outputs[channelIndex];
            var input = inputs[channelIndex];
            var gamma = layer.Gammas[channelIndex];
            var error = errors[channelIndex];

            gammaGradients[channelIndex] = error.Hadamard(xHat);
            betaGradients[channelIndex] = error;

            var mean = means[channelIndex];
            var variance = variances[channelIndex];
            var denom = Math.Sqrt(variance + epsilon);
            
            // error * gamma * (x-u)/sqrt(variance^2 + e)
            var inputError = error.Hadamard(gamma);                             // error * gamma
            var imean = input.Transform(x => (x - mean) / denom);               // (x - mean) / sqrt(variance^2 + e)
            Matrix<double>.HadamardInplace(inputError, inputError, imean);      // error * gamma * (x-u)/sqrt(variance^2 + e)
            inputErrors[channelIndex] = inputError;
        });

        clip(gammaGradients, GradientClippingThresholdWeight);
        clip(betaGradients, GradientClippingThresholdWeight);

        return new BackpropagationReturns {
            InputErrors = inputErrors,
            Gradient = new NormalizationGradients {
                GammaGradients = gammaGradients,
                BetaGradients = betaGradients
            }
        };*/
    }

    public BackpropagationReturns Visit(FlatteningLayer layer, BackpropagationArgs args) {
        FeatureSet<double>[] batched_input_errors = new FeatureSet<double>[args.OutputBatch.Batches];

        Parallel.For(0, args.OutputBatch.Batches, batchIndex => {
            var error = args.OutputErrors[batchIndex];
            var input = args.InputBatch[batchIndex];

            Matrix<double>[] input_errors;
            if (input.Channels == 1 && input.Shape == error.Shape) {
                input_errors = error.AsArray();                             // Same shape, no need to reshape
            } else {
                input_errors = error[0].Reshape(                            // Reshape to un-flatten error vector to match the input dimensions (in case next layer is not a fully connected layer)
                    input.Select(x => x.Shape).ToArray())
                .ToArray();
            } 

            batched_input_errors[batchIndex] = new FeatureSet<double>(input_errors);
        });

        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(batched_input_errors),
            Gradient = null
        };
    }

    public BackpropagationReturns Visit(FullyConnectedLayer layer, BackpropagationArgs args) {
        // Form X input matrix for all flattened input batch vectors
        var xT = new Matrix<double>(args.InputBatch.Batches, layer.InputShape.Count);
        for (var i = 0; i < args.InputBatch.Batches; i++) {
            var j = 0;
            foreach (var feature in args.InputBatch[i]) {
                foreach (var value in feature.FlattenRows()) {
                    xT.AsArray()[i,j] = args.InputBatch[i][0][j];
                    j++;
                }
            }
        }

        // Form delta matrix for all batch output errors
        var delta = new Matrix<double>(layer.NeuronCount, args.InputBatch.Batches);
        for (var i = 0; i < args.InputBatch.Batches; i++) {
            var batch_output_errors = args.OutputErrors[i][0]; // This is a column vector (only 1 column)
            for (var j = 0; j < layer.NeuronCount; j++) {
                delta.AsArray()[j, i] = batch_output_errors[j, 0];
            }
        }

        // TODO check dimensions
        // xT is a matrix of Batches x Input Features
        // delta is a matrix of Neurons x Batches
        // layer.Weights is a matrix of size Neurons x Input Features
        // layer.WeightT is a matrix of size Input Features x Neurons

        // Need this to be of size: Neurons x Input Features 
        // Delta * InputTransposed
        // Neurons x Batches * Batches x Input Features => Neurons x Input Features 
        var weight_gradients = delta * xT; // Neurons x Batches * Batches x Input Features => Neurons x Input Features 
        // Need this to be of size: Neurons
        // Delta.Rows
        // Neurons x Batches => Neurons = Delta.Rows
        var bias_gradients = delta.AggregateOverColumns((agg, next) => agg + next, initial: 0.0); // Each column is a batch 

        // Need this to be of size: Batches x Input Features | Input Features x Batches (transposed)
        // WeightsT * Delta
        // Input Features x Neurons * Neurons x Batches => Input Features x Batches
        // equivalent to layer.Weights.Transpose() * delta // using the method below removes the need to allocate a temp matrix
        var input_gradients = layer.Weights.MultiplyTransposedWith(delta); // Input Features x Neurons * Neurons x Batches => Input Features x Batches

        // TODO reshape input_error. Each input feature is a column
        // Each batch must have it's input errors have the same channel/row/column dimensions
        var shaped_input_gradients = new FeatureSet<double>[args.InputBatch.Batches];
        for (var batch = 0; batch < shaped_input_gradients.Length; batch++) {
            var shapes = args.InputBatch[batch].Select(x => x.Shape).ToArray();
            var features = new Matrix<double>[shapes.Length]; 
            var index = 0;

            for (var shapeIndex = 0; shapeIndex < shapes.Length; shapeIndex++) {
                var shape = shapes[shapeIndex];
                var mtx = new Matrix<double>(shape.Rows, shape.Columns);
                for (int row = 0; row < mtx.Rows; row++) {
                    for (int col = 0; col < mtx.Columns; col++) {
                        if (index < mtx.Size)
                            mtx.AsArray()[row, col] = input_gradients[index++, batch];
                        else 
                            mtx.AsArray()[row, col] = 0.0;
                    }
                }
                features[shapeIndex] = mtx;
            }
            shaped_input_gradients[batch] = new FeatureSet<double>(features);
        }

        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(shaped_input_gradients),
            Gradient =  new FullyConnectedGradients {
                WeightGradients = weight_gradients,
                BiasGradients = bias_gradients,
            }
        };

        /*
        // Current layer deltas
        var flattened_inputs = Matrix<double>.Row(args.Inputs.SelectMany(x => x.FlattenRows()).ToArray());
        var output = args.Outputs[0];                                       // Output vector (column)
        var error = args.Errors[0];                                         // Output vector (column)
        //var gradient = layer.ActivationFunction.InvokeDerivative(error, output);   // Gradient of vector elements
        var delta = error;
        //Matrix<double>.HadamardInplace(delta, error, gradient);                               // Delta of vector elements (column)
        
		// Do gradient clipping on the bias gradients
		clip(delta, GradientClippingThresholdBias);						    // Clip using the default clip size
        
        // Compute gradients for weight updates
        // Delta is a column matrix of size (neurons)
        // Flattened inputs is a column matrix of size (input neurons) when transposed it is a row matrix
        Matrix<double> weight_gradients = delta * flattened_inputs;

        // Do gradient clipping on weight gradients
        clip(weight_gradients, GradientClippingThresholdWeight);

        // Compute errors to get passed to the next layer
        // Weights is a (neurons x input) matrix and delta is a column vector of (neurons) elements
        var error_vec = layer.Weights.Transpose() * delta;                  // Vector of errors for the next layer
        if (error_vec.Size != flattened_inputs.Size) {
            throw new ArithmeticException("Unable to reshape errors from fully connected layer to layer input dimensions during backpropagation.");
        }
        Matrix<double>[] input_errors;
        if (args.Inputs.Channels == 1 && args.Inputs[0].Shape == error_vec.Shape) {
            input_errors = [ error_vec ];                                   // Same shape, no need to reshape
        } else {
            input_errors = error_vec.Reshape(                               // Reshape to un-flatten error vector to match the input dimensions (in case next layer is not a fully connected layer)
                args.Inputs.Select(x => x.Shape).ToArray())
            .ToArray();
        }

        return new BackpropagationReturns {
            InputErrors = input_errors,
            Gradient = new FullyConnectedGradients {
                WeightGradients = weight_gradients,
                BiasGradients = delta.ExtractColumn(0),
            }
        };*/
    }

    public BackpropagationReturns Visit(ActivationLayer layer, BackpropagationArgs args) {
        FeatureSet<double>[] batched_input_gradients = new FeatureSet<double>[args.InputBatch.Batches];

        Parallel.For(0, batched_input_gradients.Length, (batchIndex) => {
            var batch = args.InputBatch[batchIndex];

            var input_channels = batch.Channels;
            var input_gradients = new Matrix<double>[input_channels];
            var output_gradients = args.OutputErrors[batchIndex];

            Parallel.For(0, input_channels, channel => {
                var output_gradient = output_gradients[channel];
                var derivative = batch[channel].Transform(layer.ActivationFunction.InvokeDerivative);   // Gradient of vector elements
                var delta = output_gradient.Hadamard(derivative);                               // Delta of vector elements (column)
                input_gradients[channel] = delta;
            });

            batched_input_gradients[batchIndex] = new FeatureSet<double>(input_gradients);
        });

        // Return the gradients to be propagated to the previous layer
        return new BackpropagationReturns {
            InputErrors = new BatchedFeatureSet<double>(batched_input_gradients),
            Gradient = null
        };
    }

    public BackpropagationReturns Visit(SoftmaxLayer layer, BackpropagationArgs args) {
        // Basically we just pass on the errors we know of.
        // This assumes this is the LAST layer and cross-entropy is the loss function
        // Error = predicted - actual
        // This is already the calculation we use for the error given to this layer
        return new BackpropagationReturns {
            InputErrors = args.OutputErrors,
            Gradient = null,
        };
    }

		#region Gradient Clipping
		public bool UseGradientClipping {get; init;}
		public double GradientClippingThresholdWeight {get; init;}
		public double GradientClippingThresholdBias {get; init;}

        const double epsilon = 1e-8;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void sanitize_nan(ref double d) { 
            if (double.IsNaN(d)) {
                d = epsilon;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void clip(ref double d, double clip_threshold) {
			if (!UseGradientClipping)
				return;
		
            sanitize_nan(ref d);

			if (Math.Abs(d) > clip_threshold) {
				d = Math.Sign(d) * clip_threshold;
			}
		}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void clip(Vec<double> vector, double clip_threshold) {
			if (!UseGradientClipping)
				return;
		
			double[] mut = (double[])vector;
			for (var i = 0; i < mut.Length; i++) {
				var val = mut[i];
                sanitize_nan(ref val);
                
				if (Math.Abs(val) > clip_threshold) {
					val = Math.Sign(val) * clip_threshold;
				}
                mut[i] = val;
			}
		}
		
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void clip(Matrix<double> mat, double clip_threshold) {
			if (!UseGradientClipping)
				return;
		
			double[,] mut = (double[,])mat;
			for (var r = 0; r < mut.GetLength(0); r++) {
				for (var c = 0; c < mut.GetLength(1); c++) {
					var val = mut[r, c];
                    sanitize_nan(ref val);

					if (Math.Abs(val) > clip_threshold) {
						val = Math.Sign(val) * clip_threshold;
					}
                    mut[r, c] = val;
				}
			}
		}
		
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void clip(Matrix<double>[] mats, double clip_threshold) {
			if (!UseGradientClipping)
				return;
			
			foreach (var mat in mats)
				clip(mat, clip_threshold);
		}
		#endregion

}

}