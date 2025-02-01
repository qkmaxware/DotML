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

public abstract class Gradients { }

public class FullyConnectedGradients : Gradients {
    public Matrix<double> WeightGradients;
    public Vec<double> BiasGradients;
}

public class ConvolutionGradients : Gradients {
    public Matrix<double>[][]? FilterKernelGradients;
    public double[]? BiasGradients;
}

public class DepthwiseConvolutionGradients : Gradients {
    public Matrix<double>[]? KernelGradients;
}

public class NormalizationGradients : Gradients {
    public Matrix<double>[]? GammaGradients;
    public Matrix<double>[]? BetaGradients;
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
            clip(batch_features, GradientClippingThresholdWeight);
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        });

        // TOD gradient clipping

        return new BatchedFeatureSet<double>(result_features);
    }

    public BackpropagationReturns Visit(ConvolutionLayer layer, BackpropagationArgs args) {
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
                kernels[i] = new Matrix<double>(kernel.Rows, kernel.Columns);
            }
            filter_kernel_gradients[f] = kernels;
        }
        // TODO gradient clipping

        Parallel.For(0, filter_count, filterIndex => {
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
                    var kernel_gradient = kernel_gradients[channelIndex]; // make this a mutable reference

                    for (int outY = 0; outY < error_rows; outY++) {
                        var startY = outY * layer.StrideY - paddingRows;

                        for (int outX = 0; outX < error_cols; outX++) {
                            var startX = outX * layer.StrideX - paddingColumns;
                            var error = output_errors[outY, outX];

                            for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                                var inY = startY + kernelY;
                                if (inY < 0 || inY >= input_channel.Rows) continue; // Skip out-of-bounds rows

                                for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                    var inX = startX + kernelX;
                                    if (inX < 0 || inX >= input_channel.Columns) continue; // Skip out-of-bounds columns

                                    kernel_gradient[kernelY, kernelX] += input_channel[inY, inX] * error;
                                }
                            }
                        }
                    }
                }
            }

            clip(kernel_gradients, GradientClippingThresholdWeight);
        });

        clip(bias_gradients, GradientClippingThresholdBias);

        return new BackpropagationReturns {
            InputErrors = TransposeConvolve2(layer, args),
            Gradient = new ConvolutionGradients {
                FilterKernelGradients = filter_kernel_gradients,
                BiasGradients = bias_gradients,
            }
        };
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
                var input_error = new Matrix<double>(input_height, input_width);
                    
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

                batch_features[channelIndex] = input_error;
            } 
            clip(batch_features, GradientClippingThresholdWeight);
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        });

        return new BatchedFeatureSet<double>(result_features);
    }

    public BackpropagationReturns Visit(DepthwiseConvolutionLayer layer, BackpropagationArgs args) {
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
            var kernel = filter[i];
            var kernel_gradient_matrix = new Matrix<double>(kernel.Rows, kernel.Columns);

            for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) { // This can't be as two batches can access the same channel at the same time
                var input_features = args.InputBatch[batchIndex]; // This is the number of channels, not the number of filters
                var output_errors = args.OutputErrors[batchIndex][i];

                var error_rows = output_errors.Rows;
                var error_cols = output_errors.Columns;

                var input_channel = input_features[i]; // Matrix 2D
                var kernel_gradient = kernel_gradient_matrix; // make this a mutable reference

                for (int outY = 0; outY < error_rows; outY++) {
                    var startY = outY * layer.StrideY - paddingRows;

                    for (int outX = 0; outX < error_cols; outX++) {
                        var startX = outX * layer.StrideX - paddingColumns;
                        var error = output_errors[outY, outX];

                        for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                            var inY = startY + kernelY;
                            if (inY < 0 || inY >= input_channel.Rows) continue; // Skip out-of-bounds rows

                            for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                var inX = startX + kernelX;
                                if (inX < 0 || inX >= input_channel.Columns) continue; // Skip out-of-bounds columns

                                kernel_gradient[kernelY, kernelX] += input_channel[inY, inX] * error;
                            }
                        }
                    }
                }
            }

            kernel_gradients[i] = kernel_gradient_matrix;
        });

        clip(kernel_gradients, GradientClippingThresholdWeight);

        return new BackpropagationReturns {
            InputErrors = DepthwiseTransposeConvolve2(layer, args),
            Gradient = new DepthwiseConvolutionGradients {
                KernelGradients = kernel_gradients
            }
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
                var inputError = new Matrix<double>(input.Rows, input.Columns);

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
                batchErrors[featureIndex] = inputError;
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
                matrices[i] = batch[i].HadamardWith(mask_matrix);
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
            var gradient_beta = new Matrix<double>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                gradient_beta.AddWithInplace((args.OutputErrors[batchIndex])[channelIndex]);
            }
            gradient_betas[channelIndex] = gradient_beta;

            // dL/dG = Sum_b ( dL/dY * xHat )
            var gradient_gamma = new Matrix<double>(rows, columns);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                // Compute xHat from y
                var xhat_k = args.OutputBatch[batchIndex][channelIndex] - beta; // Sucks that I have to re-compute this
                xhat_k.ElementWiseInplace(gamma, (xhat, g) => xhat / (g + epsilon));
                
                var loss_wrt_y_k = args.OutputErrors[batchIndex][channelIndex];
                var loss_wrt_y_times_xHat = xhat_k;
                xhat_k.HadamardWithInplace(loss_wrt_y_k);
                gradient_gamma.AddWithInplace(loss_wrt_y_times_xHat);
            }
            gradient_gammas[channelIndex] = gradient_gamma;

            // Gradient of L with respect to xHat
            var loss_wrt_xhats = new Matrix<double>[batches];
            var avg_loss_xhat = new double[batches];
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][channelIndex];
                var loss_wrt_xhat = loss_wrt_y_k.HadamardWith(gamma);
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
                loss_wrt_x.ElementWiseInplace(loss_wrt_xHat, (_, xHat) => scale * (xHat - mean_loss_xHat));

                // Add term 2 (mean(dL/dxHat) * (x - mean))
                loss_wrt_x.ElementWiseInplace(x, (old, x) => old - mean_loss_xHat * (x - mean));

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
        Parallel.For(0, channels, (int k) => {
            var gamma = layer.Gammas[k];
            var beta = layer.Betas[k];

            // Gradient of L with respect to Beta
            // SUM (dl/dy(k))
            var loss_wrt_b = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                loss_wrt_b.AddWithInplace(loss_wrt_y_k);
            }
            beta_gradients[k] = loss_wrt_b;

            // Gradient of L with respect to Gamma
            // SUM (dl/dy(k) * xHat) where xHat = (y - b) / g
            var loss_wrt_g = new Matrix<double>(channel_height, channel_width);
            for (var batch = 0; batch < batches; batch++) {
                // Compute xHat from y
                // y = g * xHat + b  => xHat = (y - b) / g
                var xhat_k = args.OutputBatch[batch][k] - beta; // Sucks that I have to re-compute this
                xhat_k.ElementWiseInplace(gamma, (xhat, g) => xhat / (g + epsilon));

                var loss_wrt_y_k = args.OutputErrors[batch][k];
                var loss_wrt_y_times_xHat = xhat_k;
                xhat_k.HadamardWithInplace(loss_wrt_y_k);
                loss_wrt_g.AddWithInplace(loss_wrt_y_times_xHat);
            }
            gamma_gradients[k] = loss_wrt_g;

            // Gradient of L with respect to xHat
            var loss_wrt_xhats = new Matrix<double>[batches];
            for (var batch = 0; batch < batches; batch++) {
                var loss_wrt_y_k = args.OutputErrors[batch][k];
                loss_wrt_xhats[batch] = loss_wrt_y_k.HadamardWith(gamma);
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
                x_minus_mean.HadamardWithInplace(loss_wrt_y_k);

                // Term 2
                var term1_times_term2 = x_minus_mean_times_loss;
                x_minus_mean_times_loss.ElementWiseInplace(gamma, (lhs, g) => {
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
                loss_wrt_mean.AddWithInplace(loss_wrt_mean_term1);

                // Add term 2
                Matrix<double>.ElementWiseInplace(loss_wrt_mean_term2, loss_wrt_variance, x_k, (v, x) => v * one_over_batches * -2.0 * (x - mean));
                loss_wrt_mean.AddWithInplace(loss_wrt_mean_term2);
            }

            // Gradient of L with respect to x
            for (var batch = 0; batch < batches; batch++) {
                var x = args.InputBatch[batch][k];
                var loss_wrt_xHat = loss_wrt_xhats[batch];
                var loss_wrt_x = new Matrix<double>(x.Rows, x.Columns);

                // Add term 1 (dL/dxHat * scale)
                loss_wrt_x.ElementWiseInplace(loss_wrt_xHat, (_, v) => v * scale);

                // Add term 2 (dl/dV * 2 * (x - mean) / M)
                var temp1 = loss_wrt_variance.ElementWise(x, (v, x) => v * 2 * (x - mean) * one_over_batches);
                loss_wrt_x.AddWithInplace(temp1);

                // Add term 3 (dl/du * 1/M)
                loss_wrt_x.ElementWiseInplace(loss_wrt_mean, (old, u) => old + u * one_over_batches);

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
                    input.Select(x => x.Shape))
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
            var batch = args.InputBatch[i];
            foreach (var feature in batch) {
                var k = 0;
                foreach (var value in feature.FlattenRows()) {
                    xT[i,j++] = feature[k++];
                }
            }
        }

        // Form delta matrix for all batch output errors
        var delta = new Matrix<double>(layer.NeuronCount, args.InputBatch.Batches);
        for (var i = 0; i < args.InputBatch.Batches; i++) {
            var batch_output_errors = args.OutputErrors[i][0]; // This is a column vector (only 1 column)
            for (var j = 0; j < layer.NeuronCount; j++) {
                delta[j, i] = batch_output_errors[j, 0];
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
        clip(weight_gradients, GradientClippingThresholdWeight);
        // Need this to be of size: Neurons
        // Delta.Rows
        // Neurons x Batches => Neurons = Delta.Rows
        var bias_gradients = delta.AggregateOverColumns((agg, next) => agg + next, initial: 0.0); // Each column is a batch 
        clip(bias_gradients, GradientClippingThresholdBias);

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
                var rows = shape.Rows;
                var cols = shape.Columns;
                var mtx = new Matrix<double>(shape.Rows, shape.Columns);
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        if (index < mtx.Size)
                            mtx[row, col] = input_gradients[index++, batch];
                        else 
                            mtx[row, col] = 0.0;
                    }
                }
                var result = mtx;
                clip(result, GradientClippingThresholdWeight);
                features[shapeIndex] = result;
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
                var delta = output_gradient.HadamardWith(derivative);                               // Delta of vector elements (column)
                clip(delta, GradientClippingThresholdWeight);
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

    public BackpropagationReturns Visit(InputCapture layer, BackpropagationArgs args) {
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

            mat.Apply((value) => {
                var val = value;
                sanitize_nan(ref val);
                if (Math.Abs(val) > clip_threshold) {
                    val = Math.Sign(val) * clip_threshold;
                }
                return val;
            });
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