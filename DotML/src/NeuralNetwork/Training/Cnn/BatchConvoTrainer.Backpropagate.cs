using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public partial class BatchedConvolutionalBackpropagationEnumerator<TNetwork> {

public struct BackpropagationArgs {
    public Vec<double> TrueLabel;
    public int LayerIndex;
    public Matrix<double>[] Inputs;
    public Matrix<double>[] Outputs;
    public Matrix<double>[] Errors;
}
public abstract class Gradients {}

public class FullyConnectedGradients : Gradients {
    public Matrix<double> WeightGradients;
    public Vec<double> BiasGradients;
}

public class ConvolutionGradients : Gradients {
    public Matrix<double>[][]? FilterKernelGradients;
    public double[]? BiasGradients;
}

public struct BackpropagationReturns {
    public Matrix<double>[] Errors;
    public Gradients? Gradient;
}

public bool UseGradientClipping => backpropagationActions.UseGradientClipping;
public double GradientClippingThresholdWeight => backpropagationActions.GradientClippingThresholdWeight;
public double GradientClippingThresholdBias => backpropagationActions.GradientClippingThresholdBias;

private BackpropagationActions backpropagationActions {get; init;}
public class BackpropagationActions : IConvolutionalLayerVisitor<BatchedConvolutionalBackpropagationEnumerator<TNetwork>.BackpropagationArgs, BatchedConvolutionalBackpropagationEnumerator<TNetwork>.BackpropagationReturns> {

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

    public BackpropagationReturns Visit(ConvolutionLayer layer, BackpropagationArgs args) {
        // Initialize extra storage
        var filterGradients = new Matrix<double>[layer.FilterCount][];
        var biasGradients = new double[layer.FilterCount];
        var paddingRows         = layer.RowsPadding; 
        var paddingColumns      = layer.ColumnsPadding; 

        // Loop through each filter and compute gradients
        Parallel.For(0, layer.FilterCount, filterIndex => {
            var filter              = layer.Filters[filterIndex];                                                    

            var gradient            = new Matrix<double>[args.Inputs.Length];
            var error               = args.Errors[filterIndex];
            var rows                = error.Rows;
            var cols                = error.Columns;
            var output              = args.Outputs[filterIndex];
            const int gradOut       = 1;
            var biasGradient        = 0.0;

            for (var inputIndex = 0; inputIndex < args.Inputs.Length; inputIndex++) {
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
            filterGradients[filterIndex] = gradient;
            biasGradients[filterIndex] = biasGradient;
        });

        var inputErrors = TransposeConvolve(args.Inputs, args.Errors, layer.StrideX, layer.StrideY, paddingRows, paddingColumns, layer.Filters);

        return new BackpropagationReturns {
            Errors = inputErrors,
            Gradient = new ConvolutionGradients {
                FilterKernelGradients = filterGradients,
                BiasGradients = biasGradients,
            }
        };
    }

    public BackpropagationReturns Visit(PoolingLayer layer, BackpropagationArgs args) {
        // Extract inputs, outputs, and errors
        var inputs = args.Inputs;
        var outputs = args.Outputs;
        var errors = args.Errors;

        int batchSize = inputs.Length;
        var inputErrors = new Matrix<double>[batchSize];

        var filterWidth = layer.FilterWidth;
        var filterHeight = layer.FilterHeight;
        var filterElementCount = filterWidth * filterHeight;

        Parallel.For(0, batchSize, b => {
            // Get the input and output for this batch item
            var input = inputs[b];
            var output = outputs[b];
            var error = errors[b];

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

            // Assign the errors for this input
            inputErrors[b] = Matrix<double>.Wrap(inputError);
        });

        // Pass errors along for next layer
        return new BackpropagationReturns { 
            Errors = inputErrors,
            Gradient = null,
        };
    }

    public BackpropagationReturns Visit(DropoutLayer layer, BackpropagationArgs args) {
        Matrix<double>? mask = layer.GetSharedMask();
        if (!mask.HasValue) {
            return new BackpropagationReturns {
                Errors = args.Errors, // Just pass the errors to the next layer if no mask was assigned
                Gradient = null
            };
        }

        var mask_matrix = mask.Value;

        var errors = args.Errors;
        var return_errors = new Matrix<double>[errors.Length];
        for (var i = 0; i < errors.Length; i++) {
            return_errors[i] = errors[i].Hadamard(mask_matrix);
        }

        return new BackpropagationReturns {
            Errors = return_errors,
            Gradient = null
        };
    }

    public BackpropagationReturns Visit(FlatteningLayer layer, BackpropagationArgs args) {
        var error = args.Errors[0];  

        Matrix<double>[] input_errors;
        if (args.Inputs.Length == 1 && args.Inputs[0].Shape == error.Shape) {
            input_errors = [ error ];                                   // Same shape, no need to reshape
        } else {
            input_errors = error.Reshape(                               // Reshape to un-flatten error vector to match the input dimensions (in case next layer is not a fully connected layer)
                args.Inputs.Select(x => x.Shape).ToArray())
            .ToArray();
        } 

        return new BackpropagationReturns {
            Errors = input_errors,
            Gradient = null
        };
    }

    public BackpropagationReturns Visit(FullyConnectedLayer layer, BackpropagationArgs args) {
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
        if (args.Inputs.Length == 1 && args.Inputs[0].Shape == error_vec.Shape) {
            input_errors = [ error_vec ];                                   // Same shape, no need to reshape
        } else {
            input_errors = error_vec.Reshape(                               // Reshape to un-flatten error vector to match the input dimensions (in case next layer is not a fully connected layer)
                args.Inputs.Select(x => x.Shape).ToArray())
            .ToArray();
        }

        return new BackpropagationReturns {
            Errors = input_errors,
            Gradient = new FullyConnectedGradients {
                WeightGradients = weight_gradients,
                BiasGradients = delta.ExtractColumn(0),
            }
        };
    }

    public BackpropagationReturns Visit(ActivationLayer layer, BackpropagationArgs args) {
        var input_channels = args.Inputs.Length;
        var input_gradients = new Matrix<double>[input_channels];
        var output_gradients = args.Errors;

        // Compute the gradient of the error with respect to the inputs
        Parallel.For(0, input_channels, channel => {
            var output_gradient = output_gradients[channel];
            var derivative = args.Outputs[channel].Transform(layer.ActivationFunction.InvokeDerivative);   // Gradient of vector elements
            var delta = output_gradient.Hadamard(derivative);                               // Delta of vector elements (column)
            input_gradients[channel] = delta;
        });

        // Return the gradients to be propagated to the previous layer
        return new BackpropagationReturns {
            Errors = input_gradients,
            Gradient = null
        };
    }

    public BackpropagationReturns Visit(SoftmaxLayer layer, BackpropagationArgs args) {
        // Basically we just pass on the errors we know of.
        // This assumes this is the LAST layer and cross-entropy is the loss function
        // Error = predicted - actual
        // This is already the calculation we use for the error given to this layer
        return new BackpropagationReturns {
            Errors = args.Errors,
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