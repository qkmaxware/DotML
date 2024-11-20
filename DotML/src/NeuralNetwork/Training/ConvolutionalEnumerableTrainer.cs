using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

#region Enumerable
/// <summary>
/// Simple Neural Network trainer based on backpropagation
/// </summary>
/// <typeparam name="TNetwork">type of network to train (convolutional network)</typeparam>
public class ConvolutionalEnumerableBackpropagationTrainer<TNetwork>
    : IEnumerableTrainer<TNetwork>
where TNetwork : ConvolutionalFeedforwardNetwork
{
    /// <summary>
    /// Number of epochs (default: 250)
    /// </summary>
    public int Epochs {get; set;} = 250;

    /// <summary>
    /// Learning rate for changes to weights/biases, usually between 0.0001 and 0.1. (default: 0.1)
    /// </summary>
    public double LearningRate {get; set;} = 0.1;

    /// <summary>
    /// Enable to perform clipping of gradients with magnitudes larger than the GradientClipThreshold value. (default: false)
    /// </summary>
    public bool EnableGradientClipping {get; set;} = false;
    /// <summary>
    /// Threshold to compare gradients against when EnableGradientClipping is set. Sets both ClippingThresholdSynapses and ClippingThresholdBiases properties.
    /// </summary>
    public double ClippingThreshold {
        set {
            ClippingThresholdSynapses = value;
            ClippingThresholdBiases = value;
        }
    }
    /// <summary>
    /// Threshold to compare synapse gradients against when EnableGradientClipping is set. Values between 1.0 and 10.0 are common. (default 10.0)
    /// </summary>
    public double ClippingThresholdSynapses {get; set;} = 10;
    /// <summary>
    /// Threshold to compare bias gradients against when EnableGradientClipping is set. Values between 1.0 and 5.0 are common. (default 5.0)
    /// </summary>
    public double ClippingThresholdBiases {get; set;} = 5;

    /// <summary>
    /// Flag to indicate if training should stop before the MaxEpochs has been reached if the network has achieved the desired accuracy (default: true)
    /// </summary>
    public bool EarlyStop {get; set;} = true;

    /// <summary>
    /// The accuracy that is used as a condition to stop training if EarlyStop is set to true (default: 0.1)
    /// </summary>
    public double EarlyStopAccuracy {
        get => _earlyStopAccuracy;
        set {
            _earlyStopAccuracy = Math.Max(0, value);
        }
    }
    private double _earlyStopAccuracy = 0.1;

    /// <summary>
    /// The loss function used in network accuracy evaluation (default: MSE)
    /// </summary>
    public LossFunction LossFunction {get; set;} = LossFunctions.MeanSquaredError;

    /// <summary>
    /// Network initialization strategy (defaults: NormalXavierInitialization)
    /// </summary>
    public IInitializer NetworkInitializer {get; set;} = new NormalXavierInitialization();

    public IEpochEnumerator<TNetwork> EnumerateTraining(TNetwork network, IEnumerator<TrainingPair> dataset, IEnumerator<TrainingPair> validation) {
        return new ConvolutionalBackpropagationEnumerator<TNetwork>(
            network,
            dataset,
            validation,

            earlyStop:              this.EarlyStop,
            earlyStopThreshold:     this.EarlyStopAccuracy,
            lossFunction:           this.LossFunction,

            networkInitializer:     this.NetworkInitializer,

            epochs:                 this.Epochs,
            learningRate:           this.LearningRate,

            useClipping:            this.EnableGradientClipping,
            clipThresholdWeight:    this.ClippingThresholdSynapses,
            clipThresholdBias:      this.ClippingThresholdBiases
        );
    }

    public void Train(TNetwork network, IEnumerator<TrainingPair> dataset, IEnumerator<TrainingPair> validation) {
        EnumerateTraining(network, dataset, validation).MoveToEnd();
    }
}
#endregion

#region Enumerator
public class ConvolutionalBackpropagationEnumerator<TNetwork> 
    : IEpochEnumerator<TNetwork>,
    IConvolutionalLayerVisitor<ConvolutionalBackpropagationEnumerator<TNetwork>.BackpropagationArgs, ConvolutionalBackpropagationEnumerator<TNetwork>.BackpropagationReturns>, 
    IConvolutionalLayerVisitor<ConvolutionalBackpropagationEnumerator<TNetwork>.LayerUpdateArgs, ConvolutionalBackpropagationEnumerator<TNetwork>.LayerUpdateReturns>
    where TNetwork : ConvolutionalFeedforwardNetwork
{
    public int CurrentEpoch {get; private set;}
    private int CurrentUpdateTimestep;
    public int MaxEpochs {get; init;}
    public double LearningRate {get; init;}

    public TNetwork Current {get; private set;}
    object IEnumerator.Current => Current;

    private IEnumerator<TrainingPair> training;
    private IEnumerator<TrainingPair> validation;

    public bool EnableEarlyStop {get; private set;}
    public double EarlyStopThreshold {get; private set;}
    public LossFunction LossFunction {get; private set;}

    public IInitializer NetworkInitializer {get; private set;}

    public ConvolutionalBackpropagationEnumerator(
        TNetwork network,
        IEnumerator<TrainingPair> training,
        IEnumerator<TrainingPair> validation,

        bool earlyStop,
        double earlyStopThreshold,
        LossFunction lossFunction,

        IInitializer networkInitializer,

        int epochs,
        double learningRate,

        bool useClipping,
        double clipThresholdWeight,
        double clipThresholdBias
    ) {
        this.Current = network;
        this.training = training;
        this.validation = validation;

        this.inputs   = new Matrix<double>[Current.LayerCount][]; // The inputs to each layer
        this.outputs  = new Matrix<double>[Current.LayerCount][]; // The outputs from each layer
        this.layer_gradients = new Gradients?[Current.LayerCount];

        this.MaxEpochs = Math.Max(0, epochs);
        this.LearningRate = Math.Abs(learningRate);

        this.EnableEarlyStop = earlyStop;
        this.EarlyStopThreshold = earlyStopThreshold;
        this.LossFunction = lossFunction;
        this.NetworkInitializer = networkInitializer;

        this.UseGradientClipping = useClipping;
        this.GradientClippingThresholdWeight = clipThresholdWeight;
        this.GradientClippingThresholdBias = clipThresholdBias;

        Reset();
    }

    public void Dispose() { }

    Matrix<double>[][] inputs;
    Matrix<double>[][] outputs;
    Gradients?[] layer_gradients;

    public void Reset() {
        this.CurrentEpoch = 0;
        this.CurrentUpdateTimestep = 0;
        this.training.Reset();
        this.validation.Reset();

        this.Current.Initialize(this.NetworkInitializer);

        this.inputs   = new Matrix<double>[Current.LayerCount][]; // The inputs to each layer
        this.outputs  = new Matrix<double>[Current.LayerCount][]; // The outputs from each layer
        this.layer_gradients = new Gradients?[Current.LayerCount];
    }

    public bool MoveNext() {
        if (CurrentEpoch >= MaxEpochs) {
            return false;
        }

        TrainingStep();

        // TODO early STOP
        bool stopEarly = false;
        if (EnableEarlyStop) {
            validation.Reset();
            var sum_error = 0d; var count = 0;
            var max_error = double.MinValue;
            var all_less_threshold = true;
            while (validation.MoveNext()) {
                var input = validation.Current.Input;
                var expected = validation.Current.Output; 
                var actual = Current.PredictSync(input);
                
                var loss = LossFunction(expected, actual);
                sum_error += loss;
                max_error = Math.Max(max_error, loss);
                all_less_threshold &= loss < EarlyStopThreshold;
                count++;
            }
            var avg_error = sum_error / Math.Max(1, count);
            if (max_error <= EarlyStopThreshold) {
                stopEarly = true;
            }
        }

        CurrentEpoch += 1;
        var epochs_finished = CurrentEpoch >= MaxEpochs;
        
        var is_done = epochs_finished || stopEarly;
        return !is_done;
    }

    private void TrainingStep() {
        training.Reset();
        var pos = 0;
        while(training.MoveNext()) {
            var input = training.Current.Input.Shape(
                new Shape(Current.InputImageHeight, Current.InputImageWidth), 
                Current.InputImageChannels
            ).ToArray();
            var expected = training.Current.Output; // TODO Output of a FULLY CONNECTED LAYER SHOULDtm BE A COLUMN MATRIX THIS IS NOT GUARANTEED TO BE THE CASE IF THE NETWORK DOESN'T USE THEM

            try {

            // Forward pass (simulated, duplicate of ConvolutionalFeedforwardNetwork.PredictSync with some additional tracking)
            Matrix<double>[] layer_input = input;
            for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                inputs[layerIndex] = layer_input;
                var layer = Current.GetLayer(layerIndex);
                var layer_output = layer.EvaluateSync(layer_input);
                outputs[layerIndex] = layer_output;
                layer_input = layer_output;
            }
            var actual = layer_input;

            // Backwards pass
            var @true = expected.Shape(outputs[^1].Select(x => x.Shape).ToArray());
            var errors = outputs[^1].Zip(@true).Select(x => x.First-x.Second).ToArray();
            //var errors = outputs[^1][0] - expected;
            var backprop_args = new BackpropagationArgs();
            backprop_args.Errors = errors;
            backprop_args.TrueLabel = expected;
            for (var layerIndex = Current.LayerCount - 1; layerIndex >= 0; layerIndex--) {
                backprop_args.Inputs = inputs[layerIndex];
                backprop_args.Outputs = outputs[layerIndex];

                var layer = Current.GetLayer(layerIndex);
                backprop_args.LayerIndex = layerIndex;
                var returns = layer.Visit<BackpropagationArgs, BackpropagationReturns>(this, backprop_args); // Backpropagation is different for each layer kind, leverage polymorphism
                // TODO maybe store additional details like gradients here for us to apply updates in the next step
                layer_gradients[layerIndex] = returns.Gradient;
                backprop_args.Errors = returns.Errors;
            }

            // Update weights
            var update_args = new LayerUpdateArgs();
            update_args.UpdateTimestep = CurrentUpdateTimestep;
            for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                update_args.Gradients = layer_gradients[layerIndex];

                var layer = Current.GetLayer(layerIndex);
                update_args.LayerIndex = layerIndex;
                layer.Visit<LayerUpdateArgs, LayerUpdateReturns>(this, update_args);
            }
            CurrentUpdateTimestep++;
            pos++;
            } catch (Exception e) {
                // DUMP EVERYTHING
                /*for (var i = 0; i < Current.LayerCount; i++) {
                    var layer = Current.GetLayer(i);
                    Console.WriteLine($"Epoch {CurrentEpoch}, Iteration {pos}");
                    Console.WriteLine("Layer " + i+ ": " + layer.GetType().Name);
                    Console.WriteLine("Inputs: "); 
                    var ips = inputs[i];
                    for (var j = 0; j < ips.Length; j++) {
                        Console.Write(" - "); Console.WriteLine(ips[j]);
                    }
                    Console.WriteLine("Outputs: ");
                    var ops = outputs[i];
                    for (var j = 0; j < ops.Length; j++) {
                        Console.Write(" - "); Console.WriteLine(ops[j]);
                    }
                    Console.WriteLine();
                }*/
                throw new ConvolutionalBackpropagationException(Current, e);
            }
        }
    }

    #region Backpropagation
    public struct BackpropagationArgs {
        public Vec<double> TrueLabel;
        public int LayerIndex;
        public Matrix<double>[] Inputs;
        public Matrix<double>[] Outputs;
        public Matrix<double>[] Errors;
    }
    public abstract class Gradients {}

    public class FullyConnectedGradients : Gradients {
        public Matrix<double>? WeightGradients;
        public Vec<double>? BiasGradients;
    }

    public class ConvolutionGradients : Gradients {
        public Matrix<double>[][]? FilterKernelGradients;
		public double[]? BiasGradients;
    }

    public struct BackpropagationReturns {
        public Matrix<double>[] Errors;
        public Gradients? Gradient;
    }

    // Helper to perform tranpose convolution
    private Matrix<double>[] TransposeConvolve(Matrix<double>[] inputs, Matrix<double>[] errors, int strideX, int strideY, Padding padding, IList<ConvolutionFilter> filters) {
        var inputChannels = inputs.Length;
        var inputErrors = new Matrix<double>[inputChannels];

        for (var channel = 0; channel < inputChannels; channel++) {
            var input = inputs[channel];
            var inputError = new double[input.Rows, input.Columns];

            // Calculate the input errors for each filter
            for (var filterIndex = 0; filterIndex < filters.Count; filterIndex++) {
                var filter = filters[filterIndex];
                var filterRows = filter.Height;
                var filterColumns = filter.Width;
                var paddingRows         = padding == Padding.Same ? (filterRows - 1) / 2 : 0;
                var paddingColumns      = padding == Padding.Same ? (filterColumns - 1) / 2 : 0;
                var error = errors[filterIndex];
                var kernel = filter[channel];

                // Iterate over output errors to compute gradient
                for (int outY = 0; outY < error.Rows; outY++) {
                    for (int outX = 0; outX < error.Columns; outX++) {
                        // Place the error at the corresponding position in the input space
                        for (int ky = 0; ky < filterRows; ky++) {
                            for (int kx = 0; kx < filterColumns; kx++) {
                                var inY = outY * strideY - paddingRows + ky;
                                var inX = outX * strideX - paddingColumns + kx;

                                if (inY >= 0 && inY < input.Rows && inX >= 0 && inX < input.Columns) {
                                    inputError[inY, inX] += error[outY, outX] * kernel[ky, kx];
                                }
                            }
                        }
                    }
                }
            }

            inputErrors[channel] = Matrix<double>.Wrap(inputError);
        }

        return inputErrors;
    }

    public BackpropagationReturns Visit(ConvolutionLayer layer, BackpropagationArgs args) {
        // Initialize extra storage
        var filterGradients = new Matrix<double>[layer.FilterCount][];
        var biasGradients = new double[layer.FilterCount];

        // Loop through each filter and compute gradients
        for (var filterIndex = 0; filterIndex < layer.FilterCount; filterIndex++) {
            var filter = layer.Filters[filterIndex];
            var filterRows          = filter.Height;                                                     
            var filterColumns       = filter.Width;                                                        
            var paddingRows         = layer.Padding == Padding.Same ? (filterRows - 1) / 2 : 0; 
            var paddingColumns      = layer.Padding == Padding.Same ? (filterColumns - 1) / 2 : 0; 

            var gradient = new Matrix<double>[args.Inputs.Length];
            var error = args.Errors[filterIndex];
            var output = args.Outputs[filterIndex];
            var gradOut = layer.ActivationFunction.InvokeDerivative(error, output);
            var biasGradient = 0.0;

            for (var inputIndex = 0; inputIndex < args.Inputs.Length; inputIndex++) {
                var input = args.Inputs[inputIndex];
                var kernel = filter[inputIndex];
                var kernelGradient = new double[kernel.Rows, kernel.Columns];

                // Slide the kernel over the error map computing the correlation
                for (int outY = 0; outY < error.Rows; outY++) {
                    for (int outX = 0; outX < error.Columns; outX++) {
                        var slope = gradOut[outY, outX];
                        var pixel_error = error[outY, outX];
                        var errorContribution = slope * pixel_error;
                        biasGradient += errorContribution;

                        for (var ky = 0; ky < kernel.Rows; ky++) {
                            for (var kx = 0; kx < kernel.Columns; kx++) {
                                var inY = outY * layer.StrideY - paddingRows + ky;
                                var inX = outX * layer.StrideX - paddingColumns + kx;

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
        }

        var inputErrors = TransposeConvolve(args.Inputs, args.Errors, layer.StrideX, layer.StrideY, layer.Padding, layer.Filters);

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

        for (int b = 0; b < batchSize; b++) {
            // Get the input and output for this batch item
            var input = inputs[b];
            var output = outputs[b];
            var error = errors[b];

            // Initialize the error matrix for the input
            var inputError = new double[input.Rows, input.Columns];

            // Loop over output
            for (int row = 0; row < output.Rows; row++) {
                for (int col = 0; col < output.Columns; col++) {
                    // Calculate the region of the input corresponding to this output
                    (int StartX, int StartY, int EndX, int EndY) region = (
                        col * layer.StrideX, 
                        row * layer.StrideY,
                        Math.Min(col * layer.StrideX + filterWidth, input.Columns),
                        Math.Min(row * layer.StrideY + filterHeight, input.Rows)
                    );

                    // Loop over input values where the filter is applied
                    switch (layer) {
                        case LocalMaxPoolingLayer maxPool:
                            int maxRow = 0, maxCol = 0; double maxVal = double.MinValue; // Values for max pooling
                            for (int kr = region.StartY; kr < region.EndY; kr++) {
                                for (int kc = region.StartX; kc < region.EndX; kc++) {
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
                            for (int kr = region.StartY; kr < region.EndY; kr++) {
                                for (int kc = region.StartX; kc < region.EndX; kc++) {
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
        }

        // Pass errors along for next layer
        return new BackpropagationReturns { 
            Errors = inputErrors,
            Gradient = null,
        };
    }

    public BackpropagationReturns Visit(FullyConnectedLayer layer, BackpropagationArgs args) {
        // Current layer deltas
        var flattened_inputs = Matrix<double>.Column(args.Inputs.SelectMany(x => x.FlattenRows()).ToArray());
        var output = args.Outputs[0];                                       // Output vector (column)
        var error = args.Errors[0];                                         // Output vector (column)
        var gradient = layer.ActivationFunction.InvokeDerivative(error, output);   // Gradient of vector elements
        var delta = error.Hadamard(gradient);                               // Delta of vector elements (column)

		// Do gradient clipping on the bias gradients
		clip(delta, GradientClippingThresholdBias);						    // Clip using the default clip size
        
        // Compute gradients for weight updates
        // Delta is a column matrix of size (neurons)
        // Flattened inputs is a column matrix of size (input neurons) when transposed it is a row matrix
        Matrix<double> weight_gradients = delta * flattened_inputs.Transpose();

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

    #endregion

    #region Weight Updates
    public struct LayerUpdateArgs {
        public int UpdateTimestep;
        public int LayerIndex;
        public Gradients? Gradients;
    }
    public struct LayerUpdateReturns {
        // Empty but left in case we need this in the future
    }

    public LayerUpdateReturns Visit(ConvolutionLayer layer, LayerUpdateArgs args) {
        if (args.Gradients is null || args.Gradients is not ConvolutionGradients gradients)
            throw new NullReferenceException(nameof(args.Gradients));

        
        if (gradients.FilterKernelGradients is not null) {
            for (var filterIndex = 0; filterIndex < layer.FilterCount; filterIndex++) {
                var filter = layer.Filters[filterIndex];

                for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                    var kernel = filter[kernelIndex];
                    var grads = gradients.FilterKernelGradients[filterIndex][kernelIndex];
                    foreach (var elem in grads) {
                        if (double.IsNaN(elem)) {
                            throw new ArithmeticException($"NaN detected in kernel gradients for Filter {filterIndex}, Kernel {kernelIndex} while updating weights of a ConvolutionalLayer");
                        }
                    }
                    var next_kernel = kernel - LearningRate * grads;
                    filter[kernelIndex] = next_kernel;
                }
            }
        }

        if (gradients.BiasGradients is not null) {
            foreach (var elem in gradients.BiasGradients) {
                if (double.IsNaN(elem)) {
                    throw new ArithmeticException($"NaN detected in bias gradients while updating weights of a ConvolutionalLayer");
                }
            }
			for (var filterIndex = 0; filterIndex < layer.FilterCount; filterIndex++) {
				var filter = layer.Filters[filterIndex];
                var grads = gradients.BiasGradients[filterIndex];
				filter.Bias -= LearningRate * grads;
			}
		}

        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(PoolingLayer layer, LayerUpdateArgs args) {
        // Do nothing for gradient updates on the pooling layer
        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(FullyConnectedLayer layer, LayerUpdateArgs args) {
        if (args.Gradients is null || args.Gradients is not FullyConnectedGradients gradients)
            throw new NullReferenceException(nameof(args.Gradients));

        if (gradients.WeightGradients.HasValue) {
            layer.Weights = layer.Weights - LearningRate * gradients.WeightGradients.Value;
        
            foreach (var elem in gradients.WeightGradients.Value) {
                if (double.IsNaN(elem)) {
                    throw new ArithmeticException($"NaN detected in weight gradients for a FullyConnectedLayer");
                }
            }
        }
        
        if (gradients.BiasGradients.HasValue) {
            layer.Biases = layer.Biases - LearningRate * gradients.BiasGradients.Value;
            foreach (var elem in gradients.BiasGradients.Value) {
                if (double.IsNaN(elem)) {
                    throw new ArithmeticException($"NaN detected in bias gradients for a FullyConnectedLayer");
                }
            }
        }
        
        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(SoftmaxLayer layer, LayerUpdateArgs args) { 
        // Do nothing for gradient updates on the pooling layer
        return new LayerUpdateReturns {};
    }
    #endregion
    
}
#endregion
