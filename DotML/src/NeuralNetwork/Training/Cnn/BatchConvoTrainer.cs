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
public class BatchedConvolutionalEnumerableBackpropagationTrainer<TNetwork>
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
    /// Strategy for learning rate adjusting during training. (default: ConstantRate)
    /// </summary>
    public ILearningRateOptimizer LearningRateOptimizer {get; set;} = new ConstantRate();

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
    /// Gets or sets a place to report testing validation results to (default: DefaultValidationReport)
    /// </summary>
    public IValidationReport? ValidationReport {get; set;} = new DefaultValidationReport();

    /// <summary>
    /// Gets or sets a place to report training run-time performance results to (default: null)
    /// </summary>
    public IPerformanceReport? PerformanceReport {get; set;} = null;

    /// <summary>
    /// Regularization function (default: NoRegularization)
    /// </summary>
    public RegularizationFunction Regularization {get; set;} = new NoRegularization();

    /// <summary>
    /// Network initialization strategy (default: NormalXavierInitialization)
    /// </summary>
    public IInitializer NetworkInitializer {get; set;} = new NormalXavierInitialization();

    /// <summary>
    /// Size of batches per epoch (default: 1)
    /// </summary>
    public int BatchSize {get; set;} = 1;

    public IEpochEnumerator<TNetwork> EnumerateTraining(TNetwork network, IEnumerator<TrainingPair> dataset, IEnumerator<TrainingPair> validation) {
        return new BatchedConvolutionalBackpropagationEnumerator<TNetwork>(
            network,
            dataset,
            validation,
            batchSize:              this.BatchSize,

            earlyStop:              this.EarlyStop,
            earlyStopThreshold:     this.EarlyStopAccuracy,
            lossFunction:           this.LossFunction,
            validationReport:       this.ValidationReport,
            performanceReport:      this.PerformanceReport,
            regularization:         this.Regularization,

            networkInitializer:     this.NetworkInitializer,
            optimizer:              this.LearningRateOptimizer,

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
public partial class BatchedConvolutionalBackpropagationEnumerator<TNetwork> 
    : IEpochEnumerator<TNetwork>
    where TNetwork : ConvolutionalFeedforwardNetwork
{
    public int CurrentEpoch {get; private set;}
    private int CurrentUpdateTimestep;
    public int MaxEpochs {get; init;}
    public double LearningRate => layerUpdateActions.LearningRate;
    public int BatchSize {get; init;}

    public IValidationReport? ValidationReport {get; set;}
    public IPerformanceReport? PerformanceReport {get; set;}

    public TNetwork Current {get; private set;}
    object IEnumerator.Current => Current;

    private IEnumerator<TrainingPair> training;
    private IEnumerator<TrainingPair> validation;

    public bool EnableEarlyStop {get; private set;}
    public double EarlyStopThreshold {get; private set;}
    public LossFunction LossFunction {get; private set;}
    public RegularizationFunction Regularization => layerUpdateActions.Regularization;

    public IInitializer NetworkInitializer {get; private set;}

    public ILearningRateOptimizer LearningRateOptimizer => layerUpdateActions.LearningRateOptimizer;

    public BatchedConvolutionalBackpropagationEnumerator(
        TNetwork network,
        IEnumerator<TrainingPair> training,
        IEnumerator<TrainingPair> validation,
        int batchSize,

        bool earlyStop,
        double earlyStopThreshold,
        LossFunction lossFunction,
        IValidationReport? validationReport,
        IPerformanceReport? performanceReport,
        RegularizationFunction regularization,

        IInitializer networkInitializer,
        ILearningRateOptimizer optimizer,

        int epochs,
        double learningRate,

        bool useClipping,
        double clipThresholdWeight,
        double clipThresholdBias
    ) {
        this.Current = network;
        this.training = training;
        this.validation = validation;
        this.BatchSize = Math.Max(1, batchSize);
        this.batch = new List<TrainingPair>(this.BatchSize);

        this.batch_inputs   = new FeatureSet<double>[this.BatchSize][]; // The inputs to each layer
        this.batch_outputs  = new FeatureSet<double>[this.BatchSize][]; // The outputs from each layer
        this.batch_gradients = new Gradients?[this.BatchSize][];

        this.MaxEpochs = Math.Max(0, epochs);

        this.EnableEarlyStop = earlyStop;
        this.EarlyStopThreshold = earlyStopThreshold;
        this.ValidationReport = validationReport;
        this.PerformanceReport = performanceReport;
        this.LossFunction = lossFunction;
        this.NetworkInitializer = networkInitializer;

        this.backpropagationActions = new BackpropagationActions(useClipping, clipThresholdWeight, clipThresholdBias);
        this.layerUpdateActions = new LayerUpdateActions(Math.Abs(learningRate), regularization, optimizer);

        Reset();
    }

    public void Dispose() { }

    private List<TrainingPair> batch;
    private int num_batches;
    FeatureSet<double>[][] batch_inputs;
    FeatureSet<double>[][] batch_outputs;
    Gradients?[][] batch_gradients;

    private int count_training_items() {
        int count = 0;
        training.Reset();
        while(training.MoveNext()) {
            count++;
        }
        return count;
    }

    public void Reset() {
        this.CurrentEpoch = 0;
        this.CurrentUpdateTimestep = 1; // Timesteps start at 1. Avoids divide by 0s common with Adam
        this.num_batches = (count_training_items() + this.BatchSize - 1) / this.BatchSize;
        this.training.Reset();
        this.validation.Reset();

        this.Current.Initialize(this.NetworkInitializer);
        this.LearningRateOptimizer.Initialize(this.Current);

        this.batch.Clear(); this.batch.EnsureCapacity(this.BatchSize);
        this.batch_inputs   = new FeatureSet<double>[this.BatchSize][]; // The inputs to each layer
        this.batch_outputs  = new FeatureSet<double>[this.BatchSize][]; // The outputs from each layer
        this.batch_gradients = new Gradients?[this.BatchSize][];

        for (var b = 0; b < this.BatchSize; b++) {
            this.batch_inputs[b] = new FeatureSet<double>[Current.LayerCount];
            this.batch_outputs[b] = new FeatureSet<double>[Current.LayerCount];
            this.batch_gradients[b] = new Gradients?[Current.LayerCount];
        }
    }

    public bool MoveNext() {
        if (CurrentEpoch >= MaxEpochs) {
            return false;
        }

        OnEpochStart(this.CurrentEpoch, this.MaxEpochs);
        TrainingStep();
        OnEpochEnd(this.CurrentEpoch, this.MaxEpochs);

        // TODO early STOP
        bool stopEarly = false;
        if (EnableEarlyStop) {
            ValidationReport?.Reset();
            OnValidationStart(this.CurrentEpoch, this.MaxEpochs);
            validation.Reset();
            var sum_error = 0d; var count = 0;
            var max_error = double.MinValue;
            var all_less_threshold = true;
            while (validation.MoveNext()) {
                var input = validation.Current.Input;
                var @true = validation.Current.Output; 
                var predicted = Current.PredictSync(input);
                
                var loss = LossFunction(predicted, @true);
                sum_error += loss;
                max_error = Math.Max(max_error, loss);
                var passed = loss < EarlyStopThreshold;
                all_less_threshold &= passed;
                OnValidated(this.CurrentEpoch, this.MaxEpochs, count, loss);
                count++;
                ValidationReport?.Append(input, @true, predicted, passed, loss);
            }
            if (double.IsNaN(sum_error)) {
                throw new ArithmeticException("NaN detected during output evaluation.");
            }
            var avg_error = sum_error / Math.Max(1, count);
            if (max_error <= EarlyStopThreshold) {
                stopEarly = true;
            }
            OnValidationEnd(this.CurrentEpoch, this.MaxEpochs, max_error);
        }

        CurrentEpoch += 1;
        var epochs_finished = CurrentEpoch >= MaxEpochs;
        
        var is_done = epochs_finished || stopEarly;
        return !is_done;
    }

    const string FeedforwardPerformanceKey = "Feed Forward";
    const string BackpropagationPerformanceKey = "Backpropagation";
    const string WeightUpdatePerformanceKey = "Weight Update";

    private void TrainingStep() {
        training.Reset();
        try {

        // Compute initial batch
        batch.Clear(); batch.EnsureCapacity(this.BatchSize);
        while (batch.Count < this.BatchSize && training.MoveNext()) {
            batch.Add(training.Current);
        }

        // Do batch
        int batch_number = 0; int total_batches = num_batches;
        while (batch.Count > 0) { 
            var batch_size = batch.Count;
            OnBatchStart(batch_number, total_batches);

            // Init layers for batch
            for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                Current.GetLayer(layerIndex).Visit(this.batchInitializer);
            }

            var batch_features = new BatchedFeatureSet<double>(
                batch.Select(batchPair => new FeatureSet<double>(
                    batchPair.Input.Shape(Current.InputShape).ToArray()
                )).ToArray()
            );

            // Forward pass (simulated, duplicate of ConvolutionalFeedforwardNetwork.PredictSync with some additional tracking)
            using (var metric = PerformanceReport?.Begin(FeedforwardPerformanceKey)) {
                BatchedFeatureSet<double> layer_input = batch_features;
                for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                    // Store input to this layer (less than ideal to have a loop here)
                    for (var batchIndex = 0; batchIndex < batch_size; batchIndex++) {
                        this.batch_inputs[batchIndex][layerIndex] = layer_input[batchIndex];
                    }

                    // Evaluate the layer
                    var layer = Current.GetLayer(layerIndex);
                    var layer_output = layer.EvaluateSync(layer_input);

                    // Store outputs from evaluation of this layer (less than ideal to have a loop here)
                    for (var batchIndex = 0; batchIndex < batch_size; batchIndex++) {
                        this.batch_outputs[batchIndex][layerIndex] = layer_output[batchIndex];
                    }

                    // Set the next-layer input to the output of this layer
                    layer_input = layer_output;
                }
                var actual = layer_input;
            }

            // Backwards pass
            using (var metric = PerformanceReport?.Begin(BackpropagationPerformanceKey)) {
                Parallel.For(0, batch_size, (batchIndex) => {
                    var currentPair = batch[batchIndex];
                    var expected = currentPair.Output;

                    var inputs = this.batch_inputs[batchIndex];
                    var outputs = this.batch_outputs[batchIndex];
                    var layer_gradients = this.batch_gradients[batchIndex];

                    var @true = expected.Shape(outputs[^1].Select(x => x.Shape).ToArray());
                    var errors = outputs[^1].Zip(@true).Select(x => x.First-x.Second).ToArray();
                    
                    var backprop_args = new BackpropagationArgs();
                    backprop_args.Errors = errors;
                    backprop_args.TrueLabel = expected;
                    for (var layerIndex = Current.LayerCount - 1; layerIndex >= 0; layerIndex--) {
                        backprop_args.Inputs = inputs[layerIndex];
                        backprop_args.Outputs = outputs[layerIndex];

                        var layer = Current.GetLayer(layerIndex);
                        backprop_args.LayerIndex = layerIndex;
                        //using (var backpropLayerMetric = PerformanceReport?.Begin(BackpropagationPerformanceKey + "/" + layerIndex)) {
                            var returns = layer.Visit(this.backpropagationActions, backprop_args); // Backpropagation is different for each layer kind, leverage polymorphism
                            layer_gradients[layerIndex] = returns.Gradient;
                            backprop_args.Errors = returns.Errors;
                        //}
                    }
                });
            }

            // For forward pass for batch
            /*Parallel.For(0, batch.Count, (batchIndex) => {
                var currentPair = batch[batchIndex];
                var input = new FeatureSet<double>(currentPair.Input.Shape(
                    Current.InputShape
                ).ToArray());
                var expected = currentPair.Output; // TODO Output of a FULLY CONNECTED LAYER SHOULDtm BE A COLUMN MATRIX THIS IS NOT GUARANTEED TO BE THE CASE IF THE NETWORK DOESN'T USE THEM

                var inputs = this.batch_inputs[batchIndex];
                var outputs = this.batch_outputs[batchIndex];
                var layer_gradients = this.batch_gradients[batchIndex];

                // Forward pass (simulated, duplicate of ConvolutionalFeedforwardNetwork.PredictSync with some additional tracking)
                using (var metric = PerformanceReport?.Begin(FeedforwardPerformanceKey)) {
                    FeatureSet<double> layer_input = input;
                    for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                        inputs[layerIndex] = layer_input;
                        var layer = Current.GetLayer(layerIndex);
                        var layer_output = layer.EvaluateSync(layer_input);
                        outputs[layerIndex] = layer_output;
                        layer_input = layer_output;
                    }
                    var actual = layer_input;
                }

                // Backwards pass
                using (var metric = PerformanceReport?.Begin(BackpropagationPerformanceKey)) {
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
                        var returns = layer.Visit<BackpropagationArgs, BackpropagationReturns>(this.backpropagationActions, backprop_args); // Backpropagation is different for each layer kind, leverage polymorphism
                        // TODO maybe store additional details like gradients here for us to apply updates in the next step
                        layer_gradients[layerIndex] = returns.Gradient;
                        backprop_args.Errors = returns.Errors;
                    }
                }
            });*/

            // Update weights
            using (var metric = PerformanceReport?.Begin(WeightUpdatePerformanceKey)) {
                var update_args = new LayerUpdateArgs();
                update_args.UpdateTimestep = CurrentUpdateTimestep;
                update_args.ParameterOffset = 0;
                //used_params.Clear();
                for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                    // Average gradients across batch
                    Gradients? avgGradient = batch_gradients[0][layerIndex];
                    if (batch_size > 0) {
                        switch (avgGradient) {
                            case ConvolutionGradients convo:
                                #pragma warning disable CS8602 
                                #pragma warning disable CS8604
                                var filters = convo.FilterKernelGradients?.Length ?? 0;
                                var all_c_batches = batch_gradients.Select(x => x[layerIndex]).OfType<ConvolutionGradients>();
                                var new_filter_kernel_gradients = new Matrix<double>[filters][];
                                for (var filterIndex = 0; filterIndex < filters; filterIndex++) {
                                    var kernels = convo.FilterKernelGradients?[filterIndex]?.Length ?? 0;
                                    var kernel_grads = new Matrix<double>[kernels];

                                    for (var kernelIndex = 0; kernelIndex < kernels; kernelIndex++) {
                                        kernel_grads[kernelIndex] = Matrix<double>.Average(all_c_batches.Select(c => c.FilterKernelGradients[filterIndex][kernelIndex]));
                                    }

                                    new_filter_kernel_gradients[filterIndex] = kernel_grads;
                                }
                                convo.FilterKernelGradients = new_filter_kernel_gradients;
                                convo.BiasGradients = (double[])Vec<double>.Average(all_c_batches.Select(c => Vec<double>.Wrap(c.BiasGradients)));
                                #pragma warning restore CS8602
                                #pragma warning restore CS8604
                                break;
                            case FullyConnectedGradients connect:
                                var all_fc_batches = batch_gradients.Select(x => x[layerIndex]).OfType<FullyConnectedGradients>();
                                connect.WeightGradients = Matrix<double>.Average(all_fc_batches.Select(fcg => fcg.WeightGradients));
                                connect.BiasGradients = Vec<double>.Average(all_fc_batches.Select(fcg => fcg.BiasGradients));
                                break;
                            case DepthwiseConvolutionGradients depth:
                                {
                                    #pragma warning disable CS8602 
                                    var kernels = depth.KernelGradients?.Length ?? 0;
                                    var kernel_grads = new Matrix<double>[kernels];
                                    var all_batches = batch_gradients.Select(x => x[layerIndex]).OfType<DepthwiseConvolutionGradients>();
                                    for (var i = 0; i < kernels; i++) {
                                        kernel_grads[i] = Matrix<double>.Average(all_batches.Select(c => c.KernelGradients[i]));
                                    }
                                    depth.KernelGradients = kernel_grads;
                                    #pragma warning restore CS8602
                                }
                                break;
                            case LayerNormGradients norm:
                                {
                                    #pragma warning disable CS8602 
                                    var all_batches = batch_gradients.Select(x => x[layerIndex]).OfType<LayerNormGradients>();

                                    var gamma_c = norm.GammaGradients?.Length ?? 0;
                                    var beta_c = norm.BetaGradients?.Length ?? 0;

                                    var new_gammas = new Matrix<double>[gamma_c];
                                    for (var i = 0; i < gamma_c; i++) {
                                        new_gammas[i] = Matrix<double>.Average(all_batches.Select(
                                            batch_elem => batch_elem.GammaGradients[i]
                                        ));
                                    }
                                    norm.GammaGradients = new_gammas;
                                    
                                    var new_betas = new Matrix<double>[beta_c];
                                    for (var i = 0; i < gamma_c; i++) {
                                        new_betas[i] = Matrix<double>.Average(all_batches.Select(
                                            batch_elem => batch_elem.BetaGradients[i]
                                        ));
                                    }
                                    norm.BetaGradients = new_betas;
                                    #pragma warning restore CS8602
                                }
                                break;
                        }
                    }

                    // Perform update
                    update_args.Gradients = avgGradient;
                    var layer = Current.GetLayer(layerIndex);
                    update_args.LayerIndex = layerIndex;
                    layer.Visit<LayerUpdateArgs, LayerUpdateReturns>(this.layerUpdateActions, update_args);
                    update_args.ParameterOffset += layer.TrainableParameterCount();
                }
            }
            CurrentUpdateTimestep++;
            OnBatchEnd(batch_number++, total_batches);

            // Cleanup layers for batch
            for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                Current.GetLayer(layerIndex).Visit(this.batchCleanup);
            }

            // Compute next batch
            batch.Clear(); batch.EnsureCapacity(this.BatchSize);
            while (batch.Count < this.BatchSize && training.MoveNext()) {
                batch.Add(training.Current);
            }
        }

        } catch (Exception e) {
            throw new ConvolutionalBackpropagationException(Current, e);
        }
    }

    public event EpochStartHandler OnEpochStart = delegate {};
    public event BatchStartHandler OnBatchStart = delegate {};
    public event BatchEndHandler OnBatchEnd = delegate {};
    public event ValidationStartHandler OnValidationStart = delegate {};
    public event ValidationStepHandler OnValidated = delegate {};
    public event ValidationEndHandler OnValidationEnd = delegate {};
    public event EpochEndHandler OnEpochEnd = delegate {};
}
#endregion
