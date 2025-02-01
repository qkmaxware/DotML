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
public class EnumerableBatchTrainer<TNetwork>
    : IEnumerableTrainer<TNetwork>
where TNetwork : FeedforwardNetwork
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

    private int _earlyStopPatience = 1;
    /// <summary>
    /// The number of epochs in a row where the early stop condition has been met before early stop is triggered (default: 1)
    /// </summary>
    public int EarlyStopPatience {
        get => _earlyStopPatience;
        set => _earlyStopPatience = Math.Max(1, value); // Always have at least 1 
    }

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
    public IProfilingReport? Profiler {get; set;} = null;

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
        return new BatchTrainerEnumerator<TNetwork>(
            network,
            dataset,
            validation,
            batchSize:              this.BatchSize,

            earlyStop:              this.EarlyStop,
            earlyStopThreshold:     this.EarlyStopAccuracy,
            earlyStopPatience:      this.EarlyStopPatience,
            lossFunction:           this.LossFunction,
            validationReport:       this.ValidationReport,
            performanceReport:      this.Profiler,
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
public partial class BatchTrainerEnumerator<TNetwork> 
    : IEpochEnumerator<TNetwork>
    where TNetwork : FeedforwardNetwork
{
    public int CurrentEpoch {get; private set;}
    private int CurrentUpdateTimestep;
    public int MaxEpochs {get; init;}
    public double LearningRate => layerUpdateActions.LearningRate;
    public int BatchSize {get; init;}

    public IValidationReport? ValidationReport {get; set;}
    public IProfilingReport? Profiler {get; set;}

    public TNetwork Current {get; private set;}
    object IEnumerator.Current => Current;

    private IEnumerator<TrainingPair> training;
    private IEnumerator<TrainingPair> validation;

    public bool EnableEarlyStop {get; private set;}
    public double EarlyStopThreshold {get; private set;}
    public int EarlyStopPatience {get; private set;}
    private int _patience_count = 0;
    public LossFunction LossFunction {get; private set;}
    public RegularizationFunction Regularization => layerUpdateActions.Regularization;

    public IInitializer NetworkInitializer {get; private set;}

    public ILearningRateOptimizer LearningRateOptimizer => layerUpdateActions.LearningRateOptimizer;

    public BatchTrainerEnumerator(
        TNetwork network,
        IEnumerator<TrainingPair> training,
        IEnumerator<TrainingPair> validation,
        int batchSize,

        bool earlyStop,
        double earlyStopThreshold,
        int earlyStopPatience,
        LossFunction lossFunction,
        IValidationReport? validationReport,
        IProfilingReport? performanceReport,
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
        this.layer_gradients = new Gradients?[Current.LayerCount];

        this.MaxEpochs = Math.Max(0, epochs);

        this.EnableEarlyStop = earlyStop;
        this.EarlyStopThreshold = earlyStopThreshold;
        this.ValidationReport = validationReport;
        this.Profiler = performanceReport;
        this.LossFunction = lossFunction;
        this.NetworkInitializer = networkInitializer;

        this.EarlyStopPatience = Math.Max(1, earlyStopPatience);
        this._patience_count = this.EarlyStopPatience;

        this.backpropagationActions = new BackpropagationActions(useClipping, clipThresholdWeight, clipThresholdBias);
        this.layerUpdateActions = new LayerUpdateActions(Math.Abs(learningRate), regularization, optimizer);

        Reset();
    }

    public void Dispose() { }

    private List<TrainingPair> batch;
    private int num_batches;
    FeatureSet<double>[][] batch_inputs;
    FeatureSet<double>[][] batch_outputs;
    Gradients?[] layer_gradients;

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
        this.layer_gradients = new Gradients?[Current.LayerCount];

        this._patience_count = this.EarlyStopPatience;

        for (var b = 0; b < this.BatchSize; b++) {
            this.batch_inputs[b] = new FeatureSet<double>[Current.LayerCount];
            this.batch_outputs[b] = new FeatureSet<double>[Current.LayerCount];
        }
    }

    public bool MoveNext() {
        if (CurrentEpoch >= MaxEpochs) {
            return false;
        }

        OnEpochStart(this.CurrentEpoch, this.MaxEpochs);
        TrainingStep();
        OnEpochEnd(this.CurrentEpoch, this.MaxEpochs);

        bool stopEarly = ValidateStep();

        CurrentEpoch += 1;
        var epochs_finished = CurrentEpoch >= MaxEpochs;
        
        var is_done = epochs_finished || stopEarly;
        return !is_done;
    }

    private bool ValidateStep() {
        bool stopEarly = false;
        if (EnableEarlyStop) {
            ValidationReport?.Reset();
            validation.Reset();
            OnValidationStart(this.CurrentEpoch, this.MaxEpochs);
            var sum_error = 0d; var count = 0;
            var max_error = double.MinValue;
            var all_less_threshold = true;

            var concurrency_level = this.BatchSize; // or Environment.ProcessorCount
            List<(FeatureSet<double> In, Vec<double> Out)> batch = new List<(FeatureSet<double>, Vec<double>)>(concurrency_level);
            while (batch.Count < concurrency_level && validation.MoveNext()) {
                var pair = validation.Current;
                var input = new FeatureSet<double>(pair.Input.Shape(Current.InputShape).ToArray());
                var output = pair.Output;
                batch.Add((input, output));
            }
            var batch_input = new BatchedFeatureSet<double>(batch.Select(x => x.In).ToArray());

            while (batch.Count > 0) {
                // Perform Feed-Forward
                var batch_predicted = Current.PredictSync(batch_input);

                // Measure loss across batch
                for (var batchIndex = 0; batchIndex < batch.Count; batchIndex++) {
                    var input = Vec<double>.Wrap(batch_input[batchIndex].SelectMany(mtx => mtx.FlattenRows()).ToArray());
                    var @true = batch[batchIndex].Out;
                    var predicted =  Vec<double>.Wrap(batch_predicted[batchIndex].SelectMany(mtx => mtx.FlattenRows()).ToArray());
                    
                    var loss = LossFunction(predicted, @true);
                    sum_error += loss;
                    max_error = Math.Max(max_error, loss);
                    var passed = loss < EarlyStopThreshold;
                    all_less_threshold &= passed;
                    OnValidated(this.CurrentEpoch, this.MaxEpochs, count, loss);
                    count++;
                    ValidationReport?.Append(input, @true, predicted, passed, loss);
                }

                // Compute next batch
                batch.Clear();
                while (batch.Count < concurrency_level && validation.MoveNext()) {
                    var pair = validation.Current;
                    var input = new FeatureSet<double>(pair.Input.Shape(Current.InputShape).ToArray());
                    var output = pair.Output;
                    batch.Add((input, output));
                }
                batch_input = new BatchedFeatureSet<double>(batch.Select(x => x.In).ToArray());
            }

            if (double.IsNaN(sum_error)) {
                throw new ArithmeticException("NaN detected during validation.");
            }
            var avg_error = sum_error / Math.Max(1, count);
            if (max_error <= EarlyStopThreshold) {
                // Decrement the patience count
                _patience_count --;
                if (_patience_count <= 0)
                    stopEarly = true;
            } else {
                // Reset the early stop patience
                _patience_count = this.EarlyStopPatience;
            }
            OnValidationEnd(this.CurrentEpoch, this.MaxEpochs, max_error);
        }
        return stopEarly;
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
                Current.GetLayer(layerIndex).BeginTraining();
            }

            var batch_features = new BatchedFeatureSet<double>(
                batch.Select(batchPair => new FeatureSet<double>(
                    batchPair.Input.Shape(Current.InputShape).ToArray()
                )).ToArray()
            );

            // Forward pass (simulated, duplicate of FeedforwardNetwork.PredictSync with some additional tracking)
            using (var metric = Profiler?.Begin(FeedforwardPerformanceKey)) {
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
            using (var metric = Profiler?.Begin(BackpropagationPerformanceKey)) {
                // Roll up inputs/outputs into batched feature sets again in case they are needed by a backpropagation method
                var input_batches = new BatchedFeatureSet<double>[Current.LayerCount];
                var output_batches = new BatchedFeatureSet<double>[Current.LayerCount];
                for (var layerIndex = 0; layerIndex < input_batches.Length; layerIndex++) {
                    var infeatures = new FeatureSet<double>[batch_size];
                    var outfeatures = new FeatureSet<double>[batch_size];
                    for (var batchIndex = 0; batchIndex < batch_size; batchIndex++) {
                        infeatures[batchIndex] = this.batch_inputs[batchIndex][layerIndex];
                        outfeatures[batchIndex] = this.batch_outputs[batchIndex][layerIndex];
                    }
                    input_batches[layerIndex] = new BatchedFeatureSet<double>(infeatures);
                    output_batches[layerIndex] = new BatchedFeatureSet<double>(outfeatures);
                }
                // Setup initial backpropagation arguments
                var backprop_args = new BackpropagationArgs();
                backprop_args.BatchTrueLabels = new Vec<double>[batch_size];
                FeatureSet<double>[] output_errors = new FeatureSet<double>[batch_size];
                for (var b = 0; b < batch_size; b++) {
                    var currentPair = batch[b];
                    var expected = currentPair.Output;
                    backprop_args.BatchTrueLabels[b] = expected;

                    var predicted = output_batches[^1][b];                                      // The outputs of the last layer for batch 'b'
                    var @true = expected.Shape(predicted.Shape);                                // Make the expected vector match the output shape
                    var errors = predicted.Zip(@true).Select(x => x.First-x.Second).ToArray();  // predicted - expected

                    output_errors[b] = new FeatureSet<double>(errors);
                }
                backprop_args.OutputErrors = new BatchedFeatureSet<double>(output_errors);

                // Do backwards pass through the layers
                for (var layerIndex = Current.LayerCount - 1; layerIndex >= 0; layerIndex--) {
                    backprop_args.InputBatch = input_batches[layerIndex]; // Will be needed for backpropagation of BatchNorm (as we need to see ALL batched inputs to compute mean/variance)
                    backprop_args.OutputBatch = output_batches[layerIndex];
                    
                    var layer = Current.GetLayer(layerIndex);
                    backprop_args.LayerIndex = layerIndex;
                    var returns = layer.Visit(this.backpropagationActions, backprop_args); // Backpropagation is different for each layer kind, leverage polymorphism
                    
                    layer_gradients[layerIndex] = returns.Gradient;
                    backprop_args.OutputErrors = returns.InputErrors;
                }
            }
            
            // Update weights
            using (var metric = Profiler?.Begin(WeightUpdatePerformanceKey)) {
                var update_args = new LayerUpdateArgs();
                update_args.UpdateTimestep = CurrentUpdateTimestep;
                update_args.ParameterOffset = 0;
                this.layerUpdateActions.TrackUsedParameters(false); // TODO only do this in DEBUG mode
                for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                    // Average gradients across batch
                    Gradients? avgGradient = layer_gradients[layerIndex];

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
                Current.GetLayer(layerIndex).EndTraining();
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
