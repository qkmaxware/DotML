using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

#region Enumerator
public partial class BatchTrainerEnumerator<TNetwork> 
    : IEpochEnumerator<TNetwork>
    where TNetwork : FeedforwardNetwork
{
    public int CurrentEpoch {get; private set;}
    private int CurrentUpdateTimestep;
    public int MaxEpochs {get; init;}
    public double LearningRate {get; init;}
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
    public RegularizationFunction Regularization {get; init;}

    public IInitializer NetworkInitializer {get; private set;}

    public ILearningRateOptimizer LearningRateOptimizer {get; init;}

    public bool UseGradientClipping {get; init;}
    public double GradientClippingThresholdWeight {get; init;}
    public double GradientClippingThresholdBias {get; init;}
    
    #region Initialization

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

        this.inputs = new BatchedFeatureSet<double>[Current.LayerCount];
        this.outputs = new BatchedFeatureSet<double>[Current.LayerCount];
        this.layer_gradients = new LayerGradients?[Current.LayerCount];

        this.MaxEpochs = Math.Max(0, epochs);

        this.EnableEarlyStop = earlyStop;
        this.EarlyStopThreshold = earlyStopThreshold;
        this.ValidationReport = validationReport;
        this.Profiler = performanceReport;
        this.LossFunction = lossFunction;
        this.NetworkInitializer = networkInitializer;

        this.EarlyStopPatience = Math.Max(1, earlyStopPatience);
        this._patience_count = this.EarlyStopPatience;

        this.LearningRate = Math.Abs(learningRate);
        this.Regularization = regularization;
        this.LearningRateOptimizer = optimizer;

        this.UseGradientClipping = useClipping;
        this.GradientClippingThresholdWeight = clipThresholdWeight;
        this.GradientClippingThresholdBias = clipThresholdBias;

        Reset();
    }

    public void Dispose() { }

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

        this._patience_count = this.EarlyStopPatience;

        this.batch.Clear(); this.batch.EnsureCapacity(this.BatchSize);
        this.inputs = new BatchedFeatureSet<double>[Current.LayerCount];
        this.outputs = new BatchedFeatureSet<double>[Current.LayerCount];
        this.layer_gradients = new LayerGradients?[Current.LayerCount];
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

    #endregion

    #region Validation Step

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
                    
                    var loss = LossFunction.Invoke(predicted, @true);
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

    #endregion

    const string FeedforwardPerformanceKey = "Feed Forward";
    const string BackpropagationPerformanceKey = "Backpropagation";
    const string WeightUpdatePerformanceKey = "Weight Update";

    #region Training Step

    private List<TrainingPair> batch;
    private int num_batches;
    private BatchedFeatureSet<double>[] inputs;
    private BatchedFeatureSet<double>[] outputs;
    LayerGradients?[] layer_gradients;

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
                    // Store input to this layer
                    this.inputs[layerIndex] = layer_input;

                    // Evaluate the layer
                    var layer = Current.GetLayer(layerIndex);
                    var layer_output = layer.EvaluateSync(layer_input);

                    // Store outputs from evaluation of this layer
                    this.outputs[layerIndex] = layer_output;

                    // Set the next-layer input to the output of this layer
                    layer_input = layer_output;
                }
                var actual = layer_input;
            }

            // Backwards pass
            using (var metric = Profiler?.Begin(BackpropagationPerformanceKey)) {
                // Setup initial backpropagation arguments
                FeatureSet<double>[] output_errors = new FeatureSet<double>[batch_size];
                for (var b = 0; b < batch_size; b++) {
                    var currentPair = batch[b];
                    var expected_vec = currentPair.Output;                                                          // The outputs as recorded in the training pair
                    var predicted = outputs[^1][b];                                                                 // The outputs of the last layer for batch 'b'
                    
                    var predicted_vec = Vec<double>.Wrap(predicted.SelectMany(mtx => mtx.FlattenRows()).ToArray()); // Convert the predicted outputs to a vector
                    var error_vec = this.LossFunction.Gradient(@predicted: predicted_vec, @true: expected_vec);     // Compute the gradient values for the loss function
                    var errors = error_vec.Shape(predicted.Shape).ToArray();                                        // Make the error vector match the output shape for backpropagation    
                    
                    //var @true = expected.Shape(predicted.Shape);                                // Make the expected vector match the output shape
                    //var errors = predicted.Zip(@true).Select(x => x.First-x.Second).ToArray();  // predicted - expected
                    output_errors[b] = new FeatureSet<double>(errors);
                }
                var backprop_args = new BackpropagationArgs(
                    Current.LayerCount - 1, 
                    new BatchedFeatureSet<double>(),                // Gets replaced later
                    new BatchedFeatureSet<double>(),                // Gets replaced later
                    new BatchedFeatureSet<double>(output_errors)    // OutputErrors at output layer = predicted - expected as computed above
                );

                // Do backwards pass through the layers
                for (var layerIndex = Current.LayerCount - 1; layerIndex >= 0; layerIndex--) {
                    backprop_args.LayerIndex = layerIndex;
                    backprop_args.InputBatch = inputs[layerIndex];
                    backprop_args.OutputBatch = outputs[layerIndex];
                    
                    var layer = Current.GetLayer(layerIndex);
                    var returns = layer.Backpropagate(backprop_args);
                    if (UseGradientClipping) {
                        // Do gradient clipping on update gradients
                        returns.Gradients?.Clip(GradientClippingThresholdWeight, GradientClippingThresholdBias);
                        // Do gradient clipping on input gradients (so that the clipped gradients get backpropagated)
                        ClipBatch(returns.InputErrors, GradientClippingThresholdWeight);
                    }

                    layer_gradients[layerIndex] = returns.Gradients;
                    backprop_args.OutputErrors = returns.InputErrors;
                }
            }
            
            // Update weights
            using (var metric = Profiler?.Begin(WeightUpdatePerformanceKey)) {
                var update_args = new LayerUpdateArgs();
                update_args.UpdateTimestep = CurrentUpdateTimestep;
                update_args.ParameterOffset = 0;
                BeginParameterTracking();
                for (var layerIndex = 0; layerIndex < Current.LayerCount; layerIndex++) {
                    // Modify the gradient based on optimizer to get actual gradient to apply
                    LayerGradients? avgGradient = layer_gradients[layerIndex];
                    avgGradient?.Apply((index, parameter, grad) =>  gradient_update(update_args.UpdateTimestep, LearningRate, parameter, grad, update_args.ParameterOffset + index));

                    // Perform update
                    update_args.Gradients = avgGradient;
                    var layer = Current.GetLayer(layerIndex);
                    update_args.LayerIndex = layerIndex;
                    layer.SubtractGradients(avgGradient);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected double ClipValue(double d, double threshold) {
        if (double.IsNaN(d))
            d = 1e-8;
        return Math.Abs(d) > threshold ? Math.Sign(d) * threshold : d;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipVector(Vec<double> vec, double threshold) {
        vec.Apply((value) => ClipValue(value, threshold));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipMatrix(Matrix<double> mat, double threshold) {
        mat.Apply((value) => ClipValue(value, threshold));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipFeatures(FeatureSet<double> features, double threshold) {
        foreach (var matrix in features)
            ClipMatrix(matrix, threshold);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipBatch(BatchedFeatureSet<double> batch, double threshold) {
        foreach (var features in batch)
            ClipFeatures(features, threshold);
    }

    private bool IsTrackingUsedParameters = false;
    private void BeginParameterTracking() {
        #if DEBUG
        IsTrackingUsedParameters = true;
        used_params.Clear();
        #endif
    }
    private HashSet<int> used_params = new HashSet<int>();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double gradient_update(int updateTimestep, double learningRate, double prevWeight, double gradient, int parameterIndex) {
        // Verify the parameter has not been used this iteration already
        if (IsTrackingUsedParameters) {
            lock(used_params) {
                if (used_params.Contains(parameterIndex))
                    throw new Exception("Parameter " + parameterIndex + " has already been used this iteration");
                used_params.Add(parameterIndex);
            }
        }

        var regularized_grad = gradient + Regularization.Invoke(prevWeight);
        var optimized_grad = LearningRateOptimizer.GetParameterUpdate(updateTimestep, learningRate, regularized_grad, parameterIndex);
        return optimized_grad;
    }

    #endregion

    #region Event Handlers

    public event EpochStartHandler OnEpochStart = delegate {};
    public event BatchStartHandler OnBatchStart = delegate {};
    public event BatchEndHandler OnBatchEnd = delegate {};
    public event ValidationStartHandler OnValidationStart = delegate {};
    public event ValidationStepHandler OnValidated = delegate {};
    public event ValidationEndHandler OnValidationEnd = delegate {};
    public event EpochEndHandler OnEpochEnd = delegate {};

    #endregion

    #region Utility Classes

    public struct LayerUpdateArgs {
        public int ParameterOffset;
        public int UpdateTimestep;
        public int LayerIndex;
        public LayerGradients? Gradients;
    }

    #endregion
}
#endregion
