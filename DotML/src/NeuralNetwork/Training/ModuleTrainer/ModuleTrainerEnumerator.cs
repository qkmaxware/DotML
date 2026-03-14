using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public class ModuleTrainingEnumerator: IEnumerator<ModuleTrainingEnumerator.Report> {

    /// <summary>
    /// Training report
    /// </summary>
    public class Report : IValidationReport
    {
        #region Basic Metrics
        /// <summary>
        /// Current epoch this report is about
        /// </summary>
        public int Epoch;
        /// <summary>
        /// Number of samples tested
        /// </summary>
        public int SampleCount { get; set; }
        /// <summary>
        /// Computed loss
        /// </summary>
        public Metric<float> Loss { get; set; } = new Metric<float>();
        #endregion
    
        private Dictionary<Type, IMetricsProvider> providers;

        public Report()
        {
            this.providers = new Dictionary<Type, IMetricsProvider>();
        }
        public Report(IEnumerable<IMetricsProvider> providers)
        {
            this.providers = providers.ToDictionary((v) => v.GetType());
        }

        /// <summary>
        /// Get a particular metric provider by type
        /// </summary>
        /// <typeparam name="TMetric">metric provider type</typeparam>
        /// <returns>metric provider or throws</returns>
        public TMetric Metrics<TMetric>() where TMetric:IMetricsProvider
        {
            return (TMetric)providers[typeof(TMetric)];
        }

        /// <summary>
        /// Get a particular metric provider by type
        /// </summary>
        /// <typeparam name="TMetric">metric provider type</typeparam>
        /// <param name="metric">metric or null if it doesn't exist</param>
        /// <returns>true if metric provider exists</returns>
        public bool TryGetMetrics<TMetric>([NotNullWhen(true)] out TMetric? metric) where TMetric:IMetricsProvider
        {
            if (providers.TryGetValue(typeof(TMetric), out var m)){
                metric = (TMetric)m;
                return true;
            } else {
                metric = default;
                return false;
            }
        }

        /// <summary>
        /// Get a particular metric provider by type
        /// </summary>
        /// <typeparam name="TMetric">metric provider type</typeparam>
        /// <returns>metric provider or null if it doesn't exist</returns>
        public TMetric? MetricsOrNull<TMetric>() where TMetric:IMetricsProvider
        {
            return providers.TryGetValue(typeof(TMetric), out var p)
                ? (TMetric)p
                : default;
        }

        /// <summary>
        /// Enumerate over all metric providers
        /// </summary>
        /// <returns>enumerable of metric providers</returns>
        public IEnumerable<IMetricsProvider> AllMetrics() => this.providers.Values;

        /// <summary>
        /// Reset the statistics of the training report
        /// </summary>
        public void Reset()
        {
            Epoch = 0;
            SampleCount = 0;
            this.Loss.Reset();

            foreach (var provider in providers.Values)
                provider.Reset();
        }

        /// <summary>
        /// Add a sample to the training report
        /// </summary>
        /// <param name="loss">loss value</param>
        /// <param name="logits">logits from prediction</param>
        /// <param name="truth">output of ground truth or one-hot label</param>
        public void AddSample(float loss, ReadOnlySpan<float> input, ReadOnlySpan<float> logits, ReadOnlySpan<float> truth)
        {
            // Update loss stats
            Loss.AddSample(loss);

            foreach (var provider in providers.Values)
                provider.AddSample(loss, input, logits, truth);

            // Increase sample count
            SampleCount++;
        }
    }

    /// <summary>
    /// Progress along a single epoch
    /// </summary>
    public struct EpochProgress
    {
        /// <summary>
        /// Current epoch
        /// </summary>
        public int Epoch;
        /// <summary>
        /// Index into the training iterations
        /// </summary>
        public int TrainingIndex;           // Goes from 0 to TrainingSampleCount - 1 (it's an index after all)
        /// <summary>
        /// Total number of training samples
        /// </summary>
        public int TrainingSampleCount;     
        /// <summary>
        /// Index into the validation iterations (after training iterations are complete)
        /// </summary>
        public int ValidationIndex;         // Goes from 0 to ValidationSampleCount - 1 (it's an index after all)
        /// <summary>
        /// Total number of validation samples
        /// </summary>
        public int ValidationSampleCount;
        /// <summary>
        /// Current step in the entire process
        /// </summary>
        public int ProcessIndex => IsTraining
            ? TrainingIndex
            : TrainingSampleCount + ValidationIndex;
        /// <summary>
        /// Total number of steps in the entire process
        /// </summary>
        public int ProcessSteps => TrainingSampleCount + ValidationSampleCount;
        /// <summary>
        /// Flag to indicate if the process is currently in the training phase
        /// </summary>
        public bool IsTraining => ValidationIndex == 0 && TrainingIndex < TrainingSampleCount;
        /// <summary>
        /// Flag to indicate if the process is currently in the validation phase
        /// </summary>
        public bool IsValidating => ValidationIndex > 0 && ValidationIndex < ValidationSampleCount;
        /// <summary>
        /// Percent of the epoch that has been completed
        /// </summary>
        public float CompletedPercent => (float)(ProcessIndex + 1) / ProcessSteps;
    }

    public Report Current { get; private set; }
    object IEnumerator.Current => Current;

    public INetworkModule Network { get; init; }
    public ITrainingDataSampler<float> TrainingData {get; init;}
    public ITrainingDataSampler<float> TestingData {get; init;}

    public RegularizationFunction Regularization {get; init;}
    public IOptimizer Optimizer {get; init;}
    public ILocalClippingStrategy<float>? LocalClipping { get; init; }
    public IGlobalClippingStrategy<float>? GlobalClipping {get; init;}
    public IInitializer Initializer {get; init;}
    public LossFunction Loss {get; init;}
    public Predicate<Report>? StopCondition {get; init;}
    public int Epoch {get; private set;}
    public int MaxEpochs {get; init;}
    public ILearningRateScheduler LearningRateScheduler { get; init; }

    public int BatchSize = 8;

    public int Patience {get; init;}

    private int _patienceCounter;
    private bool target_reached;

    public ModuleTrainingEnumerator(
        INetworkModule module,
        ITrainingDataSampler<float> training,
        ITrainingDataSampler<float> testing,
        RegularizationFunction regularization,
        IOptimizer optimizer,
        ILocalClippingStrategy<float>? localClipping,
        IGlobalClippingStrategy<float>? globalClipping,
        IInitializer initializer,
        LossFunction loss,
        Predicate<Report>? stopCondition,
        int maxEpochs,
        ILearningRateScheduler scheduler,
        int batchSize,
        int patience,
        List<IMetricsProvider>? metricsProviders = null
    )
    {
        this.Network = module;
        this.TrainingData = training;
        this.TestingData = testing;
        this.Regularization = regularization;
        this.Optimizer = optimizer;
        this.LocalClipping = localClipping;
        this.GlobalClipping = globalClipping;
        this.Initializer = initializer;
        this.Loss = loss;
        this.StopCondition = stopCondition;
        this.MaxEpochs = maxEpochs;
        this.LearningRateScheduler = scheduler;
        this.BatchSize = batchSize;
        this.Patience = patience;

        this.Current = metricsProviders is not null ? new Report(metricsProviders) : new Report();

        Reset();
    }

    public void Dispose() { }

    public void Reset()
    {
        this.Epoch = 0;
        this.Network.Initialize(this.Initializer);
        this.Optimizer.ClearCaches();
        this.Current.Reset();

        this._patienceCounter = Patience;
        this.target_reached = false;
    }

    public bool MoveNext(IProgress<EpochProgress>? progress)
    {
        if (Epoch >= MaxEpochs || target_reached)
        {
            return false;
        }

        // Report batch started
        int trainingIndex = 0;
        progress?.Report(new EpochProgress
        {
            Epoch = this.Epoch,
            TrainingIndex = trainingIndex,
            TrainingSampleCount = TrainingData.Count,
            ValidationIndex = 0,
            ValidationSampleCount = TestingData.Count
        });

        // Training loop 
        foreach ((Tensor<float> batch, Tensor<float> truth) in TrainingData.Sample(batchSize: BatchSize))
        {
            // Forward step
            var context = new EvaluationContext(EvaluationMode.Training);
            var outputs = Network.Forward(batch, context);

            // Compute loss/error/dy
            var batches = outputs.Shape.Length(0);

            var dY = Tensor<float>.Defaults(outputs.Shape);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                Loss.Gradient(
                    gradient: dY.SubtensorSpan(batchIndex),         // Subspan of dY to store the results of the gradient computation in
                    predicted: outputs.SubtensorSpan(batchIndex),   // Treat subspan of output as a vector across non-batch dimensions
                    @true: truth.SubtensorSpan(batchIndex)          // Treat subspan of truth as a vector across non-batch dimensions
                );
            }
            if (this.LocalClipping is not null)
                this.LocalClipping.ClipInput(dY);                                       // Clip these gradients too in case they are too large

            // Backward step (with local clipping if provided)
            var gradients = Network.Backward(dY, ctx: context, clipping: this.LocalClipping);

            // Global clipping (if provided)
            if (this.GlobalClipping is not null)
            {
                gradients.Clip(this.GlobalClipping);
            }

            // Update step
            var lr = this.LearningRateScheduler.RateForEpoch(Epoch);
            Network.Update(
                lr,
                gradients,
                optimizer: this.Optimizer,
                regularization: this.Regularization
            );

            // Report batch completed
            trainingIndex += batch.Shape.Length(0); // Batch size
            progress?.Report(new EpochProgress
            {
                Epoch = this.Epoch,
                TrainingIndex = trainingIndex,
                TrainingSampleCount = TrainingData.Count,
                ValidationIndex = 0,
                ValidationSampleCount = TestingData.Count
            });
        }

        // Validate model accuracy
        validate(progress, batchSize: 1);
        this.LearningRateScheduler?.Step(Epoch, Current.Loss);

        // Move onto next epoch
        this.Epoch++;
        return true;
    }

    public bool MoveNext()
    {
        return MoveNext(null);
    }
    
    public static Report Test(ITrainingDataSampler<float> data, INetworkModule network, LossFunction lossFunction, int batchSize, params IEnumerable<IMetricsProvider> metrics)
    {
        var report = new Report(metrics);
        report.Reset();
        report.Epoch = -1;

        // Testing loop
        foreach ((Tensor<float> batch, Tensor<float> truth) in data.Sample(batchSize: batchSize))
        {
            // Feedforward step
            var result = network.Forward(batch);

            // Compute loss (should this be broken up by batch size?)
            var batches = result.Shape.Length(0);

            for (var batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                var inputSpan = batch.SubtensorSpan(batchIndex);
                var predictedSpan = result.SubtensorSpan(batchIndex);
                var truthSpan = truth.SubtensorSpan(batchIndex);
                var loss = lossFunction.Invoke(
                    predicted: predictedSpan,    // Treat subspan of output as a vector across non-batch dimensions
                    @true: truthSpan          // Treat subspan of truth as a vector across non-batch dimensions
                );

                // Record statistics to the report
                report.AddSample(loss, inputSpan, predictedSpan, truthSpan);
            }

        }

        return report;
    }

    public Report Test(ITrainingDataSampler<float> TestingData)
    {
        return Test(TestingData, Network, Loss, BatchSize);
    }

    private void validate(IProgress<EpochProgress>? progress, int batchSize)
    {
        // Reset the training iteration report (or create a new one, but reusing is fine)
        Current.Reset();
        Current.Epoch = this.Epoch;

        // Testing loop
        int validationIndex = 0;
        foreach ((Tensor<float> batch, Tensor<float> truth) in TestingData.Sample(batchSize: batchSize))
        {
            // Feedforward step
            var result = Network.Forward(batch);

            // Compute loss (should this be broken up by batch size?)
            var batches = result.Shape.Length(0);
            var batchStride = result.Shape.Stride(0);

            for (var batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                var inputSpan = batch.SubtensorSpan(batchIndex);
                var predictedSpan = result.SubtensorSpan(batchIndex);
                var truthSpan = truth.SubtensorSpan(batchIndex);
                var loss = Loss.Invoke(
                    predicted: predictedSpan,    // Treat subspan of output as a vector across non-batch dimensions
                    @true: truthSpan          // Treat subspan of truth as a vector across non-batch dimensions
                );

                // Record statistics to the report
                Current.AddSample(loss, inputSpan, predictedSpan, truthSpan);
            }

            // Report batch completed progress
            validationIndex += batch.Shape.Length(0);
            progress?.Report(new EpochProgress
            {
                Epoch = this.Epoch,
                TrainingIndex = TrainingData.Count - 1,
                TrainingSampleCount = TrainingData.Count,
                ValidationIndex = validationIndex,
                ValidationSampleCount = TestingData.Count
            });
        }

        // Check stop condition based on the report
        if (StopCondition is not null && StopCondition.Invoke(this.Current))
        {
            // Decrement patience and stop if patience reached
            _patienceCounter--;
            if (_patienceCounter <= 0)
                this.target_reached = true;
        }
        else
        {
            // Reset patience count
            _patienceCounter = this.Patience;
        }
    }

    private static int ArgMaxSoftmax(ReadOnlySpan<float> logits) {
        // Optional: Apply softmax to get probabilities first
        float maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++) {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }

        float sumExp = 0f;
        float[] probs = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++) {
            probs[i] = MathF.Exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }

        for (int i = 0; i < logits.Length; i++) {
            probs[i] /= sumExp;
        }

        // Now get argmax of probs
        return ArgMax(probs);
    }

    private static int ArgMax(ReadOnlySpan<float> vector)
    {
        int maxIndex = 0;
        float maxVal = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (vector[i] > maxVal)
            {
                maxVal = vector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}