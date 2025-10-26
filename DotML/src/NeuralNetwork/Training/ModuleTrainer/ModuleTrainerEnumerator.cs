using System.Collections;
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
        /// <summary>
        /// Current epoch this report is about
        /// </summary>
        public int Epoch;
        /// <summary>
        /// Number of samples tested
        /// </summary>
        public int SampleCount { get; set; }
        /// <summary>
        /// Maximum value of the loss function
        /// </summary>
        public float MaxLoss { get; set; }
        /// <summary>
        /// Minimum value of the loss function
        /// </summary>
        public float MinLoss { get; set; }
        /// <summary>
        /// Sum of all loss function values across all samples tested
        /// </summary>
        public float SumLoss;
        /// <summary>
        /// Average value of the loss function across all samples tested
        /// </summary>
        public float AvgLoss => SampleCount > 0 ? SumLoss / SampleCount : 0f;
        /// <summary>
        /// Number of samples with the correct labels
        /// </summary>
        public int TestsPassedCount { get; set; }
        /// <summary>
        /// Number of samples with the incorrect labels
        /// </summary>
        public int TestsFailedCount => SampleCount - TestsPassedCount;
        /// <summary>
        /// Number of class labels
        /// </summary>
        public int NumberOfClasses;
        /// <summary>
        /// Accuracy of the training iteration based on the number of correct samples
        /// </summary>
        public float Accuracy => SampleCount > 0 ? (float)TestsPassedCount / SampleCount : 0f;
        /// <summary>
        /// Precision of the training iteration based on the number of true positives and false positives
        /// </summary>
        public float Precision
        {
            get
            {
                int classes = confusionMatrix.GetLength(0);
                float totalPrecision = 0f;
                int validClasses = 0;

                for (int k = 0; k < classes; k++)
                {
                    int tp = confusionMatrix[k, k];
                    int fp = 0;

                    for (int i = 0; i < classes; i++)
                    {
                        if (i == k)
                            continue;

                        fp += confusionMatrix[i, k];
                    }

                    int denominator = tp + fp;
                    if (denominator > 0)
                    {
                        totalPrecision += (float)tp / denominator;
                        validClasses++;
                    }
                }

                return validClasses > 0 ? totalPrecision / validClasses : 0f;
            }
        }
        /// <summary>
        /// Recall of the training iteration based on the number of true positives and false negatives 
        /// </summary>
        public float Recall
        {
            get
            {
                int classes = confusionMatrix.GetLength(0);
                float totalRecall = 0f;
                int validClasses = 0;

                for (int k = 0; k < classes; k++)
                {
                    int tp = confusionMatrix[k, k];
                    int fn = 0;

                    for (int j = 0; j < classes; j++)
                    {
                        if (j == k)
                            continue;

                        fn += confusionMatrix[k, j];
                    }

                    int denominator = tp + fn;
                    if (denominator > 0)
                    {
                        totalRecall += (float)tp / denominator;
                        validClasses++;
                    }
                }

                return validClasses > 0 ? totalRecall / validClasses : 0f;
            }
        }
        /// <summary>
        /// F1 score of the training iteration based on precision and recall
        /// </summary>
        public float F1
        {
            get
            {
                var precision = this.Precision;
                var recall = this.Recall;
                if (precision + recall <= 0)
                    return 0.0f;

                return 2f * precision * recall / (precision + recall);
            }
        }

        private int[,] confusionMatrix = new int[0, 0];

        /// <summary>
        /// Reset the statistics of the training report
        /// </summary>
        public void Reset()
        {
            SampleCount = 0;
            MaxLoss = 0;
            MinLoss = 0;
            SumLoss = 0;

            TestsPassedCount = 0;
            NumberOfClasses = 0;

            for (var i = 0; i < confusionMatrix.GetLength(0); i++)
                for (var j = 0; j < confusionMatrix.GetLength(0); j++)
                    confusionMatrix[i, j] = 0;
        }

        /// <summary>
        /// Add a sample to the training report
        /// </summary>
        /// <param name="loss">loss value</param>
        /// <param name="logits">logits from prediction</param>
        /// <param name="truth">output of ground truth or one-hot label</param>
        public void AddSample(float loss, ReadOnlySpan<float> logits, ReadOnlySpan<float> truth)
        {
            // Update loss stats
            UpdateLosses(loss, logits, truth);
            TestCorrectness(loss, logits, truth);

            // Increase sample count
            SampleCount++;
        }

        protected void UpdateLosses(float loss, ReadOnlySpan<float> logits, ReadOnlySpan<float> truth)
        {
            SumLoss += loss;
            if (SampleCount == 0)
            {
                // First sample to be added
                MaxLoss = loss;
                MinLoss = loss;
            }
            else
            {
                // Subsequent samples added
                MinLoss = Math.Min(MinLoss, loss);
                MaxLoss = Math.Max(MaxLoss, loss);
            }
        }

        protected void TestCorrectness(float loss, ReadOnlySpan<float> predictedSpan, ReadOnlySpan<float> truthSpan)
        {
            int predictedClass = ArgMaxSoftmax(predictedSpan);
            int trueClass = ArgMax(truthSpan);
            if (predictedClass == trueClass)
                TestsPassedCount++;

            var numClasses = truthSpan.Length; // Assumes one-hot encoding and no multi-classes
            NumberOfClasses = numClasses;

            // Populate the confusion matrix (make a new matrix if required)
            if (confusionMatrix is null || confusionMatrix.GetLength(0) < numClasses)
                confusionMatrix = new int[numClasses, numClasses];

            confusionMatrix[trueClass, predictedClass]++;
        }

        public int[,] GetConfusionMatrix()
        {
            return this.confusionMatrix;
        }
    }

    /// <summary>
    /// Progress along a single epoch
    /// </summary>
    public struct EpochProgress
    {
        public int Epoch;
        public int TrainingIndex;
        public int TrainingSampleCount;

        public int ValidationIndex;
        public int ValidationSampleCount;

        public float Completed => (TrainingIndex + ValidationIndex) / (TrainingSampleCount + ValidationSampleCount - 2);
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
    public float LearningRate { get; init; }
    public ILearningRateScheduler? LearningRateScheduler { get; init; }

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
        float learningRate,
        ILearningRateScheduler? scheduler,
        int batchSize,
        int patience
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
        this.LearningRate = learningRate;
        this.LearningRateScheduler = scheduler;
        this.BatchSize = batchSize;
        this.Patience = patience;

        this.Current = new Report();

        Reset();
    }

    public void Dispose() { }

    public void Reset()
    {
        this.Epoch = 0;
        this.Network.Initialize(this.Initializer);
        this.Optimizer.ClearCaches();

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
            var batchStride = outputs.Shape.Stride(0);

            var dY = Tensor<float>.Defaults(outputs.Shape);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                Loss.Gradient(
                    gradient: dY.AsSpan(batchIndex * batchStride, batchStride),         // Subspan of dY to store the results of the gradient computation in
                    predicted: outputs.AsSpan(batchIndex * batchStride, batchStride),   // Treat subspan of output as a vector across non-batch dimensions
                    @true: truth.AsSpan(batchIndex * batchStride, batchStride)          // Treat subspan of truth as a vector across non-batch dimensions
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
            var lr = this.LearningRateScheduler?.RateForEpoch(this.LearningRate, Epoch) ?? this.LearningRate;
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
        validate(progress);

        // Move onto next epoch
        this.Epoch++;
        return true;
    }

    public bool MoveNext()
    {
        return MoveNext(null);
    }
    
    public static Report Test(ITrainingDataSampler<float> TestingData, INetworkModule Network, LossFunction Loss, int BatchSize = 1)
    {
        var report = new Report();
        report.Reset();
        report.Epoch = -1;

        // Testing loop
        foreach ((Tensor<float> batch, Tensor<float> truth) in TestingData.Sample(batchSize: BatchSize))
        {
            // Feedforward step
            var result = Network.Forward(batch);

            // Compute loss (should this be broken up by batch size?)
            var batches = result.Shape.Length(0);
            var batchStride = result.Shape.Stride(0);

            for (var batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                var predictedSpan = result.AsSpan(batchIndex * batchStride, batchStride);
                var truthSpan = truth.AsSpan(batchIndex * batchStride, batchStride);
                var loss = Loss.Invoke(
                    predicted: predictedSpan,    // Treat subspan of output as a vector across non-batch dimensions
                    @true: truthSpan          // Treat subspan of truth as a vector across non-batch dimensions
                );

                // Record statistics to the report
                report.AddSample(loss, predictedSpan, truthSpan);
            }

        }

        return report;
    }

    public Report Test(ITrainingDataSampler<float> TestingData)
    {
        return Test(TrainingData, Network, Loss, BatchSize);
    }

    private void validate(IProgress<EpochProgress>? progress)
    {
        // Reset the training iteration report (or create a new one, but reusing is fine)
        Current.Reset();
        Current.Epoch = this.Epoch;

        // Testing loop
        int validationIndex = 0;
        foreach ((Tensor<float> batch, Tensor<float> truth) in TestingData.Sample(batchSize: BatchSize))
        {
            // Feedforward step
            var result = Network.Forward(batch);

            // Compute loss (should this be broken up by batch size?)
            var batches = result.Shape.Length(0);
            var batchStride = result.Shape.Stride(0);

            for (var batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                var predictedSpan = result.AsSpan(batchIndex * batchStride, batchStride);
                var truthSpan = truth.AsSpan(batchIndex * batchStride, batchStride);
                var loss = Loss.Invoke(
                    predicted: predictedSpan,    // Treat subspan of output as a vector across non-batch dimensions
                    @true: truthSpan          // Treat subspan of truth as a vector across non-batch dimensions
                );

                // Record statistics to the report
                Current.AddSample(loss, predictedSpan, truthSpan);
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