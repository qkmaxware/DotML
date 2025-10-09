using System.Collections;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public class ModuleTrainingEnumerator: IEnumerator<ModuleTrainingEnumerator.Report> {
    
    /// <summary>
    /// Training report
    /// </summary>
    public class Report
    {
        public float MaxLoss;
        public float MinLoss;
        public float AvgLoss;
    }

    public Report Current { get; private set; }
    object IEnumerator.Current => Current;

    public INetworkModule Network { get; init; }
    public ITrainingDataSampler<float> TrainingData {get; init;}
    public ITrainingDataSampler<float> TestingData {get; init;}

    public RegularizationFunction Regularization {get; init;}
    public IOptimizer Optimizer {get; init;}
    public IClippingStrategy? GradientClipping {get; init;}
    public IInitializer Initializer {get; init;}
    public LossFunction Loss {get; init;}
    public Predicate<Report>? StopCondition {get; init;}
    public int Epoch {get; private set;}
    public int MaxEpochs {get; init;}
    public float LearningRate {get; init;}

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
        IClippingStrategy? gradientClipping,
        IInitializer initializer,
        LossFunction loss,
        Predicate<Report>? stopCondition,
        int maxEpochs,
        float learningRate,
        int batchSize,
        int patience
    )
    {
        this.Network = module;
        this.TrainingData = training;
        this.TestingData = testing;
        this.Regularization = regularization;
        this.Optimizer = optimizer;
        this.GradientClipping = gradientClipping;
        this.Initializer = initializer;
        this.Loss = loss;
        this.StopCondition = stopCondition;
        this.MaxEpochs = maxEpochs;
        this.LearningRate = learningRate;
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

    public bool MoveNext() {
        if (Epoch >= MaxEpochs || target_reached)
            return false;

        // Training loop
        foreach ((Tensor<float> batch, Tensor<float> truth) in TrainingData.Sample(batchSize: BatchSize))
        {
            // Forward step
            var context = new EvaluationContext(EvaluationMode.Training);
            var outputs = Network.Forward(batch);

            // Compute loss/error/dy
            var batches = outputs.Shape.Length(0);
            var batchStride = outputs.Shape.Stride(0);

            var dY = Tensor<float>.Defaults(outputs.Shape);
            for (var batchIndex = 0; batchIndex < batches; batchIndex++) {
                Loss.Gradient(
                    gradient: dY.AsSpan(batchIndex * batchStride, batchStride),         // Subspan of dY to store the results of the gradient computation in
                    predicted: outputs.AsSpan(batchIndex * batchStride, batchStride),   // Treat subspan of output as a vector across non-batch dimensions
                    @true: truth.AsSpan(batchIndex * batchStride, batchStride)          // Treat subspan of truth as a vector across non-batch dimensions
                );
            }

            // Backward step
            var gradients = Network.Backward(dY, ctx: context, clipping: this.GradientClipping);

            // Update step
            Network.Update(
                this.LearningRate,
                gradients,
                optimizer: this.Optimizer,
                regularization: this.Regularization
            );
        }

        // Validate model accuracy
        validate();

        // Move onto next epoch
        this.Epoch++;
        return true;
    }

    private void validate() {
        float loss_sum = 0;                 // sum of all losses
        float loss_min = float.MinValue;    // min loss
        float loss_max = float.MaxValue;    // max loss
        int loss_count = 0;                 // number of losses computed
        int batch_count = 0;                // number of batches checked

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
                var loss = Loss.Invoke(
                    predicted: result.AsSpan(batchIndex * batchStride, batchStride),    // Treat subspan of output as a vector across non-batch dimensions
                    @true: truth.AsSpan(batchIndex * batchStride, batchStride)          // Treat subspan of truth as a vector across non-batch dimensions
                );

                // Update metrics
                loss_sum += loss;
                loss_min = Math.Min(loss_min, loss);
                loss_max = Math.Max(loss_max, loss);
                loss_count++;
            }

            batch_count++;
        }

        float loss_avg = loss_sum / loss_count;

        // Update report    
        this.Current.MaxLoss = loss_max;
        this.Current.MinLoss = loss_min;
        this.Current.AvgLoss = loss_avg;

        // Check stop condition based on the report
        if (StopCondition is not null && StopCondition.Invoke(this.Current)) {
            // Decrement patience and stop if patience reached
            _patienceCounter--;
            if (_patienceCounter <= 0)
                this.target_reached = true;
        } else {
            // Reset patience count
            _patienceCounter = this.Patience;
        }
    }
}