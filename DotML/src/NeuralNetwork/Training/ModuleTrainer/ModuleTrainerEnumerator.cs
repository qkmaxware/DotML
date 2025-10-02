using System.Collections;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public class ModuleTrainingReport {
    public float MaxLoss;
    public float MinLoss;
    public float AvgLoss;
}

public class ModuleTrainingEnumerator: IEnumerator<ModuleTrainingReport> {
    public ModuleTrainingReport Current {get; private set;}
    object IEnumerator.Current => Current;

    public INetworkModule Network { get; init; }
    public ITrainingDataSampler<float> TrainingData {get; init;}
    public ITrainingDataSampler<float> TestingData {get; init;}

    public RegularizationFunction Regularization {get; init;}
    public IOptimizer Optimizer {get; init;}
    public IClippingStrategy? GradientClipping {get; init;}
    public IInitializer Initializer {get; init;}
    public LossFunction Loss {get; init;}
    public Predicate<ModuleTrainingReport>? StopCondition {get; init;}
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
        Predicate<ModuleTrainingReport>? stopCondition,
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

        this.Current = new ModuleTrainingReport();
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
        foreach ((Tensor<float> batch, Tensor<float> truth) in TrainingData.Sample(batchSize: BatchSize)) {
            // Forward step
            var context = new EvaluationContext();
            var outputs = Network.Forward(batch);

            // Compute loss/error/dy
            var loss = Loss.Gradient(Vec<float>.Wrap(outputs.AsArray()), Vec<float>.Wrap(truth.AsArray()));
            var dy = Tensor<float>.FromFlattenedArray(outputs.Shape, loss.AsArray());

            // Backward step
            var gradients = Network.Backward(dy, ctx: context, clipping: this.GradientClipping);

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
        float loss_sum = 0;
        float loss_min = float.MinValue;
        float loss_max = float.MaxValue;
        int batch_count = 0;

        // Testing loop
        foreach ((Tensor<float> batch, Tensor<float> truth) in TestingData.Sample(batchSize: BatchSize)) {
            // Feedforward step
            var result = Network.Forward(batch);

            // Compute loss (should this be broken up by batch size?)
            var loss = Loss.Invoke(Vec<float>.Wrap(result.AsArray()), Vec<float>.Wrap(truth.AsArray()));

            // Update metrics
            loss_sum += loss;
            loss_min = Math.Min(loss_min, loss);
            loss_max = Math.Max(loss_max, loss);
            batch_count++;
        }

        float loss_avg = loss_sum / batch_count;

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