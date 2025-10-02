
using DotML.Network.Initialization;

namespace DotML.Network.Training;

/// <summary>
/// Simple Neural Network trainer based on backpropagation
/// </summary>
/// <typeparam name="TNetwork">type of network to train (convolutional network)</typeparam>
public class ModuleTrainer
{
    public RegularizationFunction Regularization { get; set; } = new NoRegularization();
    public IOptimizer Optimizer { get; set; } = new SgdOptimizer();
    public IClippingStrategy? GradientClipping { get; set; } = null;
    public IInitializer Initializer { get; set; } = new NormalXavierInitialization();
    public LossFunction Loss {get; set;} = LossFunctions.MeanSquaredError;
    public Predicate<ModuleTrainingReport>? StopCondition { get; set; } = defaultStopCondition;
    public int MaxEpochs { get; set; } = 500;
    public float LearningRate { get; set; } = 0.01f;
    public int BatchSize { get; set; } = 1;
    public int Patience { get; set; } = 1;
    
    public IValidationReport? ValidationReport { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

    private static bool defaultStopCondition(ModuleTrainingReport report) {
        return report.AvgLoss < 0.01;
    }

    public IEnumerator<ModuleTrainingReport> EnumerateTraining(INetworkModule network, ITrainingDataSampler<float> dataset, ITrainingDataSampler<float>? validation)
    {
        return new ModuleTrainingEnumerator(
            module:             network,
            training:           dataset,
            testing:            validation is null ? dataset : validation,
            regularization:     this.Regularization,
            optimizer:          this.Optimizer,
            gradientClipping:   this.GradientClipping,
            initializer:        this.Initializer,
            loss:               this.Loss,
            stopCondition:      this.StopCondition,
            maxEpochs:          Math.Max(1, this.MaxEpochs),
            learningRate:       Math.Max(0, this.LearningRate),
            batchSize:          Math.Max(1, this.BatchSize),
            patience:           Math.Max(1, this.Patience)
        );
    }

    public void Train(INetworkModule network, ITrainingDataSampler<float> dataset, ITrainingDataSampler<float>? validation)
    {
        EnumerateTraining(network, dataset, validation).MoveEnd();
    }
}