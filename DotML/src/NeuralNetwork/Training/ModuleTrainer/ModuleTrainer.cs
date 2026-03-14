
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
    public ILocalClippingStrategy<float>? LocalClipping { get; set; } = null;
    public IGlobalClippingStrategy<float>? GlobalClipping { get; set; } = null;
    public IInitializer Initializer { get; set; } = new NormalXavierInitialization();
    public LossFunction Loss {get; set;} = LossFunctions.MeanSquaredError;
    public Predicate<ModuleTrainingEnumerator.Report>? StopCondition { get; set; } = StopOnAvgLossDefault;
    public int MaxEpochs { get; set; } = 500;
    public ILearningRateScheduler LearningRateScheduler { get; set; } = new ConstantRate(0.01f);
    public int BatchSize { get; set; } = 1;
    public int Patience { get; set; } = 1;

    public const float DefaultLossThreshold = 0.01f;

    public List<IMetricsProvider> Metrics {get; private set;} = new List<IMetricsProvider>();

    public static bool StopOnAvgLossDefault(ModuleTrainingEnumerator.Report report)
    {
        return report.Loss.Average < DefaultLossThreshold;
    }
    
    public static bool StopOnMaxLossDefault(ModuleTrainingEnumerator.Report report) {
        return report.Loss.Max < DefaultLossThreshold;
    }

    public static bool StopOnMinLossDefault(ModuleTrainingEnumerator.Report report) {
        return report.Loss.Min < DefaultLossThreshold;
    }

    public IEnumerator<ModuleTrainingEnumerator.Report> EnumerateTraining(INetworkModule network, ITrainingDataSampler<float> dataset, ITrainingDataSampler<float>? validation)
    {
        var trainer = new ModuleTrainingEnumerator(
            module: network,
            training: dataset,
            testing: validation is null ? dataset : validation,
            regularization: this.Regularization,
            optimizer: this.Optimizer,
            localClipping: this.LocalClipping,
            globalClipping: this.GlobalClipping,
            initializer: this.Initializer,
            loss: this.Loss,
            stopCondition: this.StopCondition,
            maxEpochs: Math.Max(1, this.MaxEpochs),
            scheduler: LearningRateScheduler,
            batchSize: Math.Max(1, this.BatchSize),
            patience: Math.Max(1, this.Patience),
            metricsProviders: this.Metrics
        );

        return trainer;
    }

    public void Train(INetworkModule network, ITrainingDataSampler<float> dataset, ITrainingDataSampler<float>? validation)
    {
        EnumerateTraining(network, dataset, validation).MoveEnd();
    }
}