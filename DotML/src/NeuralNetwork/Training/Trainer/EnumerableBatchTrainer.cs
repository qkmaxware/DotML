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

