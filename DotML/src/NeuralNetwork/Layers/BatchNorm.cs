using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
[Untested]
public class BatchNorm : ConvolutionalFeedforwardNetworkLayer {

    private double running_mean_momentum = 0.9;
    public Vec<double> RunningMean;
    private double running_variance_momentum = 0.9;
    public Vec<double> RunningVariance;

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    public Matrix<double>[] Betas {get; set;}

    public BatchNorm(Shape3D input_size, double mean_momentum = 0.9, double variance_momentum = 0.9) {
        this.InputShape = input_size;
        this.OutputShape = input_size;

        this.running_mean_momentum = mean_momentum;
        this.running_variance_momentum = variance_momentum;

        var rmean = new double[input_size.Channels];
        Array.Fill(rmean, 0.0);
        this.RunningMean = rmean;
        var rvariance = new double[input_size.Channels];
        Array.Fill(rvariance, 1.0);
        this.RunningVariance = rvariance;

        this.Gammas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 1.0);
        this.Betas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 0.0);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        // Basically this is used when not training
        // Not sure if this is how to do it when there is no batching (like during evaluation rather than training)
        return EvaluateSync(new BatchedFeatureSet<double>(channels))[0];
    }

    /*
    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
    int batchSize = features.Length;
    int channels = features[0].Length;
    int rows = features[0][0].Rows;
    int cols = features[0][0].Columns;

    // Compute mean and variance across the whole batch for each channel
    var means = new double[channels];
    var variances = new double[channels];

    The two formulas for variance calculation are indeed mathematically equivalent. 
    You can use either method to compute the variance. 
    The second method, using the sum of squares, is often preferred in numerical computations for its stability and efficiency.
    
    for (var channelIndex = 0; channelIndex < channels; channelIndex++) {
        double sum = 0.0;
        double sumSq = 0.0;
        int count = 0;

        foreach (var featureSet in features) {
            var matrix = featureSet[channelIndex];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double value = matrix[i, j];
                    sum += value;
                    sumSq += value * value;
                    count++;
                }
            }
        }

        double mean = sum / count;
        double variance = (sumSq / count) - (mean * mean);

        means[channelIndex] = mean;
        variances[channelIndex] = variance;
    }

    // Normalize, scale, and shift
    var normalized = new BatchedFeatureSet<double>(
        features.Select(featureSet => {
            return new FeatureSet<double>(
                featureSet.Select((matrix, channelIndex) => {
                    var mean = means[channelIndex];
                    var variance = variances[channelIndex];

                    var normalizedMatrix = matrix.Transform(x => (x - mean) / Math.Sqrt(variance + 1e-8));
                    var scaledAndShiftedMatrix = normalizedMatrix.Transform((x, i, j) => Gammas[channelIndex][i, j] * x + Betas[channelIndex][i, j]);

                    return scaledAndShiftedMatrix;
                }).ToArray()
            );
        }).ToArray()
    );

    return normalized;
}
    */

    public bool IsTrainingMode {get; set;} = false;
    public bool IsInferenceMode {get => !IsTrainingMode; set => IsTrainingMode = !value; }

    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        // Basically this is used only when training
        // Compute mean and variance across the whole batch per channel
        double[] means = new double[features.Channels];
        double[] variances = new double[features.Channels];

        if (IsInferenceMode || features.Batches < 2) {
            means = (double[])this.RunningMean;
            variances = (double[])this.RunningVariance;
        } else {
            for (var channelIndex = 0; channelIndex < variances.Length; channelIndex++) {
                double sum = 0.0;
                double sumSq = 0.0;
                int count = 0;

                foreach (var featureSet in features) {
                    var matrix = featureSet[channelIndex];
                    var rows = matrix.Rows;
                    var cols = matrix.Columns;
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            double value = matrix[i, j];
                            sum += value;
                            sumSq += value * value;
                            count++;
                        }
                    }
                }

                count = Math.Max(count, 1); // Avoid division by zero
                double mean = sum / count;
                double variance = (sumSq / count) - (mean * mean);

                means[channelIndex] = mean;
                variances[channelIndex] = variance;

                // Perform running mean/variance computation
                ((double[])RunningMean)[channelIndex] = running_mean_momentum * mean + (1 - running_mean_momentum) * RunningMean[channelIndex];
                ((double[])RunningVariance)[channelIndex] = running_variance_momentum * variance + (1 - running_variance_momentum) * RunningVariance[channelIndex];
            }
        }

        // When run with a batch size of > 1
        var normalized = new BatchedFeatureSet<double>(
            features.Select(featureSet => {
                return 
                new FeatureSet<double>(
                    featureSet.Select((matrix, channelIndex) => {
                        // Compute mean and variance for the channel
                        var mean = means[channelIndex];
                        var variance = variances[channelIndex];

                        // Normalize the channel using mean and variance
                        var normalizedMatrix = matrix.Transform(x => (x - mean)  / Math.Sqrt(variance + 1e-8));

                        // Apply scaling (gamma) and shifting (beta)
                        Matrix<double>.HadamardInplace(normalizedMatrix, normalizedMatrix, Gammas[channelIndex]);  // output = output .* gamma
                        Matrix<double>.AddInplace(normalizedMatrix, normalizedMatrix, Betas[channelIndex]);        // output = output + beta

                        return normalizedMatrix;
                    }).ToArray()
                );
            }).ToArray()
        );
        return normalized;
    }

    public override void Initialize(IInitializer initializer) {
        // No need to initialize any weights
    }

    public override int TrainableParameterCount() {
        return InputShape.Count * 2;
    }

    /// <summary>
    /// Number of un-trainable parameters in this layer
    /// </summary>
    /// <returns>Number of un-trainable parameters</returns>
    public override int UnTrainableParameterCount() => this.RunningVariance.Dimensionality + this.RunningMean.Dimensionality;

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);

    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}