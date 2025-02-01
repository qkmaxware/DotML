using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
[Untested]
public class BatchNorm : FeedforwardNetworkLayer {

    private double running_mean_momentum = 0.9;
    public Vec<double> RunningMean;
    private double running_variance_momentum = 0.9;
    public Vec<double> RunningVariance;

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    [JsonIgnore]
    public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    [JsonIgnore]
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

    public void ComputeMeansAndVariances(BatchedFeatureSet<double> features, out double[] mean_vec, out double[] variance_vec) {
        if (IsInference || features.Batches < 2) {
            mean_vec = (double[])this.RunningMean;
            variance_vec = (double[])this.RunningVariance;
        } else {
            var means = new double[features.Channels];
            var variances = new double[features.Channels];
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
            }

            mean_vec = means;
            variance_vec = variances;
        }
    }

    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        // Compute mean and variance across the whole batch per channel
        this.ComputeMeansAndVariances(features, out var means, out var variances);

        // Perform running mean/variance computation
        if (this.IsInference && features.Batches > 2) {
            for (var channelIndex = 0; channelIndex < variances.Length; channelIndex++) {
                var mean = means[channelIndex];
                var variance = variances[channelIndex];
                
                ((double[])RunningMean)[channelIndex] = running_mean_momentum * mean + (1 - running_mean_momentum) * RunningMean[channelIndex];
                ((double[])RunningVariance)[channelIndex] = running_variance_momentum * variance + (1 - running_variance_momentum) * RunningVariance[channelIndex];
            }
        }

        // When run with a batch size of > 1
        var results = new FeatureSet<double>[features.Batches];
        Parallel.For(0, features.Batches, (batchIndex) => {
            var featureSet = features[batchIndex];
            results[batchIndex] = new FeatureSet<double>(
                featureSet.Select((matrix, channelIndex) => {
                    // Compute mean and variance for the channel
                    var mean = means[channelIndex];
                    var variance = variances[channelIndex];

                    // Normalize the channel using mean and variance
                    var normalizedMatrix = matrix.Transform(x => (x - mean)  / Math.Sqrt(variance + 1e-8));

                    // Apply scaling (gamma) and shifting (beta)
                    normalizedMatrix.HadamardWithInplace(Gammas[channelIndex]); // output = output .* gamma
                    normalizedMatrix.AddWithInplace(Betas[channelIndex]); // output = output + beta     

                    return normalizedMatrix;
                }).ToArray()
            );
        });
        return new BatchedFeatureSet<double>(results);
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

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);

    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}