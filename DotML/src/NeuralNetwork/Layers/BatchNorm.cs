using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
[Untested]
[WorkInProgress]
public class BatchNorm : ConvolutionalFeedforwardNetworkLayer {

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    public Matrix<double>[] Betas {get; set;}

    public BatchNorm(Shape3D input_size) {
        this.InputShape = input_size;
        this.OutputShape = input_size;

        this.Gammas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 1.0);
        this.Betas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 0.0);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        return EvaluateSync(new BatchedFeatureSet<double>(channels))[0];
    }

    public override BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        // Compute mean and variance across the whole batch per channel
        var means = new double[features.Channels];
        var variances = new double[features.Channels];

        for (var channelIndex = 0; channelIndex < variances.Length; channelIndex++) {
            var mean = features.SelectMany(featureSet => featureSet[channelIndex]).Average();
            var variance = features.SelectMany(featureSet => featureSet[channelIndex]).Select(x => Math.Pow(x - mean, 2)).Average();

            means[channelIndex] = mean;
            variances[channelIndex] = variance;
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
        // No trainable parameters
        return 0;
    }

    public override void Visit(IConvolutionalLayerVisitor visitor) {
        throw new NotImplementedException();
    }

    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) {
        throw new NotImplementedException();
    }

    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) {
        throw new NotImplementedException();
    }
}