using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer that performs layer (non batch) normalization 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
[Untested()]
public class LayerNorm : FeedforwardNetworkLayer {

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    [JsonIgnore] public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    [JsonIgnore] public Matrix<double>[] Betas {get; set;}

    public LayerNorm(Shape3D input_size) {
        this.InputShape = input_size;
        this.OutputShape = input_size;

        this.Gammas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Gammas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 1.0);
        this.Betas = new Matrix<double>[input_size.Channels];
        for (var i = 0; i < input_size.Channels; i++)
            Betas[i] = new Matrix<double>(input_size.Rows, input_size.Columns, 0.0);
    }

    public void ComputeMeansAndVariances(FeatureSet<double> features, out double[] mean_vec, out double[] variance_vec) {
        var len = features.Channels;

        var local_means = new double[len];
        var local_variances = new double[len];

        for (var i = 0; i < len; i++) {
            var neurons = features[i];

            var local_mean = neurons.Average();
            var local_variance = neurons.Select(v => Math.Pow(v - local_mean, 2)).Average();

            local_means[i] = local_mean;
            local_variances[i] = local_variance;
        }

        mean_vec = local_means;
        variance_vec = local_variances;
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var len = channels.Channels;
        Matrix<double>[] outputs = new Matrix<double>[len];

        // Compute the mean and variance across all inputs 
        ComputeMeansAndVariances(channels, out double[] means, out double[] variances);

        // Perform the normalization for each feature
        for (var channel = 0; channel < len; channel++) {
            // Get feature at channel
            Matrix<double> features = channels[channel];

            // Get mean, variance as computed
            var mean = means[channel];
            var variance = variances[channel];
            var sqrt = 1.0 / Math.Sqrt(variance + 1e-8);

            // Normalize the channel using mean and variance
            var output = features.Transform(v => (v - mean) * sqrt);

            // Apply scaling (gamma) and shifting (beta)
            output.HadamardWithInplace(Gammas[channel]); // output = output .* gamma
            output.AddWithInplace(Betas[channel]); // output = output + beta

            // Save results
            outputs[channel] = output;                              
        }

        return new FeatureSet<double>(outputs);
    }

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() {
        return InputShape.Count * 2;
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}