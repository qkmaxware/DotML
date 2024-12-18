using System.Drawing;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer that performs layer (non batch) normalization 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class LayerNorm : ConvolutionalFeedforwardNetworkLayer {

    /// <summary>
    /// Normalization scaling factor
    /// </summary>
    public Matrix<double>[] Gammas {get; set;}
    /// <summary>
    /// Normalization shifting offset
    /// </summary>
    public Matrix<double>[] Betas {get; set;}

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

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] channels) {
        var len = channels.Length;
        Matrix<double>[] outputs = new Matrix<double>[len];

        for (var channel = 0; channel < len; channel++) {
            Matrix<double> features = channels[channel];

            // Compute mean and variance for the channel
            var mean = features.Average();
            var variance = features.Select(v => Math.Pow(v - mean, 2)).Average();

            // Normalize the channel using mean and variance
            var output = features.Transform(v => (v - mean) / Math.Sqrt(variance + 1e-8));

            // Apply scaling (gamma) and shifting (beta)
            Matrix<double>.HadamardInplace(output, output, Gammas[channel]);  // output = output .* gamma
            Matrix<double>.AddInplace(output, output, Betas[channel]);        // output = output + beta

            // Save results
            outputs[channel] = output;                              
        }

        return outputs;
    }

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() {
        return InputShape.Count * 2;
    }

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}