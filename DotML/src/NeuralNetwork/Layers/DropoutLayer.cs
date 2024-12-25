using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer which performs dropout
/// <see href="https://en.wikipedia.org/wiki/Dilution_(neural_networks)"/>
/// </summary>
[Untested()]
public class DropoutLayer : ConvolutionalFeedforwardNetworkLayer {
    public double DropoutRate {get; init;}
    public double KeepRate => 1 - DropoutRate;

    public DropoutLayer(Shape3D input_size, double dropoutRate) {
        this.InputShape = input_size;
        this.OutputShape = input_size;
        this.DropoutRate = Math.Clamp(dropoutRate, 0.0, 1.0);
    }

    public override void Initialize(IInitializer initializer) { }
    public override int TrainableParameterCount() => 0;

    private Random rng = new Random();

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) { 
        var channelCount = inputs.Length;
        var outputs = new Matrix<double>[channelCount];

        if (channelCount > 0) {
            var first = inputs[0];
            var mask = GetMask(first.Rows, first.Columns); // Assumes all channels are the same length

            for (var channel = 0; channel < channelCount; channel++) {
                var input = inputs[channel];
                outputs[channel] = inputs[channel].Hadamard(mask); // Elementwise multiplication with the mask
            }
        }

        return outputs;
    }

    public bool UseSharedMask {get; set;}
    private Matrix<double>? batchMask = null;
    public void ClearSharedMask() { batchMask = null; }
    public Matrix<double>? GetSharedMask() => this.batchMask;

    private Matrix<double> GetMask(int rows, int cols) {
        // Not using a shared mask, just generate a new one
        if (!this.UseSharedMask) {
            return Matrix<double>.Generate(rows, cols, () => rng.NextDouble() < DropoutRate ? 0.0 : 1.0);
        }

        // Using a shared mask, see if one already exists, or generate a new one
        lock(this) {
            if (this.batchMask.HasValue) {
                // Return the existing mask
                return this.batchMask.Value;
            } else {
                // Generate a new mask and save it for sharing
                var matrix = Matrix<double>.Generate(rows, cols, () => rng.NextDouble() < DropoutRate ? 0.0 : 1.0);
                this.batchMask = matrix;
                return matrix;
            }
        }
    }

     public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

}