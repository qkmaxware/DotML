using System.Drawing;
using System.Numerics;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs layer (non batch) normalization. Each channel is normalized across all channels.
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class LayerNorm2 : NormalizationLayer
{

    public Tensor<float> Weights { get; private set; }
    public Tensor<float> Biases { get; private set; }

    public LayerNorm2(int channels, int height, int width)
    {
        Weights = Tensor<float>.Ones(new TensorShape(channels, height, width));
        Biases = Tensor<float>.Zeros(new TensorShape(channels, height, width));
    }

    public override int TrainableParameterCount()
    {
        return Weights.ElementCount + Biases.ElementCount;
    }

    public override void Initialize(IInitializer initializer)
    {
        var parameters = this.TrainableParameterCount();

        var neurons = Weights.ElementCount;
        Weights.FillGenerated(() => initializer.RandomWeight(neurons, neurons, parameters));
        Biases.FillGenerated(() => initializer.RandomWeight(neurons, neurons, parameters));
    }
    
    const float epsilon = 1e-8f;

    public override Tensor<float> Forward(Tensor<float> channels)
    {
        // Assume input is [N, C, H, W], if not force it to be by collapsing leading dimensions or 1 padding
        channels = channels.Clone().ReshapeShared(channels.Shape.NormalizeRank(4));
        var batches = channels.Shape.Length(0); var batchStride = channels.Shape.Stride(0);

        // Get references to the underlying weights and biases in row-major order
        var gammas = Weights.AsSpan();
        var betas = Biases.AsSpan();

        for (int b = 0; b < batches; b++)
        {
            // Compute means and variances across each layer
            var c = b * batchStride;                                        // Chunk size of C * H * W
            var batch = channels.AsSpan(c, batchStride);                    // Span for the current batch
            MeanAndVariance(batch, out float mean, out float variance);     // Compute mean and variance
            var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);               // Precompute sqrt once

            // Normalize each layer
            for (var i = 0; i < batch.Length; i++)
            {
                batch[i] = (batch[i] - mean) * sqrt;
            }

            // Shift-scale
            for (var i = 0; i < batch.Length; i++)
            {
                batch[i] = batch[i] * gammas[c + i] + betas[c + i];
            }
        } 

        return channels;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        throw new NotImplementedException();
    }

    public override void SubtractGradients(Gradients grads)
    {
        throw new NotImplementedException();
    }
}