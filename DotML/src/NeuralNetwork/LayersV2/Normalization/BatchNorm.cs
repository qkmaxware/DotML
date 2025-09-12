using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs batch normalization. Each channel is normalized across all batches.
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class BatchNorm2 : NormalizationLayer
{

    public Tensor<float> Weights { get; private set; }
    public Tensor<float> Biases { get; private set; }

    public BatchNorm2 (int channels) {

        Weights = Tensor<float>.Ones(new TensorShape(channels));
        Biases = Tensor<float>.Zeros(new TensorShape(channels));
    }

    public override int TrainableParameterCount()
    {
        return Weights.ElementCount + Biases.ElementCount;
    }

    const float epsilon = 1e-8f;

    public override void Initialize(IInitializer initializer)
    {
        var parameters = this.TrainableParameterCount();

        var neurons = Weights.ElementCount;
        Weights.FillGenerated(() => initializer.RandomWeight(neurons, neurons, parameters));
        Biases.FillGenerated(() => initializer.RandomWeight(neurons, neurons, parameters));
    }

    public override Tensor<float> Forward(Tensor<float> x)
    {
        var shape = x.Shape.NormalizeRank(4); // Force to be [N, C, H, W]
        x = x.Clone();
        var batches = shape.Length(0); var batchStride = shape.Stride(0);
        var channels = shape.Length(1); var channelStride = shape.Stride(1);
        var arr = x.AsArray();

        Parallel.For(0, channels, (channel) =>
        {
            // Get references to all spans to this channel across all batches
            var channelInEachBatch = new SpanSurrogate<float>[batches];
            for (var i = 0; i < channelInEachBatch.Length; i++)
            {
                channelInEachBatch[i] = new SpanSurrogate<float>(arr, i * batchStride + channel * channelStride, channelStride);
            }

            // Compute the mean and variance across the entire list of channels
            MeanAndVariance(channelInEachBatch, out float mean, out float variance);
            var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);

            var gamma = Weights[channel];
            var beta = Biases[channel];

            foreach (var chan in channelInEachBatch)
            {
                var batch = chan.AsSpan();

                // Normalize 
                for (var i = 0; i < batch.Length; i++)
                {
                    batch[i] = (batch[i] - mean) * sqrt;
                }

                // Shift-scale
                for (var i = 0; i < batch.Length; i++)
                {
                    batch[i] = batch[i] * gamma + beta;
                }
            }
        });

        return x;
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