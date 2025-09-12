using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs group normalization. Each channel is put into groups and normalized across the group. 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class GroupNorm2 : NormalizationLayer
{
    public Tensor<float> Weights { get; private set; }
    public Tensor<float> Biases { get; private set; }

    public int Groups {get; private set;}

    public GroupNorm2 (int channels, int height, int width, int num_groups) {
        this.Groups = num_groups;

        if (channels % num_groups != 0) {
            throw new ArgumentException($"Number of groups {num_groups} must divide the number of channels {channels} evenly.");
        }

        Weights = Tensor<float>.Ones(new TensorShape(channels, height, width));
        Biases = Tensor<float>.Zeros(new TensorShape(channels, height, width));
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
        // Assume input is [N, C, H, W], if not force it to be by collapsing leading dimensions or 1 padding
        x = x.Clone().ReshapeShared(x.Shape.NormalizeRank(4));
        var batches = x.Shape.Length(0); var batchStride = x.Shape.Stride(0);
        var channels = x.Shape.Length(1); var channelStride = x.Shape.Stride(1);
        

        // Get references to the underlying weights and biases in row-major order
        var gammas = Weights.AsSpan();
        var betas = Biases.AsSpan();

        var groups = Groups;
        if (channels % groups != 0) {
            throw new ArgumentException($"Number of groups {groups} must divide the number of channels {channels} evenly.");
        }
        var channels_per_group = channels / groups;
        var groupStride = channels_per_group * channelStride;

        for (int b = 0; b < batches; b++)
        {
            var offset = b * batchStride;  
            for (var g = 0; g < groups; g++) {
                // Compute means and variances across each group                                      
                var c = offset + g * groupStride;                               // Chunk size of G * H * W
                var batch = x.AsSpan(c, groupStride);                           // Span for the current group
                MeanAndVariance(batch, out float mean, out float variance);     // Compute mean and variance
                var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);               // Precompute sqrt once

                // Normalize each group
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
        } 

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