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
        var gammas = Weights.AsSpan();  // [C, H, W]
        var betas = Biases.AsSpan();    // [C, H, W]

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
                var gOffset = g * groupStride;                             
                var c = offset + gOffset;                                       // Chunk size of G * H * W
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
                    batch[i] = batch[i] * gammas[g + i] + betas[g + i];
                }
            }
        } 

        return x;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var originalShape = x.Shape;
        x = x.ReshapeShared(x.Shape.NormalizeRank(4));
        dy = dy.ReshapeShared(dy.Shape.NormalizeRank(4));

        int N = x.Shape.Length(0);
        int C = x.Shape.Length(1);
        int H = x.Shape.Length(2);
        int W = x.Shape.Length(3);

        int channelsPerGroup = C / Groups;
        int batchStride = x.Shape.Stride(0);
        int channelStride = x.Shape.Stride(1);
        int groupStride = channelsPerGroup * channelStride;
        int spatialSize = H * W;
        int groupSize = groupStride; // Total elements per group: channelsPerGroup * H * W

        var gamma = Weights.AsSpan(); // [C, H, W]
        var dW_tensor = Tensor<float>.Zeros(Weights.Shape); var dW = dW_tensor.AsSpan();
        var dB_tensor = Tensor<float>.Zeros(Biases.Shape); var dB = dB_tensor.AsSpan();

        var dX = Tensor<float>.Zeros(x.Shape);
        var xSpan = x.AsSpan();
        var dySpan = dy.AsSpan();
        var dxSpan = dX.AsSpan();

        for (int n = 0; n < N; n++)
        {
            int nOffset = n * batchStride;

            for (int g = 0; g < Groups; g++)
            {
                int gOffset = g * groupStride;
                int baseOffset = nOffset + gOffset;

                var xGroup = xSpan.Slice(baseOffset, groupSize);
                var dyGroup = dySpan.Slice(baseOffset, groupSize);
                var dxGroup = dxSpan.Slice(baseOffset, groupSize);

                // Compute mean/var of xGroup
                MeanAndVariance(xGroup, out float mean, out float variance);
                float stdInv = 1.0f / MathF.Sqrt(variance + epsilon);

                Span<float> x_hat = new float[groupSize];
                Span<float> dy_gamma = new float[groupSize];

                // Compute x_hat and dy * gamma
                for (int i = 0; i < groupSize; i++)
                {
                    x_hat[i] = (xGroup[i] - mean) * stdInv;
                    int globalIdx = gOffset + i;
                    dy_gamma[i] = dyGroup[i] * gamma[globalIdx];
                }

                // Accumulate dW and dB
                for (int i = 0; i < groupSize; i++)
                {
                    int globalIdx = gOffset + i;
                    dW[globalIdx] += dyGroup[i] * x_hat[i];
                    dB[globalIdx] += dyGroup[i];
                }

                // Mean values for dx
                float mean1 = 0f, mean2 = 0f;
                for (int i = 0; i < groupSize; i++)
                {
                    mean1 += dy_gamma[i];
                    mean2 += dy_gamma[i] * x_hat[i];
                }

                mean1 /= groupSize;
                mean2 /= groupSize;

                // Compute dX for this group
                for (int i = 0; i < groupSize; i++)
                {
                    dxGroup[i] = stdInv * (dy_gamma[i] - mean1 - x_hat[i] * mean2);
                }
            }
        }

        return new WeightAndBiasGradients(dX.ReshapeShared(originalShape), dW_tensor, dB_tensor);
    }

    public override void SubtractGradients(Gradients grads)
    {
        if (grads is not WeightAndBiasGradients wbg)
            throw new ArgumentException("Expected WeightAndBiasGradients", nameof(grads));

        this.Weights.SubtractWithInplace(wbg.dW);
        this.Biases.SubtractWithInplace(wbg.dB);
    }
}