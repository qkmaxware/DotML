using System.Drawing;
using System.Numerics;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer that performs group normalization. Each channel is put into groups and normalized across the group. 
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class GroupNorm2 : NormalizationLayer, IWeightsAndBiasNetworkModule
{
    private Tensor<float> _weights;
    public Tensor<float> Weights
    {
        get => _weights;
        set
        {
            if (!value.Shape.Equals(_weights.Shape))
                throw new ArgumentException("Cannot change the shape of the layer weights via assignment");
            _weights = value;
        }
    }
    private Tensor<float> _biases;
    public Tensor<float> Biases
    {
        get => _biases;
        set
        {
            if (!value.Shape.Equals(_biases.Shape))
                throw new ArgumentException("Cannot change the shape of the layer biases via assignment");
            _biases = value;
        }
    }

    public TensorShape NormalizedShape { get; init; }

    public int Groups { get; private set; }

    public GroupNorm2(int num_groups, TensorShape normalizedShape)
    {
        this.NormalizedShape = normalizedShape;
        this.Groups = num_groups;

        if (normalizedShape.Length(0) % num_groups != 0)
        {
            throw new ArgumentException($"Number of groups {num_groups} must divide the number of channels {normalizedShape.Length(0)} evenly.");
        }

        _weights = Tensor<float>.Ones(normalizedShape);
        _biases = Tensor<float>.Zeros(normalizedShape);
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

    public override TensorShape ForwardShape(TensorShape input) => input;

    public override Tensor<float> Forward(Tensor<float> x)
    {
        // Assume input is [N, C, H, W], if not force it to be by collapsing leading dimensions or 1 padding
        var originalShape = x.Shape;
        x = x.Clone().ReshapeShared(x.Shape.NormalizeRank(NormalizedShape.Rank + 1));
        var batches = x.Shape.Length(0); var batchStride = x.Shape.Stride(0);
        var channels = x.Shape.Length(1); var channelStride = x.Shape.Stride(1);


        // Get references to the underlying weights and biases in row-major order
        var gammas = Weights.AsSpan();  
        var betas = Biases.AsSpan();    

        var groups = Groups;
        if (channels % groups != 0)
        {
            throw new ArgumentException($"Number of groups {groups} must divide the number of channels {channels} evenly.");
        }
        var channels_per_group = channels / groups;
        var groupStride = channels_per_group * channelStride;

        for (int b = 0; b < batches; b++)
        {
            var offset = b * batchStride;
            for (var g = 0; g < groups; g++)
            {
                // Compute means and variances across each group         
                var gOffset = g * groupStride;
                var c = offset + gOffset;                                       // Chunk size of G * H * W
                var batch = x.AsSpan(c, groupStride);                           // Span for the current group
                MeanAndVariance(batch, out float mean, out float variance);     // Compute mean and variance
                var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);               // Precompute sqrt once

                // Normalize each group
                int i = 0;
                if (Vector.IsHardwareAccelerated && Vector<float>.IsSupported)
                {
                    var sqrtVector = new Vector<float>(sqrt);
                    var meanVector = new Vector<float>(mean);
                    int simdLength = Vector<float>.Count;
                    int simdLimit = batch.Length - simdLength + 1;

                    for (; i < simdLimit; i += simdLength)
                    {
                        var simdSlice = batch.Slice(i, simdLength);
                        var batchVec = new Vector<float>(simdSlice);
                        ((batchVec - meanVector) * sqrtVector).CopyTo(simdSlice);
                    }
                }
                for (; i < batch.Length; i++)
                {
                    batch[i] = (batch[i] - mean) * sqrt;
                }

                // Shift-scale
                i = 0;
                if (Vector.IsHardwareAccelerated && Vector<float>.IsSupported)
                {
                    int simdLength = Vector<float>.Count;
                    int simdLimit = batch.Length - simdLength + 1;

                    for (; i < simdLimit; i += simdLength)
                    {
                        var simdSlice = batch.Slice(i, simdLength);
                        var batchVec = new Vector<float>(simdSlice);
                        var gammaVec = new Vector<float>(gammas.Slice(i, simdLength));
                        var betaVec = new Vector<float>(betas.Slice(i, simdLength));
                        ((batchVec * gammaVec) + betaVec).CopyTo(simdSlice);
                    }
                }
                for (; i < batch.Length; i++)
                {
                    batch[i] = batch[i] * gammas[g + i] + betas[g + i];
                }
            }
        }

        return x.ReshapeShared(originalShape);
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var originalShape = x.Shape;
        x = x.ReshapeShared(x.Shape.NormalizeRank(NormalizedShape.Rank + 1));
        dy = dy.ReshapeShared(dy.Shape.NormalizeRank(NormalizedShape.Rank + 1));

        int N = x.Shape.Length(0);
        int C = x.Shape.Length(1);

        int channelsPerGroup = C / Groups;
        int batchStride = x.Shape.Stride(0);
        int channelStride = x.Shape.Stride(1);
        int groupStride = channelsPerGroup * channelStride;
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

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not WeightAndBiasGradients wbg)
            throw new ArgumentException("Expected WeightAndBiasGradients", nameof(gradients));

        var dW = wbg.dW;
        var dB = wbg.dB;

        // Apply regularization to weights
        if (regularization is not null)
        {
            dW.ElementWiseBinaryInplace(Weights, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
            dB.ElementWiseBinaryInplace(Biases, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
        }

        // Apply optimizer
        optimizer.UpdateParameter(this, nameof(Weights), learningRate, this.Weights, dW);
        optimizer.UpdateParameter(this, nameof(Biases), learningRate, this.Biases, dB);
    }
    
    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}