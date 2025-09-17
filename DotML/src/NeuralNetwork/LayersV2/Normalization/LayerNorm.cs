using System.Drawing;
using System.Numerics;
using DotML.Network.Initialization;
using DotML.Network.Training;
using Microsoft.VisualBasic;

namespace DotML.Network;

/// <summary>
/// Layer that performs layer (non batch) normalization. Each channel is normalized across all channels.
/// <see href="https://en.wikipedia.org/wiki/Normalization_(machine_learning)"/>>
/// </summary>
public class LayerNorm2 : NormalizationLayer
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

    public LayerNorm2(int channels, int height, int width)
    {
        _weights = Tensor<float>.Ones(new TensorShape(channels, height, width));
        _biases = Tensor<float>.Zeros(new TensorShape(channels, height, width));
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
        var originalRank = channels.Shape.Rank;
        channels = channels.Clone().ReshapeShared(channels.Shape.NormalizeRank(4));
        var batches = channels.Shape.Length(0); var batchStride = channels.Shape.Stride(0);

        // Get references to the underlying weights and biases in row-major order
        var gammas = Weights.AsSpan();  // [C, H, W]
        var betas = Biases.AsSpan();    // [C, H, W]

        for (int b = 0; b < batches; b++)
        {
            // Compute means and variances across each layer
            var c = b * batchStride;                                        // Chunk size of C * H * W
            var batch = channels.AsSpan(c, batchStride);                    // Span for the current batch
            MeanAndVariance(batch, out float mean, out float variance);     // Compute mean and variance across the span
            var sqrt = 1.0f / MathF.Sqrt(variance + epsilon);               // Precompute sqrt once

            // Normalize each layer
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
                batch[i] = batch[i] * gammas[i] + betas[i];
            }
        } 

        return channels.Squeeze(0..^originalRank);
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var originalShape = x.Shape;
        x = x.ReshapeShared(x.Shape.NormalizeRank(4));
        dy = dy.ReshapeShared(dy.Shape.NormalizeRank(4));

        int batches = x.Shape.Length(0);
        int normSize = x.Shape.Length(1) * x.Shape.Length(2) * x.Shape.Length(3); // size of C*H*W
        int batchStride = x.Shape.Stride(0);

        var gamma = Weights.AsSpan(); // [C, H, W]
        var dW_tensor = Tensor<float>.Zeros(Weights.Shape); var dW = dW_tensor.AsSpan();
        var dB_tensor = Tensor<float>.Zeros(Biases.Shape); var dB = dB_tensor.AsSpan();

        var dX = Tensor<float>.Zeros(x.Shape);

        var xSpan = x.AsSpan();
        var dySpan = dy.AsSpan();
        var dxSpan = dX.AsSpan();

        float epsilon = 1e-8f;

        for (int b = 0; b < batches; b++)
        {
            int offset = b * batchStride;

            var xBatch = xSpan.Slice(offset, normSize);
            var dyBatch = dySpan.Slice(offset, normSize);
            var dxBatch = dxSpan.Slice(offset, normSize);

            // Compute mean and variance of x
            MeanAndVariance(xBatch, out float mean, out float variance);
            float stdInv = 1.0f / MathF.Sqrt(variance + epsilon);

            // Allocate temporary x_hat and dy*gamma
            Span<float> x_hat = new float[normSize];
            Span<float> dy_gamma = new float[normSize];

            // Compute x_hat and dy*gamma
            for (int i = 0; i < normSize; i++)
            {
                x_hat[i] = (xBatch[i] - mean) * stdInv;
                dy_gamma[i] = dyBatch[i] * gamma[i];
            }

            // Accumulate dW and dB
            for (int i = 0; i < normSize; i++)
            {
                dW[i] += dyBatch[i] * x_hat[i];
                dB[i] += dyBatch[i];
            }

            // Compute means needed for dx
            float mean1 = 0f;
            float mean2 = 0f;
            for (int i = 0; i < normSize; i++)
            {
                mean1 += dy_gamma[i];
                mean2 += dy_gamma[i] * x_hat[i];
            }
            mean1 /= normSize;
            mean2 /= normSize;

            // Compute dx
            for (int i = 0; i < normSize; i++)
            {
                dxBatch[i] = stdInv * (dy_gamma[i] - mean1 - x_hat[i] * mean2);
            }
        }

        return new WeightAndBiasGradients(dX.ReshapeShared(originalShape) /*Return x to its original shape (just in case we need that)*/, dW_tensor, dB_tensor);
    }

    public override void SubtractGradients(Gradients grads)
    {
        if (grads is not WeightAndBiasGradients wbg)
            throw new ArgumentException("Expected WeightAndBiasGradients", nameof(grads));

        this.Weights.SubtractWithInplace(wbg.dW);
        this.Biases.SubtractWithInplace(wbg.dB);
    }
}