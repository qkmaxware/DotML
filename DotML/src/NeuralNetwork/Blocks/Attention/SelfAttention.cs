using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class SelfAttention : Attention
{
    private DenseLinear q_proj;
    private DenseLinear k_proj;
    private DenseLinear v_proj;
    private DenseLinear output_proj;

    public DenseLinear QueryProjection
    {
        get => q_proj;
        set
        {
            if (value.InputSize != q_proj.InputSize || value.OutputSize != q_proj.OutputSize)
                throw new ArgumentException("New Q projection must have same input and output sizes as existing projection.");
            q_proj = value;
        }
    }

    public DenseLinear KeyProjection
    {
        get => k_proj;
        set
        {
            if (value.InputSize != k_proj.InputSize || value.OutputSize != k_proj.OutputSize)
                throw new ArgumentException("New K projection must have same input and output sizes as existing projection.");
            k_proj = value;
        }
    }

    public DenseLinear ValueProjection
    {
        get => v_proj;
        set
        {
            if (value.InputSize != v_proj.InputSize || value.OutputSize != v_proj.OutputSize)
                throw new ArgumentException("New V projection must have same input and output sizes as existing projection.");
            v_proj = value;
        }
    }

    public DenseLinear OutputProjection
    {
        get => output_proj;
        set
        {
            if (value.InputSize != output_proj.InputSize || value.OutputSize != output_proj.OutputSize)
                throw new ArgumentException("New output projection must have same input and output sizes as existing projection.");
            output_proj = value;
        }
    }


    public SelfAttention(int d_model, int d_k, int heads = 1) : base(d_model, d_k, heads)
    {
        q_proj = new DenseLinear(d_model, d_k);
        k_proj = new DenseLinear(d_model, d_k);
        v_proj = new DenseLinear(d_model, d_k);

        output_proj = new DenseLinear(d_k * heads, d_model);
    }

    public override void Initialize(IInitializer initializer)
    {
        q_proj.Initialize(initializer);
        k_proj.Initialize(initializer);
        v_proj.Initialize(initializer);
    }

    protected override Tensor<float> ComputeKey(Tensor<float> input, EvaluationContext? ctx)
    {
        // Project input -> [B, T, d_k] and replicate per-head -> [B, H, T, d_k]
        var flatinput = input.ReshapeShared(input.Shape.NormalizeRank(2)); // reshape to [B * T, d_model] for batched matrix-vector multiplication within DenseLinear
        var base_k = k_proj.Forward(flatinput, ctx, null); // [B * T, d_k]
        base_k = base_k.ReshapeShared(input.Shape); // Reshape back to [B, T, d_k]

        int B = base_k.Shape.Length(0);
        int T = base_k.Shape.Length(1);
        int DK = base_k.Shape.Length(2);
        int H = this.Heads;

        var K = Tensor<float>.Defaults(new Shape(B, H, T, DK));
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int k = 0; k < DK; k++)
                    {
                        K[b, h, t, k] = base_k[b, t, k];
                    }
                }
            }
        }

        return K;
    }

    protected override Gradients ComputeKeyGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        // dY: [B, H, T, d_k] -> sum over heads -> [B, T, d_k]
        int B = dY.Shape.Length(0);
        int H = dY.Shape.Length(1);
        int T = dY.Shape.Length(2);
        int DK = dY.Shape.Length(3);

        var outputShape = new Shape(B, T, DK);
        var dySum = Tensor<float>.Defaults(outputShape);
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int k = 0; k < DK; k++)
                    {
                        dySum[b, t, k] += dY[b, h, t, k];
                    }
                }
            }
        }

        // Use layer wrapper Backward that reads saved IOContext and applies clipping if provided
        dySum = dySum.ReshapeShared(new Shape(B * T, DK)); // Reshape to [B * T, d_k] for DenseLinear backward
        var sub = (WeightAndBiasGradients)k_proj.Backward(dySum, ctx, clipping);
        return new WeightAndBiasGradients(sub.dX.ReshapeShared(outputShape), sub.dW, sub.dB); // Reshape dX back to [B, T, d_k] to be compatible with next parts of the Attention backwards algorithm
    }

    protected override Tensor<float> ComputeQuery(Tensor<float> input, EvaluationContext? ctx)
    {
        // Project input -> [B, T, d_k] and replicate per-head -> [B, H, T, d_k]
        var flatinput = input.ReshapeShared(input.Shape.NormalizeRank(2)); // Collapse B,T to single dimension for batched matrix-vector multiplication within DenseLinear
        var base_q = q_proj.Forward(flatinput, ctx, null); // [B * T, d_k]
        base_q = base_q.ReshapeShared(input.Shape); // Reshape back to [B, T, d_k]

        int B = base_q.Shape.Length(0);
        int T = base_q.Shape.Length(1);
        int DK = base_q.Shape.Length(2);
        int H = this.Heads;

        var Q = Tensor<float>.Defaults(new Shape(B, H, T, DK));
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int k = 0; k < DK; k++)
                    {
                        Q[b, h, t, k] = base_q[b, t, k];
                    }
                }
            }
        }

        return Q;
    }

    protected override Gradients ComputeQueryGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        // dY: [B, H, T, d_k] -> sum over heads -> [B, T, d_k]
        int B = dY.Shape.Length(0);
        int H = dY.Shape.Length(1);
        int T = dY.Shape.Length(2);
        int DK = dY.Shape.Length(3);

        var outputShape = new Shape(B, T, DK);
        var dySum = Tensor<float>.Defaults(outputShape);
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int k = 0; k < DK; k++)
                    {
                        dySum[b, t, k] += dY[b, h, t, k];
                    }
                }
            }
        }

        dySum = dySum.ReshapeShared(new Shape(B * T, DK)); // Reshape to [B * T, d_k] for DenseLinear backward
        var sub = (WeightAndBiasGradients)q_proj.Backward(dySum, ctx, clipping);
        return new WeightAndBiasGradients(sub.dX.ReshapeShared(outputShape), sub.dW, sub.dB);
    }

    protected override Tensor<float> ComputeValue(Tensor<float> input, EvaluationContext? ctx)
    {
        // Project input -> [B, T, d_k] and replicate per-head -> [B, H, T, d_k]
        var flatinput = input.ReshapeShared(input.Shape.NormalizeRank(2)); // Collapse B,T to single dimension for batched matrix-vector multiplication within DenseLinear
        var base_v = v_proj.Forward(flatinput, ctx, null); // [B * T, d_k]
        base_v = base_v.ReshapeShared(input.Shape); // Reshape back to [B, T, d_k]

        int B = base_v.Shape.Length(0);
        int T = base_v.Shape.Length(1);
        int DK = base_v.Shape.Length(2);
        int H = this.Heads;

        var V = Tensor<float>.Defaults(new Shape(B, H, T, DK));
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int k = 0; k < DK; k++)
                    {
                        V[b, h, t, k] = base_v[b, t, k];
                    }
                }
            }
        }

        return V;
    }

    protected override Gradients ComputeValueGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        // dY: [B, H, T, d_k] -> sum over heads -> [B, T, d_k]
        int B = dY.Shape.Length(0);
        int H = dY.Shape.Length(1);
        int T = dY.Shape.Length(2);
        int DK = dY.Shape.Length(3);

        var outputShape = new Shape(B, T, DK);
        var dySum = Tensor<float>.Defaults(outputShape);
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int k = 0; k < DK; k++)
                    {
                        dySum[b, t, k] += dY[b, h, t, k];
                    }
                }
            }
        }

        dySum = dySum.ReshapeShared(new Shape(B * T, DK)); // Reshape to [B * T, d_k] for DenseLinear backward
        var sub = (WeightAndBiasGradients)v_proj.Backward(dySum, ctx, clipping);
        return new WeightAndBiasGradients(sub.dX.ReshapeShared(outputShape), sub.dW, sub.dB);
    }

    public override Tensor<float> ProjectionOutput(Tensor<float> outputHeads, EvaluationContext? ctx)
    {
        // Input shape is [B, T, H*d_k]
        var B = outputHeads.Shape.Length(0);
        var T = outputHeads.Shape.Length(1);
        var flatshape = outputHeads.ReshapeShared(outputHeads.Shape.NormalizeRank(2)); // Reshape to [B * T, H*d_k] for DenseLinear forward
        var output = output_proj.Forward(flatshape, ctx); // This should project back to [B * T, d_model]
        output = output.ReshapeShared(new Shape(B, T, output.Shape[^1])); // Reshape back to [B, T, d_model] by splitting first dimension back into 2

        return output;
    }

    public override Gradients ProjectionOutputGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        // By default, attention module does not include an output projection layer, so just return the gradient as is to be backpropagated into the attention heads
        var B = dY.Shape.Length(0);
        var T = dY.Shape.Length(1);
        var dyFlat = dY.ReshapeShared(dY.Shape.NormalizeRank(2)); // Reshape to [B * T, d_model] for DenseLinear backward
        var sub = (WeightAndBiasGradients)output_proj.Backward(dyFlat, ctx, clipping); 

        return new WeightAndBiasGradients(sub.dX.ReshapeShared(new Shape(B, T, sub.dX.Shape[^1])), sub.dW, sub.dB);// Reshape back to the expected shape for the next part of the backward algorithm, which is [B, T, H*d_k]
    }


    protected override void UpdateKey(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        k_proj.Update(learningRate, gradients, optimizer, regularization);
    }

    protected override void UpdateQuery(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        q_proj.Update(learningRate, gradients, optimizer, regularization);
    }

    protected override void UpdateValue(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        v_proj.Update(learningRate, gradients, optimizer, regularization);
    }

    public override void UpdateProjection(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        output_proj.Update(learningRate, gradients, optimizer, regularization);
    }
}