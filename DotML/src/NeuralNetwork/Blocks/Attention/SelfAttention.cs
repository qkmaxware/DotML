using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class SelfAttention : Attention
{
    private readonly DenseLinear q_proj;
    private readonly DenseLinear k_proj;
    private readonly DenseLinear v_proj;

    public SelfAttention(int d_model, int d_k, int heads = 1) : base(d_model, d_k, heads)
    {
        q_proj = new DenseLinear(d_model, d_k);
        k_proj = new DenseLinear(d_model, d_k);
        v_proj = new DenseLinear(d_model, d_k);
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
        var base_k = k_proj.Forward(input, ctx, null); // [B, T, d_k]

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

        var dySum = Tensor<float>.Defaults(new Shape(B, T, DK));
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
        var sub = k_proj.Backward(dySum, ctx, clipping);
        return sub;
    }

    protected override Tensor<float> ComputeQuery(Tensor<float> input, EvaluationContext? ctx)
    {
        // Project input -> [B, T, d_k] and replicate per-head -> [B, H, T, d_k]
        var base_q = q_proj.Forward(input, ctx, null); // [B, T, d_k]

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

        var dySum = Tensor<float>.Defaults(new Shape(B, T, DK));
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

        var sub = q_proj.Backward(dySum, ctx, clipping);
        return sub;
    }

    protected override Tensor<float> ComputeValue(Tensor<float> input, EvaluationContext? ctx)
    {
        // Project input -> [B, T, d_k] and replicate per-head -> [B, H, T, d_k]
        var base_v = v_proj.Forward(input, ctx, null); // [B, T, d_k]

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

        var dySum = Tensor<float>.Defaults(new Shape(B, T, DK));
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

        var sub = v_proj.Backward(dySum, ctx, clipping);
        return sub;
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
}