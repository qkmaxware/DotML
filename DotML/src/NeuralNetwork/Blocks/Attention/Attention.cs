using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public abstract class Attention : INetworkModule
{
    private readonly float scale;

    private readonly int heads;
    private readonly int d_model;
    private readonly int d_k;

    public int Heads => heads;
    public int ModelEmbeddingLength => d_model;
    public int OutputEmbeddingLength => d_k;

    /// <summary>
    /// Number of submodules contained within this module
    /// </summary>
    public int SubmoduleCount => 0; // Eh maybe make this more accurate... idk imma treat attention as a single layer for a while rather than a block

    public Attention(int d_model, int d_k, int heads = 1)
    {
        this.heads      = Math.Max(1, heads);
        this.d_model    = Math.Max(1, d_model);
        this.d_k        = Math.Max(1, d_k);
        this.scale      = 1.0f / MathF.Sqrt(this.d_k);
    }

    /// <summary>
    /// Compute the query (Q) tensor from the provided input.
    /// Expected shape: [B, H, T, d_k] (batch, heads, time, head-dim)
    /// </summary>
    /// <param name="input">input [B,T,d_model]</param>
    /// <returns>query tensor [B,H,T,d_k]</returns>
    protected abstract Tensor<float> ComputeQuery(Tensor<float> input, EvaluationContext? ctx);
    protected abstract Gradients ComputeQueryGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null);
    protected abstract void UpdateQuery(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null);

    /// <summary>
    /// Compute the key (K) tensor from the provided input.
    /// Expected shape: [B, H, T, d_k] (batch, heads, time, head-dim)
    /// </summary>
    /// <param name="input">input (B,T,d_model)</param>
    /// <returns>key tensor [B,H,T,d_k]</returns>
    protected abstract Tensor<float> ComputeKey(Tensor<float> input, EvaluationContext? ctx);
    protected abstract Gradients ComputeKeyGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null);
    protected abstract void UpdateKey(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null);

    /// <summary>
    /// Compute the value (V) tensor from the provided input.
    /// Expected shape: [B, H, T, d_k] (batch, heads, time, head-dim)
    /// </summary>
    /// <param name="input">input (B,T,d_model)</param>
    /// <returns>value tensor [B,H,T,d_k]</returns>
    protected abstract Tensor<float> ComputeValue(Tensor<float> input, EvaluationContext? ctx);
    protected abstract Gradients ComputeValueGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null);
    protected abstract void UpdateValue(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null);

    public abstract void Initialize(IInitializer initializer);

    public Shape ForwardShape(Shape input)
    {
        Shape normalized = input.NormalizeRank(3);
        if (normalized[2] != d_model)
            throw new ArgumentException("Input embedding length not compatible with model embedding length", nameof(input));
        // Output embedding length is heads * d_k (concatenated heads)
        return new Shape(normalized[0], normalized[1], heads * d_k);
    }

    public Tensor<float> Forward(Tensor<float> input, EvaluationContext? ctx = null, ISteppedProgress? progress = null)
    {
        // Compress leading dimensions or expand with dimensions of len 1 to reshape as rank 3 tensor
        var X = input.ReshapeShared(input.Shape.NormalizeRank(3));  // [B, T, d_model]

        // Compute Q, K, V
        // Expected shapes: Q,K,V => [B, H, T, d_k]
        var Q = ComputeQuery(X, ctx);                               // [B, H, T, d_k]
        var K = ComputeKey(X, ctx);                                 // [B, H, T, d_k]
        var V = ComputeValue(X, ctx);                               // [B, H, T, d_k]

        // Transpose last two dims of K -> [B, H, d_k, T]
        Tensor<float> K_T = K.MatrixTranspose();

        // Scores per head -> [B, H, T, T]
        var Scores = Q.BatchedMatMul(K_T);
        Scores.ScaleByInplace(scale);

        // Softmax across the key/time axis (last dim)
        var Attention = Scores.Softmax(dim: 3);                     // [B, H, T, T]

        // Attention applied to V -> [B, H, T, d_k]
        Tensor<float> OutputHeads = Attention.BatchedMatMul(V);     // [B, H, T, d_k]
        Console.WriteLine($"Output head shape: {OutputHeads.Shape}");

        // Concatenate heads -> [B, T, H, d_k] -> reshape to [B, T, H*d_k]
        var OutputHeadsTransposed = OutputHeads.Transpose(1, 2);         // [B, T, H, d_k]
        Tensor<float> OutputHeadsReshaped = OutputHeadsTransposed.ReshapeShared(new Shape(OutputHeadsTransposed.Shape.Length(0), OutputHeadsTransposed.Shape.Length(1), heads * d_k));

        Tensor<float> output = ProjectionOutput(OutputHeadsReshaped, ctx);

        if (ctx is not null)
        {
            ctx.Save(this, new AttentionContext(input: X, output: output, outputHeads: OutputHeads, query: Q, key: K, value: V, scores: Scores, attention: Attention));
        }

        progress?.Advance(steps: 1);
        return output;
    }

    public virtual Tensor<float> ProjectionOutput(Tensor<float> outputHeads, EvaluationContext? ctx)
    {
        // By default, attention module does not include an output projection layer, so just return the concatenated heads
        return outputHeads;
    }

    public virtual Gradients ProjectionOutputGradient(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        // By default, attention module does not include an output projection layer, so just return the gradient as is to be backpropagated into the attention heads
        return new Gradient(dY);
    }

    public virtual void UpdateProjection(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        // By default, attention module does not include an output projection layer, so nothing to update
    }

    public Gradients Backward(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        var context = ctx.Get<AttentionContext>(this);
        var proj_grad = ProjectionOutputGradient(dY, ctx, clipping); // backprop through output projection if it exists, otherwise just pass dY through
        dY = proj_grad.dX;
        // dY is gradient wrt concatenated output: [B, T, H*d_k]
        // reshape to per-head gradients: [B, T, H, d_k] -> transpose to [B, H, T, d_k]
        var dYReshaped = dY.ReshapeShared(new Shape(dY.Shape.Length(0), dY.Shape.Length(1), heads, d_k)); // [B, T, H, d_k]
        var dYHeads = dYReshaped.Transpose(1, 2); // [B, H, T, d_k]

        // dV = Attention^T * dYHeads  -> [B, H, T, d_k]
        var dV = context.Attention.MatrixTranspose().BatchedMatMul(dYHeads);

        // dAttention = dYHeads * V^T -> [B, H, T_q, T_kv]
        var dAttention = dYHeads.BatchedMatMul(context.Value.MatrixTranspose());

        Tensor<float> dScores = Tensor<float>.Defaults(context.Scores.Shape);
        int B = context.Scores.Shape.Length(0);
        int H = context.Scores.Shape.Length(1);
        int T_q = context.Scores.Shape.Length(2);
        int T_kv = context.Scores.Shape.Length(3);

        // Apply softmax Jacobian per head, per (b,h,t) row
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T_q; t++)
                {
                    var dScoreRow = new float[T_kv];
                    for (int i = 0; i < T_kv; i++)
                    {
                        float sum = 0f;
                        for (int j = 0; j < T_kv; j++)
                        {
                            float delta = (i == j) ? 1f : 0f;
                            sum += context.Attention[b, h, t, i] *
                                (delta - context.Attention[b, h, t, j]) *
                                dAttention[b, h, t, j];
                        }
                        dScoreRow[i] = sum;
                    }
                    for (int i = 0; i < T_kv; i++)
                        dScores[b, h, t, i] = dScoreRow[i];
                }
            }
        }
        dScores.ScaleByInplace(scale);

        var dQ = dScores.BatchedMatMul(context.Key);

        var dScores_T = dScores.MatrixTranspose(); // swap last 2 dims

        var dK = dScores_T.BatchedMatMul(context.Query);

        var grad_q = ComputeQueryGradient(dQ, ctx, clipping); // returns Gradients (clipped if need be)
        var grad_k = ComputeKeyGradient(dK, ctx, clipping);
        var grad_v = ComputeValueGradient(dV, ctx, clipping);

        // Merge dX = dX_q + dX_k + dX_v
        var dKV = grad_k.dX.AddWith(grad_v.dX);
        var dX = grad_q.dX.AddWith(dKV);

        var grads = new AttentionGradients(dX, dKV, grad_q, grad_k, grad_v, proj_grad);
        if (clipping is not null)
            grads.Clip(clipping);
        return grads;
    }

    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not AttentionGradients grad)
            throw new Exception("Expecting AttentionGradients");

        UpdateQuery(learningRate, grad.dQ, optimizer, regularization);
        UpdateKey(learningRate, grad.dK, optimizer, regularization);
        UpdateValue(learningRate, grad.dV, optimizer, regularization);

        UpdateProjection(learningRate, grad.dProj, optimizer, regularization);
    }
}


public class AttentionContext : IModuleContext
{
    public Tensor<float> Input { get; init; }
    public Tensor<float> Output { get; init; }
    public Tensor<float> OutputHeads { get; init; }

    public Tensor<float> Query { get; init; }
    public Tensor<float> Key { get; init; }
    public Tensor<float> Value { get; init; }

    public Tensor<float> Scores { get; init; }
    public Tensor<float> Attention { get; init; }

    public AttentionContext(Tensor<float> input, Tensor<float> output, Tensor<float> outputHeads, Tensor<float> query, Tensor<float> key, Tensor<float> value, Tensor<float> scores, Tensor<float> attention)
    {
        this.Input = input;
        this.Output = output;
        this.OutputHeads = outputHeads;

        this.Query = query;
        this.Key = key;
        this.Value = value;

        this.Scores = scores;
        this.Attention = attention;
    }
}

public class AttentionGradients : Gradients
{
    public Tensor<float> dKV { get; init; }
    public Gradients dProj { get; init; }
    public Gradients dQ { get; init; }
    public Gradients dK { get; init; } 
    public Gradients dV { get; init; }

    public AttentionGradients(Tensor<float> dX, Tensor<float> dKV, Gradients dQ, Gradients dK, Gradients dV, Gradients dProj) : base(dX)
    {
        this.dKV = dKV;
        this.dQ = dQ;
        this.dK = dK;
        this.dV = dV;
        this.dProj = dProj;
    }

    public override void Clip(ILocalClippingStrategy<float> clipping)
    {
        clipping.ClipInput(this.dX);
        if (!ReferenceEquals(dX, dKV))
        {
            clipping.ClipInput(this.dKV);
        }
        // dQ, dK, dV are already clipped by their layer's backward function before we get here so no need to clip again
    }

    public override IEnumerable<Tensor<float>> EnumerateParameterGradients()
    {
        yield return dKV;
        foreach (var t in dQ.EnumerateParameterGradients())
            yield return t;
        foreach (var t in dK.EnumerateParameterGradients())
            yield return t;
        foreach (var t in dV.EnumerateParameterGradients())
            yield return t;
    }
}
