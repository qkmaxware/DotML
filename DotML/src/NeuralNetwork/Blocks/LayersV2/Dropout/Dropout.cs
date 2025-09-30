using System;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer which performs dropout
/// <see href="https://en.wikipedia.org/wiki/Dilution_(neural_networks)"/>
/// </summary>
public class Dropout : NetworkLayer
{
    public float DropoutRate { get; init; }
    public float KeepRate => 1 - DropoutRate;

    public Dropout() : this(0.1f) { }

    public Dropout(float dropoutRate)
    {
        this.DropoutRate = Math.Clamp(dropoutRate, 0.0f, 1.0f);
    }

    public override void Initialize(IInitializer initializer) { }

    public override Tensor<float> Forward(Tensor<float> input)
    {
        return input; // No dropout at inference
    }

    public Tensor<float> Forward(Tensor<float> input, out Tensor<float> mask)
    {
        mask = Tensor<float>.Mask(input.Shape, DropoutRate);

        // Elementwise multiply input by mask
        return input.HadamardWith(mask);
    }

    public override TensorShape ForwardShape(TensorShape input) => input;

    public override Tensor<float> Forward(Tensor<float> channels, EvaluationContext? ctx)
    {
        var res = this.Forward(channels, out var mask);
        if (ctx is not null)
        {
            ctx.Save(this, new MaskContext(channels, res, mask!));
        }
        return res;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy) => new Gradient(dy);

    public Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy, Tensor<float> mask)
    {
        if (mask is null)
            return new Gradient(dy);

        // Elementwise multiply gradient by mask
        var dx = dy.HadamardWith(mask);
        return new Gradient(dy);
    }

    public override Gradients Backward(Tensor<float> dy, EvaluationContext ctx, IClippingStrategy? clipping = null)
    {
        var io = ctx.Get<MaskContext>(this);
        var grads = this.Backward(io.Input, io.Output, dy, io.Mask);
        if (clipping is not null)
            grads.Clip(clipping);
        return grads;
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null) { /* Nothing to do here */ }

    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}