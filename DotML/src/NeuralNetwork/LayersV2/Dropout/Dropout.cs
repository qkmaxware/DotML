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

    public TensorShape InputShape { get; }
    public TensorShape OutputShape { get; }

    private Tensor<float>? mask;

    public Dropout(TensorShape inputShape) : this(inputShape, 0.1f) { }

    public Dropout(TensorShape inputShape, float dropoutRate)
    {
        this.InputShape = inputShape;
        this.OutputShape = inputShape;
        this.DropoutRate = Math.Clamp(dropoutRate, 0.0f, 1.0f);
    }

    public override void Initialize(IInitializer initializer) { }

    protected override void OnTrainingBegin() {
        RegenerateMask();
    }

    protected override void OnTrainingEnd() {
        ClearMask();
    }

    public void ClearMask()
    {
        this.mask = null;
    }

    public void RegenerateMask()
    {
        this.mask = Tensor<float>.Mask(this.InputShape, DropoutRate);
    }

    public override Tensor<float> Forward(Tensor<float> input)
    {
        if (IsInference)
            return input; // No dropout at inference

        if (mask is null)
            RegenerateMask();

        // Elementwise multiply input by mask
        return input.HadamardWith(mask!);
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        if (mask is null)
            return new Gradient(dy);

        // Elementwise multiply gradient by mask
        var dx = dy.HadamardWith(mask);
        return new Gradient(dx);
    }

    public override void SubtractGradients(Gradients grads) { }
}