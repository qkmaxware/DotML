using System.Drawing;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// General base class for the storage of gradients
/// </summary>
public abstract class Gradients
{
    /// <summary>
    /// Gradient of loss w.r.t input
    /// </summary>
    public Tensor<float> dX { get; init; }

    public Gradients(Tensor<float> dx)
    {
        this.dX = dx;
    }

    /// <summary>
    /// Perform clipping on these gradients
    /// </summary>
    /// <param name="clipping">clipping strategy</param>
    public abstract void Clip(IClippingStrategy clipping);
}

/// <summary>
/// Simple container for storing input gradient
/// </summary>
public class Gradient : Gradients
{
    public Gradient(Tensor<float> dx) : base(dx) { }

    public override void Clip(IClippingStrategy clipping) {
        this.dX.ElementWiseInplace((x) => clipping.ClipInput(x));
    }
}

/// <summary>
/// Container for storing input gradient as well as weight and bias gradients
/// </summary>
public class WeightAndBiasGradients : Gradients
{
    /// <summary>
    /// Gradient of loss w.r.t weights
    /// </summary>
    public Tensor<float> dW { get; init; }

    /// <summary>
    /// Gradient of loss w.r.t biases
    /// </summary>
    public Tensor<float> dB { get; init; }

    public WeightAndBiasGradients(Tensor<float> dx, Tensor<float> dw, Tensor<float> db) : base(dx)
    {
        this.dW = dw;
        this.dB = db;
    }

    public override void Clip(IClippingStrategy clipping)
    {
        this.dX.ElementWiseInplace((x) => clipping.ClipInput(x));
        this.dW.ElementWiseInplace((w) => clipping.ClipWeight(w));
        this.dB.ElementWiseInplace((b) => clipping.ClipBias(b));
    }
}

/// <summary>
/// Container storing input gradient and a list of numbered component gradients
/// </summary>
public class GradientList : Gradient
{
    private Gradients[] subgradients;

    public GradientList(Tensor<float> dx, Gradients[] list) : base(dx)
    {
        this.subgradients = list;
    }

    /// <summary>
    /// gradient of loss w.r.t the n'th component
    /// </summary>
    /// <param name="n">component number</param>
    /// <returns>gradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Gradients dN(int n) => subgradients[n];

    public override void Clip(IClippingStrategy clipping) {
        this.dX.ElementWiseInplace((x) => clipping.ClipInput(x));
        // Don't clip subgradients as those should already be clipped by .Backward of other layers
    }
}