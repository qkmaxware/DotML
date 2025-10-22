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
    /// Perform local clipping on these gradients
    /// </summary>
    /// <param name="clipping">clipping strategy</param>
    public abstract void Clip(ILocalClippingStrategy<float> clipping);

    /// <summary>
    /// Perform global clipping over all gradients encapsulated by this object
    /// </summary>
    /// <param name="clipping">clipping strategy</param>
    public void Clip(IGlobalClippingStrategy<float> clipping)
    {
        clipping.Clip(this.EnumerateParameterGradients());
    }

    /// <summary>
    /// Enumerate over all parameter gradients represented by this tensor (non-dX)
    /// </summary>
    /// <returns>enumerable of tensors</returns>
    public abstract IEnumerable<Tensor<float>> EnumerateParameterGradients();
}

/// <summary>
/// Simple container for storing input gradient
/// </summary>
public class Gradient : Gradients
{
    public Gradient(Tensor<float> dx) : base(dx) { }

    public override void Clip(ILocalClippingStrategy<float> clipping)
    {
        clipping.ClipInput(dX);
    }

    public override IEnumerable<Tensor<float>> EnumerateParameterGradients() => Enumerable.Empty<Tensor<float>>();
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

    public override void Clip(ILocalClippingStrategy<float> clipping)
    {
        clipping.ClipInput(dX);
        clipping.ClipWeight(dW);
        clipping.ClipBias(dX);
    }

    public override IEnumerable<Tensor<float>> EnumerateParameterGradients()
    {
        yield return dW;
        yield return dB;
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

    public override void Clip(ILocalClippingStrategy<float> clipping)
    {
        clipping.ClipInput(dX);
        // Don't clip subgradients as those should already be clipped by .Backward of other layers
    }
    
    public override IEnumerable<Tensor<float>> EnumerateParameterGradients()
    {
        foreach (var p in this.subgradients.SelectMany(sub => sub.EnumerateParameterGradients()))
            yield return p;
    }
}