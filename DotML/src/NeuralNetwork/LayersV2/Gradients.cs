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
}

/// <summary>
/// Simple container for storing input gradient
/// </summary>
public class Gradient : Gradients
{
    public Gradient(Tensor<float> dx) : base(dx) { }
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
}