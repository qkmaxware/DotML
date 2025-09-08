using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// General base class for the storage of gradients
/// </summary>
public abstract class Gradients
{
    /// <summary>
    /// gradient of loss w.r.t input
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
    /// gradient of loss w.r.t weights
    /// </summary>
    public Tensor<float> dW { get; init; }
    
    /// <summary>
    /// gradient of loss w.r.t biases
    /// </summary>
    public Tensor<float> dB { get; init; }

    public WeightAndBiasGradients(Tensor<float> dx, Tensor<float> dw, Tensor<float> db) : base(dx)
    {
        this.dW = dw;
        this.dB = db;
    }
}