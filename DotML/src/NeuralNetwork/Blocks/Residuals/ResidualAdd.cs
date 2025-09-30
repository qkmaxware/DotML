using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Addition skip connection / residual addition
/// </summary>
public class ResidualAdd: ResidualBlock
{
    public ResidualAdd(INetworkModule gx, INetworkModule? residual = null) : base(gx, residual) { }

    public override TensorShape ForwardShape(TensorShape input) => MainPath.ForwardShape(input);

    protected override Tensor<float> Combine(Tensor<float> output, Tensor<float> residual)
    {
        var result = output + residual;

        return result;
    }

    protected override (Tensor<float> dM, Tensor<float> dR) SplitGradient(Tensor<float> dY, TensorShape mainOutputShape, TensorShape residualOutputShape) => (dY, dY);

    protected override ResidualBlockGradients CombineGradients(Gradients main, Gradients residual)
    {
        var dX = main.dX + residual.dX;

        return new ResidualBlockGradients(dX, main, residual);
    }

    public Tensor<float> Combine(params ReadOnlySpan<Tensor<float>> items)
    {
        if (items.Length == 0)
            throw new ArgumentException(nameof(items), "Must contain at least one item to combine");

        var result = items[0];
        for (var i = 1; i < items.Length; i++)
            result = result.AddWith(items[i]);

        return result;
    }
}