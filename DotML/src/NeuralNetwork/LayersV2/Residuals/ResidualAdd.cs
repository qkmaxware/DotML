using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Addition skip connection / residual addition
/// </summary>
public class ResidualAdd
{
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