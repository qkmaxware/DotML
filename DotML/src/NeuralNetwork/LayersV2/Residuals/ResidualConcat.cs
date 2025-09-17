using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Concatenation skip connection / residual concatenation
/// </summary>
public class ResidualConcat
{
    public readonly Index Dimension;
    public ResidualConcat(Index dim)
    {
        this.Dimension = dim;
    }

    /// <summary>
    /// Concatenate along the batches dimension
    /// </summary>
    public ResidualConcat Batches() => new ResidualConcat(^4);

    /// <summary>
    /// Concatenate along the channels dimension
    /// </summary>
    public ResidualConcat Channels() => new ResidualConcat(^3); 

    /// <summary>
    /// Concatenate along the rows dimension
    /// </summary>
    public ResidualConcat Rows() => new ResidualConcat(^2); 

    /// <summary>
    /// Concatenate along the columns dimension
    /// </summary>
    public ResidualConcat Columns() => new ResidualConcat(^1); 

    public Tensor<float> Combine(params ReadOnlySpan<Tensor<float>> items)
    {
        if (items.Length == 0)
            throw new ArgumentException(nameof(items), "Must contain at least one item to combine");

        var result = items[0];
        for (var i = 1; i < items.Length; i++)
        {
            result = result.Concat(items[i], axis: Dimension);
        }

        return result;
    }
}