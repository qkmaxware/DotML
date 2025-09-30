using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Concatenation skip connection / residual concatenation
/// </summary>
public class ResidualConcat: ResidualBlock
{
    /// <summary>
    /// The concatenation dimension
    /// </summary>
    public readonly Index Dimension;

    /// <summary>
    /// Concatenate along the batches dimension
    /// </summary>
    public ResidualConcat Batches(INetworkModule gx, INetworkModule? residual = null) => new ResidualConcat(^4, gx, residual);

    /// <summary>
    /// Concatenate along the channels dimension
    /// </summary>
    public ResidualConcat Channels(INetworkModule gx, INetworkModule? residual = null) => new ResidualConcat(^3, gx, residual); 

    /// <summary>
    /// Concatenate along the rows dimension
    /// </summary>
    public ResidualConcat Rows(INetworkModule gx, INetworkModule? residual = null) => new ResidualConcat(^2, gx, residual); 

    /// <summary>
    /// Concatenate along the columns dimension
    /// </summary>
    public ResidualConcat Columns(INetworkModule gx, INetworkModule? residual = null) => new ResidualConcat(^1, gx, residual); 

    public ResidualConcat(Index axis, INetworkModule gx, INetworkModule? residual = null) : base(gx, residual)
    {
        this.Dimension = axis;
    }

    public ResidualConcat(INetworkModule gx, INetworkModule? residual = null) : this(^1, gx, residual) { }

    public override TensorShape ForwardShape(TensorShape input)
    {
        var mainShape = MainPath.ForwardShape(input);
        var resShape = ResidualPath?.ForwardShape(input) ?? input;

        // See Tensor.Concat
        // Compute the minimum rank for concatenation
        var matchedRank = Math.Max(mainShape.Rank, resShape.Rank);
        var dim = Dimension.GetOffset(matchedRank);
        if (dim < 0)
        {
            matchedRank = matchedRank + Math.Abs(dim);
            dim = Dimension.GetOffset(matchedRank);
        }

        // Force both tensors to be treated as if they are of the matched rank by padding with leading '1' if rank is to small
        var a_shape = mainShape.EnsureRank(matchedRank);

        var b_shape = resShape.EnsureRank(matchedRank);

        // Validate the dimensions for compatibility
        for (int i = 0; i < matchedRank; i++)
        {
            if (i != dim && a_shape.Length(i) != b_shape.Length(i))
                throw new ArgumentException("Tensors must have the same shape on all axes except the concatenation axis.");
        }

        // Determine output shape
        var outDims = new int[matchedRank];
        for (int i = 0; i < matchedRank; i++)
        {
            if (i != dim)
                outDims[i] = a_shape.Length(i);
            else
                outDims[i] = a_shape.Length(i) + b_shape.Length(i);
        }
        return new TensorShape(outDims);
    }

    protected override Tensor<float> Combine(Tensor<float> output, Tensor<float> residual)
    {
        var result = output.Concat(residual, Dimension);

        return result;
    }

    protected override (Tensor<float> dM, Tensor<float> dR) SplitGradient(Tensor<float> dY, TensorShape mainOutputShape, TensorShape residualOutputShape)
    {
        var len_main = mainOutputShape.Length(Dimension);
        var len_residual = residualOutputShape.Length(Dimension);

        var dY_main = dY.SliceAlong(Dimension, 0..len_main);
        var dY_residual = dY.SliceAlong(Dimension, len_main..(len_main + len_residual));

        return (dY_main, dY_residual);
    }

    protected override ResidualBlockGradients CombineGradients(Gradients main, Gradients residual)
    {
        var dX = main.dX + residual.dX;

        return new ResidualBlockGradients(dX, main, residual);
    }
}