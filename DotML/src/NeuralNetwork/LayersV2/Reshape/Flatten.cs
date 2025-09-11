using System.Drawing;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class Flatten : Reshape
{
    // [N, C, H, W] --> [N, 1, F, 1]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2H(Tensor<float> x)
    {
        var xShape = x.Shape.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = 1;
        dims[^2] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);
        dims[^1] = 1;

        return x.ReshapeShared(new TensorShape(dims));     // Flatten to column, reuse same data array
    }

    // [N, C, H, W] --> [N, F, 1, 1]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2C(Tensor<float> x)
    {
        var xShape = x.Shape.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);
        dims[^2] = 1;
        dims[^1] = 1;

        return x.ReshapeShared(new TensorShape(dims));     // Flatten to column, reuse same data array
    }

    // [N, C, H, W] --> [N, 1, 1, F]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2W(Tensor<float> x)
    {
        var xShape = x.Shape.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = 1;
        dims[^2] = 1;
        dims[^1] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);

        return x.ReshapeShared(new TensorShape(dims));     // Flatten to column, reuse same data array
    }

    public enum FlatteningMode
    {
        Height, Channel, Width
    }

    public FlatteningMode Mode { get; init; }

    public Flatten(FlatteningMode mode)
    {
        this.Mode = mode;
    }

    public Flatten() : this(FlatteningMode.Height) { }

    public override Tensor<float> Forward(Tensor<float> x)
    {
        return Mode switch {
            FlatteningMode.Height   => FlattenCHW2H(x),
            FlatteningMode.Channel  => FlattenCHW2C(x),
            FlatteningMode.Width    => FlattenCHW2W(x),
            _                       => throw new InvalidOperationException()
        };
    }
}