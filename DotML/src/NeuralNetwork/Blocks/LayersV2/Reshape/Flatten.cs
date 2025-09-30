using System.Drawing;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class Flatten : Reshape
{
    // [N, C, H, W] --> [N, 1, F, 1]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TensorShape FlattenCHW2H(TensorShape x)
    {
        var xShape = x.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = 1;
        dims[^2] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);
        dims[^1] = 1;

        return new TensorShape(dims);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2H(Tensor<float> x)
    {
        return x.ReshapeShared(FlattenCHW2H(x.Shape));     // Flatten to column, reuse same data array
    }

    // [N, C, H, W] --> [N, F, 1, 1]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TensorShape FlattenCHW2C(TensorShape x)
    {
        var xShape = x.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);
        dims[^2] = 1;
        dims[^1] = 1;

        return new TensorShape(dims);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2C(Tensor<float> x)
    {
        return x.ReshapeShared(FlattenCHW2C(x.Shape));     // Flatten to column, reuse same data array
    }

    // [N, C, H, W] --> [N, 1, 1, F]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TensorShape FlattenCHW2W(TensorShape x)
    {
        var xShape = x.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = 1;
        dims[^2] = 1;
        dims[^1] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);

        return new TensorShape(dims);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2W(Tensor<float> x)
    {
        return x.ReshapeShared(FlattenCHW2W(x.Shape));     // Flatten to column, reuse same data array
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

    public override TensorShape ForwardShape(TensorShape input)
    {
        return Mode switch
        {
            FlatteningMode.Height => FlattenCHW2H(input),
            FlatteningMode.Channel => FlattenCHW2C(input),
            FlatteningMode.Width => FlattenCHW2W(input),
            _ => throw new InvalidOperationException()
        };
    }

    public override Tensor<float> Forward(Tensor<float> x)
    {
        return Mode switch
        {
            FlatteningMode.Height => FlattenCHW2H(x),
            FlatteningMode.Channel => FlattenCHW2C(x),
            FlatteningMode.Width => FlattenCHW2W(x),
            _ => throw new InvalidOperationException()
        };
    }

    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}