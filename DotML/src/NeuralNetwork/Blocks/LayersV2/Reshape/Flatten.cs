using System.Drawing;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class Flatten : Reshape
{
    // [N, C, H, W] --> [N, 1, F, 1]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Shape FlattenCHW2H(Shape x)
    {
        var xShape = x.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = 1;
        dims[^2] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);
        dims[^1] = 1;

        return new Shape(dims);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2H(Tensor<float> x)
    {
        return x.ReshapeShared(FlattenCHW2H(x.Shape));     // Flatten to column, reuse same data array
    }

    // [N, C, H, W] --> [N, F, 1, 1]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Shape FlattenCHW2C(Shape x)
    {
        var xShape = x.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);
        dims[^2] = 1;
        dims[^1] = 1;

        return new Shape(dims);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2C(Tensor<float> x)
    {
        return x.ReshapeShared(FlattenCHW2C(x.Shape));     // Flatten to column, reuse same data array
    }

    // [N, C, H, W] --> [N, 1, 1, F]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Shape FlattenCHW2W(Shape x)
    {
        var xShape = x.EnsureRank(3);             // Rank < 3, pad with 1's
        var dims = xShape.AsDimensionSpan().ToArray();  // Copy batch dimensions
        dims[^3] = 1;
        dims[^2] = 1;
        dims[^1] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);

        return new Shape(dims);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenCHW2W(Tensor<float> x)
    {
        return x.ReshapeShared(FlattenCHW2W(x.Shape));     // Flatten to column, reuse same data array
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Shape FlattenNonBatch(Shape x)
    {
        if (x.Rank < 2)
            return x; // If only 1 dimension just return as is
        
        // If >= 2 dimensions first dimension is batch the rest get flattened
        var ys = new int[2];
        ys[0] = x[0];
        ys[1] = x.Length(1..);

        return new Shape(ys);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> FlattenNonBatch(Tensor<float> x)
    {
        return x.ReshapeShared(FlattenNonBatch(x.Shape));     // Flatten to column, reuse same data array
    }

    public enum FlatteningMode
    {
        CHW2Height, CHW2Channel, CHW2Width, CollapseNonBatch
    }

    public FlatteningMode Mode { get; init; }

    public Flatten(FlatteningMode mode)
    {
        this.Mode = mode;
    }

    public Flatten() : this(FlatteningMode.CollapseNonBatch) { }

    public override Shape ForwardShape(Shape input)
    {
        return Mode switch
        {
            FlatteningMode.CHW2Height => FlattenCHW2H(input),
            FlatteningMode.CHW2Channel => FlattenCHW2C(input),
            FlatteningMode.CHW2Width => FlattenCHW2W(input),
            FlatteningMode.CollapseNonBatch => FlattenNonBatch(input),
            _ => throw new InvalidOperationException()
        };
    }

    public override Tensor<float> Forward(Tensor<float> x)
    {
        return Mode switch
        {
            FlatteningMode.CHW2Height => FlattenCHW2H(x),
            FlatteningMode.CHW2Channel => FlattenCHW2C(x),
            FlatteningMode.CHW2Width => FlattenCHW2W(x),
            FlatteningMode.CollapseNonBatch => FlattenNonBatch(x),
            _ => throw new InvalidOperationException()
        };
    }

    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}