using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Centers a tensor spatially in 2-dimensions either cropping or padding as necessary for centering to occur. Input Tensor must have the shape [..., Height, Width].
/// </summary>
public class Center2D : NetworkLayer
{
    /// <summary>
    /// Target number of rows/height of dimension ^2
    /// </summary>
    public int TargetRows { get; init; }
    /// <summary>
    /// Target number of columns/width of dimension ^1
    /// </summary>
    public int TargetColumns { get; init; }

    /// <summary>
    /// Create a new centering layer with the given target size
    /// </summary>
    /// <param name="rows">Target number of rows/height</param>
    /// <param name="columns">Target number of columns/width</param>
    public Center2D(int rows, int columns)
    {
        this.TargetRows = Math.Max(0, rows);
        this.TargetColumns = Math.Max(0, columns);
    }

    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);

    public override Shape ForwardShape(Shape input)
    {
        var dims = input.EnsureRank(2).AsDimensionSpan().ToArray();
        dims[^2] = TargetRows;
        dims[^1] = TargetColumns;
        return new Shape(dims);
    }

    public override Tensor<float> Forward(Tensor<float> channels)
    {
        var c = channels.ReshapeShared(channels.Shape.EnsureRank(2));
        var rows = c.Shape.Length(^2);
        var cols = c.Shape.Length(^1);
        if (rows == TargetRows && cols == TargetColumns)
            return c; // Shape was unaltered, no padding/cropping

        // Negative padding = crop (my .Pad method supports this)
        var deltaRows = TargetRows - rows;
        var deltaColumns = TargetColumns - cols;

        var padLeft = deltaColumns / 2;
        var padRight = deltaColumns - padLeft;

        var padTop = deltaRows / 2;
        var padBottom = deltaRows - padTop;

        return c.Pad(
            padValue: 0.0f,

            left: padLeft,
            right: padRight,
            top: padTop,
            bottom: padBottom
        );
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var c = x.ReshapeShared(x.Shape.EnsureRank(2));
        var rows = c.Shape.Length(^2);
        var cols = c.Shape.Length(^1);
        if (rows == TargetRows && cols == TargetColumns)
            return new Gradient(dy.ReshapeShared(x.Shape)); // Shape was unaltered, return dy

        // Compute what padding would have been applied
        var deltaRows = TargetRows - rows;
        var deltaColumns = TargetColumns - cols;

        var padLeft = deltaColumns / 2;
        var padRight = deltaColumns - padLeft;

        var padTop = deltaRows / 2;
        var padBottom = deltaRows - padTop;

        // Apply reverse padding/cropping to what would have been applied
        var dx = dy.Pad(
            padValue: 0.0f,

            left: -padLeft,
            right: -padRight,
            top: -padTop,
            bottom: -padBottom
        );

        return new Gradient(dx.ReshapeShared(x.Shape));
    }

    public override void Initialize(IInitializer initializer) { /* Nothing to initialize */ }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null) { /* Nothing to update */ }
}