using System.Runtime.CompilerServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Reshape and reorder the data from [N,C,H,W] format to [B,HxW,D] format
/// </summary>
public class Nchw2Btd : Reshape
{
    public override Shape ForwardShape(Shape input)
    {
        input = input.NormalizeRank(4);
        int B = input.Length(0);
        int C = input.Length(1);
        int H = input.Length(2);
        int W = input.Length(3);
        return new Shape(B, H * W, C);
    }

    public override Tensor<float> Forward(Tensor<float> x) => Forward(x, null);

    public override Tensor<float> Forward(Tensor<float> x, EvaluationContext? ctx = null)
    {
        // Expecting [B, C, H, W]
        var shape = x.Shape.NormalizeRank(4);
        int B = shape.Length(0);
        int C = shape.Length(1);
        int H = shape.Length(2);
        int W = shape.Length(3);

        // Output: [B, H*W, C]
        var output = Tensor<float>.Zeros(new Shape(B, H * W, C));

        for (int b = 0; b < B; b++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        int t = h * W + w;
                        output[b, t, c] = x[b, c, h, w];
                    }
                }
            }
        }

        if (ctx is not null)
            ctx.Save(this, new IOContext(x, output));

        return output;
    }

    public override Gradients Backward(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        var context = ctx.Get<IOContext>(this);

        var shape = context.Input.Shape.NormalizeRank(4);
        int B = shape.Length(0);
        int C = shape.Length(1);
        int H = shape.Length(2);
        int W = shape.Length(3);

        var dX = Tensor<float>.Zeros(shape); // [B, C, H, W]

        for (int b = 0; b < B; b++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        int t = h * W + w;
                        dX[b, c, h, w] = dY[b, t, c];
                    }
                }
            }
        }

        return new Gradient(dX.ReshapeShared(context.Input.Shape)); // If the original shape had more than 4 dims, B is a combination of all of them, revert that to the original ...B dimensions
    }

    public override void Initialize(IInitializer initializer) { }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null) { }
}

/// <summary>
/// Reshape and reorder the data from [B,HxW,D] format to [N,C,H,W] format
/// </summary>
public class Btd2Nchw : Reshape
{
    public int Rows { get; init; }
    public int Columns { get; init; }

    public Btd2Nchw(int rows, int columns)
    {
        this.Rows = rows;
        this.Columns = columns;
    }

    public override void Initialize(IInitializer initializer) { }

    public override Shape ForwardShape(Shape input)
    {
        input = input.NormalizeRank(3); // [B, T, C]
        int B = input.Length(0);
        int T = input.Length(1);
        int C = input.Length(2);

        if (T != (Rows * Columns))
            throw new Exception($"Shape mismatch, the size of the T ({T}) must be the product of the rows ({Rows}) and columns ({Columns})");
        return new Shape(B, C, Rows, Columns);
    }

    public override Tensor<float> Forward(Tensor<float> x) => Forward(x, null);

    public override Tensor<float> Forward(Tensor<float> x, EvaluationContext? ctx)
    {
        // x: [B, T, C]
        var shape = x.Shape.NormalizeRank(3);
        int B = shape.Length(0);
        int T = shape.Length(1);
        int C = shape.Length(2);

        // Retrieve H, W from context if available (or infer)
        if (T != (Rows * Columns))
            throw new Exception($"Shape mismatch, the size of the T ({T}) must be the product of the rows ({Rows}) and columns ({Columns})");
        Shape targetShape = new Shape(B, C, Rows, Columns);

        var output = Tensor<float>.Zeros(targetShape); // [B, C, H, W]

        for (int b = 0; b < B; b++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < Rows; h++)
                {
                    for (int w = 0; w < Columns; w++)
                    {
                        int t = h * Columns + w;
                        output[b, c, h, w] = x[b, t, c];
                    }
                }
            }
        }

        if (ctx is not null)
            ctx.Save(this, new IOContext(x, output));

        return output;
    }

    public override Gradients Backward(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        var context = ctx.Get<IOContext>(this);

        var shape = context.Input.Shape.NormalizeRank(3); // [B, T, C]
        int B = shape.Length(0);
        int T = shape.Length(1);
        int C = shape.Length(2);

        int H = context.Output.Shape.Length(^2);
        int W = context.Output.Shape.Length(^1);

        var dX = Tensor<float>.Zeros(shape); // [B, T, C]

        for (int b = 0; b < B; b++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        int t = h * W + w;
                        dX[b, t, c] = dY[b, c, h, w];
                    }
                }
            }
        }

        return new Gradient(dX);
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null) { }
}
