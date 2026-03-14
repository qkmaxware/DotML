using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class GlobalAvgPool2D
: GlobalPooling2D
{

    protected override float Accumulate(float current, float delta, int count)
    {
        return current + delta;             // Compute the sum
    }

    protected override float Aggregate(float current, int count)
    {
        return current / Math.Max(1, count); // Sum / count
    }

    protected override void SpreadGradient(ReadOnlySpan<float> x, Span<float> dx, float dy, int batch, int channel)
    {
        float spread = dy / x.Length;
        for (var i = 0; i < x.Length; i++)
        {
            dx[i] = spread;
        }
    }


    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg)
    {
        return visitor.Visit(this, arg);
    }

}