using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class GlobalMaxPool2D
: GlobalPooling2D
{

    protected override float Accumulate(float current, float delta, int count)
    {
        if (count == 1)
            return delta;                   // First accumulated value
        return Math.Max(current, delta);    // Subsequent accumulated values
    }

    protected override float Aggregate(float current, int count)
    {
        return current;                    // Current is the max
    }

    protected override void SpreadGradient(ReadOnlySpan<float> x, Span<float> dx, float dy, int batch, int channel)
    {
        // Find index of max
        int maxIndex = 0; float maxValue = 0;
        for (int i = 0; i < x.Length; i++)
        {
            var x_i = x[i];
            if (i == 0 || x_i > maxValue)
            {
                maxValue = x_i;
                maxIndex = i;
            }
        }
        float spread = dy;
        dx[maxIndex] = spread;
    }


    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg)
    {
        return visitor.Visit(this, arg);
    }

}