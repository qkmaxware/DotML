namespace DotML.Network;

public class AvgPool2D : LocalPooling2D
{
    public AvgPool2D(int size, int stride, int padding) : base(size, stride, padding) { }

    public AvgPool2D(int width, int height, int strideX, int strideY, int paddingX, int paddingY) : base(width, height, strideX, strideY, paddingX, paddingY) { }

    protected override float Accumulate(float current, float delta, int count)
    {
        return current + delta;             // Compute the sum
    }

    protected override float Aggregate(float current, int count)
    {
        return current / Math.Max(1, count); // Sum / count
    }

    protected override void Backpropagate(Span2D<float> dx, ReadOnlySpan2D<float> x, float dy, int items, int startX, int endX, int startY, int endY)
    {
        var inputHeight = x.Rows;
        var inputWidth = x.Columns;
        float errorContribution = dy / Math.Max(1, items); // Distribute the error
        for (int kr = startY; kr < endY; kr++) {
            if (kr < 0 || kr >= inputHeight)
                continue;

            for (int kc = startX; kc < endX; kc++) {
                if (kc < 0 || kc >= inputWidth)
                    continue;

                dx[kr, kc] += errorContribution;            // Assign the error contribution to each element in the pooling region
            }   
        }
    }
}