namespace DotML.Network;

public class MinPool2D : LocalPooling2D
{
    public MinPool2D(int size, int stride, int padding) : base(size, stride, padding) { }

    public MinPool2D(int width, int height, int strideX, int strideY, int paddingX, int paddingY) : base(width, height, strideX, strideY, paddingX, paddingY) { }

    protected override float Accumulate(float current, float delta, int count)
    {
        if (count == 1)
            return delta;                   // First accumulated value
        return Math.Min(current, delta);    // Subsequent accumulated values
    }

    protected override float Aggregate(float current, int count)
    {
         return current;                    // Current is the min
    }

    protected override void Backpropagate(Span2D<float> dx, ReadOnlySpan2D<float> x, float dy, int items, int startX, int endX, int startY, int endY)
    {
        var inputHeight = x.Rows;
        var inputWidth = x.Columns;
        int minRow = startY, minCol = startX; float minValue = float.MaxValue; // Values for max pooling
        for (int kr = startY; kr < endY; kr++) {
            if (kr < 0 || kr >= inputHeight)
                continue;

            for (int kc = startX; kc < endX; kc++) {
                if (kc < 0 || kc >= inputWidth)
                    continue;
                var value = x[kr, kc];

                // Compute; Assume max pooling (avg is different)
                if (value < minValue) {
                    minValue = value;
                    minRow = kr;
                    minCol = kc;
                }
            }
        }
        if (minRow < 0 || minRow >= inputHeight || minCol < 0 || minCol >= inputWidth)
            return;
        dx[minRow, minCol] += dy; 
    }
}