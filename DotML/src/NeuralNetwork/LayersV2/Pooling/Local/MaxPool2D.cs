namespace DotML.Network;

public class MaxPool2D : LocalPooling2D
{
    public MaxPool2D(int size, int stride, int padding) : base(size, stride, padding) { }

    public MaxPool2D(int width, int height, int strideX, int strideY, int paddingX, int paddingY) : base(width, height, strideX, strideY, paddingX, paddingY) { }

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

    protected override void Backpropagate(Span2D<float> dx, ReadOnlySpan2D<float> x, float dy, int items, int startX, int endX, int startY, int endY)
    {
        var inputHeight = x.Rows;
        var inputWidth = x.Columns;
        int maxRow = startY, maxCol = startX; float maxVal = float.MinValue; // Values for max pooling
        for (int kr = startY; kr < endY; kr++) {
            if (kr < 0 || kr >= inputHeight)
                continue;

            for (int kc = startX; kc < endX; kc++) {
                if (kc < 0 || kc >= inputWidth)
                    continue;
                var value = x[kr, kc];

                // Compute; Assume max pooling (avg is different)
                if (value > maxVal) {
                    maxVal = value;
                    maxRow = kr;
                    maxCol = kc;
                }
            }
        }
        if (maxRow < 0 || maxRow >= inputHeight || maxCol < 0 || maxCol >= inputWidth)
            return;
        dx[maxRow, maxCol] += dy; 
    }
}