using System.Drawing;

namespace Qkmaxware.Terminal;

public struct Graphics
{
    private readonly CharBuffer buffer;
    public readonly LayoutRect DrawingRegion;

    public int Width => DrawingRegion.Width;

    public Graphics(CharBuffer buffer, LayoutRect region)
    {
        this.buffer = buffer;
        this.DrawingRegion = region;
    }

    public Graphics WithRegion(LayoutRect rect)
    {
        return new Graphics(this.buffer, rect);
    }

    public void DrawShape(Point at, Shape shape, char filled = '█')
    {
        var size = shape.GetSize();
        var min = at;
        var max = new Point(at.X + size.Width, at.Y + size.Height);
        buffer.EnsureHeight(DrawingRegion.Y + max.Y);

        for (var r = 0; r < size.Height; r++)
        {
            var realY = DrawingRegion.Y + min.Y + r;
            for (var c = 0; c < size.Width; c++)
            {
                var realX = DrawingRegion.X + min.X + c;
                if (realX >= buffer.Width)
                    continue;

                if (!shape.IsFilled(new Point(c, r)))
                    continue;

                buffer[realX, realY].Character = filled;
            }
        }
    }

    public LayoutSize Draw(Point at, char c)
    {
        var realX = DrawingRegion.X + at.X;
        var realY = DrawingRegion.Y + at.Y;

        buffer.EnsureHeight(realY);
        if (realX >= buffer.Width)
            return LayoutSize.Empty;

        buffer[realX, realY].Character = c;
        return new LayoutSize(1, 1);
    }

    public LayoutSize DrawClipped(Point at, string str)
    {
        var realX = DrawingRegion.X + at.X;
        var realY = DrawingRegion.Y + at.Y;

        buffer.EnsureHeight(realY);
        var len = Math.Min(Math.Max(0, DrawingRegion.Width - at.X), str.Length); // Clip string to desired length
        for (var i = 0; i < len; i++)
        {
            buffer[realX + i, realY].Character = str[i];
        }
        return new LayoutSize(len, 1);
    }

    public LayoutSize DrawWrapped(Point at, string str)
    {
        int maxWidth = DrawingRegion.Width;
        int x = at.X;
        int y = at.Y;
        int totalHeight = 0;

        int i = 0;
        while (i < str.Length)
        {
            int remaining = str.Length - i;
            int lineLength = Math.Min(maxWidth - x, remaining);

            // If there's no space to draw on the current line, move to next line
            if (lineLength <= 0)
            {
                x = 0;
                y++;
                totalHeight++;
                continue;
            }

            string line = str.Substring(i, lineLength);

            // Draw the line clipped to the drawing region
            DrawClipped(new Point(x, y), line);

            i += lineLength;

            // Prepare for next line
            x = 0;
            y++;
            totalHeight++;
        }

        return new LayoutSize(DrawingRegion.Width, totalHeight);
    }

}