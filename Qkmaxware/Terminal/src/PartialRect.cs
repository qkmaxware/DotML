using System.Drawing;

namespace Qkmaxware.Terminal;

public readonly struct LayoutRect
{
    public readonly int X;
    public readonly int Y;
    public readonly int Width;

    public readonly Point TopLeft => new Point(X, Y);
    public readonly Point TopRight => new Point(X + Width, Y);

    public LayoutRect(int x, int y, int width)
    {
        this.X = x;
        this.Y = y;
        this.Width = width;
    }
}

public readonly struct LayoutSize
{
    public readonly int Width;
    public readonly int Height;

    public static LayoutSize Empty = new LayoutSize(0, 0);

    public LayoutSize(int width, int height)
    {
        this.Width = width;
        this.Height = height;
    }
}