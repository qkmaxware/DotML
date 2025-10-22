using System.Drawing;

namespace Qkmaxware.Terminal;

public abstract class Shape
{
    public abstract LayoutSize GetSize();
    public abstract bool IsFilled(Point inner);
}

public class Square : Shape
{
    public int Width { get; private set; }
    public int Height { get; private set; }

    public Square(int width, int height)
    {
        this.Width = Math.Max(0, width);
        this.Height = Math.Max(0, height);
    }

    public override LayoutSize GetSize()
    {
        return new LayoutSize(this.Width, this.Height);
    }

    public override bool IsFilled(Point inner)
    {
        return inner.X >= 0 && inner.X < this.Width && inner.Y >= 0 && inner.Y < this.Height;
    }
}

public class Ellipse : Shape
{

    public int BoundingWidth { get; private set; }
    public int BoundingHeight { get; private set; }

    public Ellipse(int diameterX, int diameterY)
    {
        this.BoundingWidth = Math.Max(0, diameterX);
        this.BoundingHeight = Math.Max(0, diameterY);
    }

    public override LayoutSize GetSize()
    {
        return new LayoutSize(BoundingWidth, BoundingHeight);
    }

    public override bool IsFilled(Point inner)
    {
        // Center of the ellipse
        double cx = BoundingWidth / 2.0;
        double cy = BoundingHeight / 2.0;

        // Radii
        double rx = BoundingWidth / 2.0;
        double ry = BoundingHeight / 2.0;

        // Ellipse equation
        double dx = inner.X - cx;
        double dy = inner.Y - cy;

        double value = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
        
        return value <= 1.0;
    }
}