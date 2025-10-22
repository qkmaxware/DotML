using System.Drawing;

namespace Qkmaxware.Terminal.Layout;

public class Padding : IElement
{
    private int _Top { get; set; }
    private int _Bottom { get; set; }
    private int _Left { get; set; }
    private int _Right { get; set; }

    public int PadTop
    {
        get => _Top;
        set => _Top = Math.Max(0, value);
    }
    public int PadBottom {
        get => _Bottom;
        set => _Bottom = Math.Max(0, value);
    }
    public int PadLeft {
        get => _Left;
        set => _Left = Math.Max(0, value);
    }
    public int PadRight {
        get => _Right;
        set => _Right = Math.Max(0, value);
    }

    public IElement? ChildComponent { get; set; }

    public Padding(IElement? child, int top = 0, int bottom = 0, int left = 0, int right = 0)
    {
        this.ChildComponent = child;
        this.PadTop = top;
        this.PadBottom = bottom;
        this.PadLeft = left;
        this.PadRight = right;
    }

    public LayoutSize Render(Graphics graphics)
    {
        var width = graphics.DrawingRegion.Width;
        var innerWidth = Math.Max(0, width - (PadLeft + PadRight));

        var innerRegion = new LayoutRect(
            x: graphics.DrawingRegion.X + PadLeft,
            y: graphics.DrawingRegion.Y + PadTop,
            width: innerWidth
        );

        var size = this.ChildComponent?.Render(graphics.WithRegion(innerRegion)) ?? LayoutSize.Empty;

        return new LayoutSize(width, size.Height + (PadTop + PadBottom));
    }
}