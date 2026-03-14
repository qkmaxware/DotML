using System.Drawing;

namespace Qkmaxware.Terminal.Elements;

public abstract class Drawing : IElement
{
    public abstract LayoutSize Render(Graphics graphics);
}

public class ShapeDrawing : Drawing
{
    public Shape? Shape { get; set; }

    public ShapeDrawing() { }
    
    public ShapeDrawing(Shape? shape)
    {
        this.Shape = shape;
    }

    public override LayoutSize Render(Graphics graphics)
    {
        if (Shape is null)
            return LayoutSize.Empty;

        var bounds = this.Shape.GetSize();
        graphics.DrawShape(new Point(0, 0), this.Shape);

        return new LayoutSize(Math.Min(graphics.DrawingRegion.Width, bounds.Width), bounds.Height);
    }
}