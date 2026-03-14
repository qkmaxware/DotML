using System.Drawing;

namespace Qkmaxware.Terminal.Elements;

public class HorizontalRule : IElement
{
    public LayoutSize Render(Graphics graphics)
    {
        var width = graphics.DrawingRegion.Width;
        if (width < 3)
            return new LayoutSize(graphics.DrawingRegion.Width, 1);

        var str_width = width - 2;

        return graphics.DrawClipped(new Point(1, 0), new string('─', str_width));
    }
}