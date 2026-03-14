using System.Drawing;

namespace Qkmaxware.Terminal.Layout;

public class Panel: IElement
{
    public string? Title { get; set; }

    public IElement? ChildComponent { get; set; }

    public Panel(IElement? child)
    {
        this.ChildComponent = child;
    }
    public Panel(string title, IElement? child)
    {
        this.Title = title;
        this.ChildComponent = child;
    }

    const char TopLeft = '┌';
    const char TopMiddle = '─';
    const char TopRight = '┐';
    const char LeftMiddle = '│';
    const char RightMiddle = '│';
    const char BottomLeft = '└';
    const char BottomMiddle = '─';
    const char BottomRight = '┘';

    public LayoutSize Render(Graphics graphics)
    {
        if (graphics.DrawingRegion.Width < 2)
            return LayoutSize.Empty;

        // Take into account borders
        var innerWidth = graphics.DrawingRegion.Width - 2;
        var vOffset = 1;

        Graphics subgraphics = graphics.WithRegion(new LayoutRect(graphics.DrawingRegion.X + 1, graphics.DrawingRegion.Y + vOffset, innerWidth));
        var size = ChildComponent?.Render(subgraphics) ?? LayoutSize.Empty;

        // TODO Draw borders & Title 
        graphics.Draw(new Point(0, 0), TopLeft);
        graphics.Draw(new Point(graphics.DrawingRegion.Width - 1, 0), TopRight);
        var head = (this.Title ?? string.Empty).PadRight(innerWidth, TopMiddle);
        graphics.DrawClipped(new Point(1, 0), head);

        for (var i = 1; i < size.Height + 1; i++)
        {
            graphics.Draw(new Point(0, i), LeftMiddle);
            graphics.Draw(new Point(graphics.DrawingRegion.Width - 1, i), RightMiddle);
        }

        graphics.Draw(new Point(0, size.Height + 1), BottomLeft);
        graphics.Draw(new Point(graphics.DrawingRegion.Width - 1, size.Height + 1), BottomRight);
        graphics.DrawClipped(new Point(1, size.Height + 1), new string(BottomMiddle, innerWidth));

        return new LayoutSize(graphics.DrawingRegion.Width, size.Height + 2);
    }

    public Panel WithPadding(int horizontal = 0, int vertical = 0) => WithPadding(horizontal, vertical, horizontal, vertical);
    public Panel WithPadding(int pad = 0) => WithPadding(pad, pad, pad, pad);
    public Panel WithPadding(int left = 0, int top = 0, int right = 0, int bottom = 0)
    {
        this.ChildComponent = new Padding(this.ChildComponent, top: top, bottom: bottom, left: left, right: right);
        return this;
    }
}