namespace Qkmaxware.Terminal.Layout;

public class MaxWidthContainer : IElement
{
    public int MaxWidth { get; init; }
    public IElement? ChildComponent { get; set; }

    public MaxWidthContainer(int maxWidth, IElement? child)
    {
        this.MaxWidth = Math.Max(0, maxWidth);
        this.ChildComponent = child;
    }

    public LayoutSize Render(Graphics graphics)
    {
        var drawRegion = graphics;
        if (graphics.DrawingRegion.Width > MaxWidth)
        {
            drawRegion = graphics.WithRegion(
                new LayoutRect(
                    graphics.DrawingRegion.X,
                    graphics.DrawingRegion.Y,
                    MaxWidth
                )
            );
        }

        return ChildComponent?.Render(drawRegion) ?? LayoutSize.Empty;
    }
}

public class VSplitContainer : IElement
{
    public float? SplitPercent { get; init; }
    public int? SplitWidth { get; init; }

    public IElement? LeftChildComponent { get; set; }
    public IElement? RightChildComponent { get; set; }

    public VSplitContainer(float lhsPercent, IElement? lhs, IElement? rhs)
    {
        this.SplitPercent = Math.Clamp(lhsPercent, 0f, 1f);
        this.LeftChildComponent = lhs;
        this.RightChildComponent = rhs;
    }

    public VSplitContainer(int lhsWidth, IElement? lhs, IElement? rhs)
    {
        this.SplitWidth = Math.Max(0, lhsWidth);
        this.LeftChildComponent = lhs;
        this.RightChildComponent = rhs;
    }

    public LayoutSize Render(Graphics graphics)
    {
        // Compute 2 regions based on percent or width
        var lhsWidth = Math.Clamp(this.SplitWidth.HasValue
            ? this.SplitWidth.Value
            : (this.SplitPercent.HasValue
                ? (int)(graphics.DrawingRegion.Width * this.SplitPercent.Value)
                : graphics.DrawingRegion.Width / 2
            ),
            0,
            graphics.DrawingRegion.Width
        );
        var rhsWidth = Math.Clamp(graphics.DrawingRegion.Width - lhsWidth, 0, graphics.DrawingRegion.Width);

        var lhsRegion = graphics.WithRegion(
            new LayoutRect(
                graphics.DrawingRegion.X,
                graphics.DrawingRegion.Y,
                lhsWidth
            )
        );
        var rhsRegion = graphics.WithRegion(
            new LayoutRect(
                graphics.DrawingRegion.X + lhsWidth,
                graphics.DrawingRegion.Y,
                rhsWidth
            )
        );

        var lhsUsed = this.LeftChildComponent?.Render(rhsRegion) ?? LayoutSize.Empty;
        var rhsUsed = this.RightChildComponent?.Render(lhsRegion) ?? LayoutSize.Empty;

        return new LayoutSize(
            width: lhsUsed.Width + rhsUsed.Width,
            height: Math.Max(lhsUsed.Height, rhsUsed.Height)
        );
    }
}