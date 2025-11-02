namespace Qkmaxware.Terminal.Layout;

/// <summary>
/// A simple container designed for use within a flexbox layout system
/// </summary>
public class FlexContainer : IElement
{
    public float FlexGrow { get; init; } = 1.0f;
    public float FlexShrink { get; init; } = 1.0f;
    public float FlexBasis { get; init; } = 0.0f;

    public IElement? ChildComponent { get; set; }

    public FlexContainer(IElement? child)
    {
        this.ChildComponent = child;
    }

    public static FlexContainer Fixed(int width, IElement? child)
    {
        return new FlexContainer(child)
        {
            FlexGrow = 0.0f,
            FlexShrink = 0.0f,
            FlexBasis = width
        };
    }

    public static FlexContainer Flexible(IElement? child)
    {
        return new FlexContainer(child)
        {
            FlexGrow = 1.0f,
            FlexShrink = 1.0f,
            FlexBasis = 0.0f
        };
    }

    public LayoutSize Render(Graphics graphics)
    {
        return ChildComponent?.Render(graphics) ?? LayoutSize.Empty;
    }
}

public enum FlexWrap
{
    NoWrap,
    Wrap
}

public enum JustifyContent
{
    FlexStart,
    FlexEnd,
    Center,
}

public class FlexRowBox : IElement
{
    public FlexWrap FlexWrap { get; set; } = FlexWrap.NoWrap;
    public JustifyContent JustifyContent { get; set; } = JustifyContent.FlexStart;  
    
    private List<FlexContainer> elements = new();

    public FlexRowBox() { }

    public FlexRowBox(params IEnumerable<FlexContainer> elements)
    {
        this.elements.AddRange(elements);
    }

    public void Add(FlexContainer element) => this.elements.Add(element);

    public void AddRange(IEnumerable<FlexContainer> elements) => this.elements.AddRange(elements);

    public LayoutSize Render(Graphics graphics)
    {
        var availableWidth = graphics.Width;

        // Compute total base width (sum of all FlexBasis values)
        float totalBasis = elements.Sum(e => e.FlexBasis);
        float totalGrow = elements.Sum(e => e.FlexGrow);
        float totalShrink = elements.Sum(e => e.FlexShrink);

        // Determine if we have free space or overflow
        float remaining = availableWidth - totalBasis;

        // Calculate each element's target width
        var computedWidths = new List<int>(elements.Count);
        if (remaining > 0 && totalGrow > 0)
        {
            // We have extra space — distribute by FlexGrow
            foreach (var e in elements)
            {
                float extra = (e.FlexGrow / totalGrow) * remaining;
                computedWidths.Add((int)Math.Round(e.FlexBasis + extra));
            }
        }
        else if (remaining < 0 && totalShrink > 0)
        {
            // We have overflow — shrink by FlexShrink
            float overflow = -remaining;
            foreach (var e in elements)
            {
                float reduction = (e.FlexShrink / totalShrink) * overflow;
                computedWidths.Add((int)Math.Max(0, Math.Round(e.FlexBasis - reduction)));
            }
        }
        else
        {
            // Perfect fit or no flexing
            foreach (var e in elements)
                computedWidths.Add((int)Math.Round(e.FlexBasis));
        }

        // Optional line wrapping
        var lines = new List<List<(FlexContainer element, int width)>>();
        var currentLine = new List<(FlexContainer, int)>();
        int currentLineWidth = 0;

        for (int i = 0; i < elements.Count; i++)
        {
            var elem = elements[i];
            int elemWidth = computedWidths[i];

            if (FlexWrap == FlexWrap.Wrap && currentLineWidth + elemWidth > availableWidth && currentLine.Count > 0)
            {
                // Move to next line
                lines.Add(currentLine);
                currentLine = new List<(FlexContainer, int)>();
                currentLineWidth = 0;
            }

            currentLine.Add((elem, elemWidth));
            currentLineWidth += elemWidth;
        }

        if (currentLine.Count > 0)
            lines.Add(currentLine);

        // Render lines
        int yOffset = 0;
        int totalHeight = 0;

        foreach (var line in lines)
        {
            int lineWidth = line.Sum(t => t.width);
            int xOffset = 0;

            // Compute horizontal offset based on JustifyContent
            switch (JustifyContent)
            {
                case JustifyContent.Center:
                    xOffset = (availableWidth - lineWidth) / 2;
                    break;
                case JustifyContent.FlexEnd:
                    xOffset = availableWidth - lineWidth;
                    break;
                case JustifyContent.FlexStart:
                default:
                    xOffset = 0;
                    break;
            }

            int lineHeight = 0;
            foreach (var (element, width) in line)
            {
                var subGraphics = graphics.WithRegion(new LayoutRect(
                    graphics.DrawingRegion.X + xOffset,
                    graphics.DrawingRegion.Y + yOffset,
                    width
                ));
                var size = element.Render(subGraphics);
                xOffset += width;
                lineHeight = Math.Max(lineHeight, size.Height);
            }

            yOffset += lineHeight;
            totalHeight += lineHeight;
        }

        return new LayoutSize(availableWidth, totalHeight);
    }
}