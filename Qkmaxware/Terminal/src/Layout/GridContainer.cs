namespace Qkmaxware.Terminal.Layout;

public class GridContainer : IElement
{
    public int Columns { get; init; }
    private List<IElement> elements = new();

    public GridContainer(int columns)
    {
        this.Columns = Math.Max(1, columns);
    }

    public GridContainer(int columns, params IEnumerable<IElement> elements)
    {
        this.Columns = Math.Max(1, columns);
        this.elements.AddRange(elements);
    }

    public void Add(IElement element) => this.elements.Add(element);

    public void AddRange(IEnumerable<IElement> elements) => this.elements.AddRange(elements);

    public LayoutSize Render(Graphics graphics)
    {
        var columnWidth = graphics.DrawingRegion.Width / this.Columns;

        var consumedWidth = graphics.DrawingRegion.Width;
        var consumedHeight = 0;
        var consumedHeightOnRow = 0;
        foreach (var (child, index) in this.elements.Select((x, ind) => (x, ind)))
        {
            var column = index % this.Columns;
            if (column == 0)
            {
                consumedHeight += consumedHeightOnRow;
                consumedHeightOnRow = 0;
            }
            var region = new LayoutRect(
                graphics.DrawingRegion.X + columnWidth * Columns,
                graphics.DrawingRegion.Y + consumedHeight,
                columnWidth
            );
            var size = child?.Render(graphics.WithRegion(region)) ?? LayoutSize.Empty;
            consumedHeightOnRow = Math.Max(consumedHeightOnRow, size.Height);
        }
        consumedHeight += consumedHeightOnRow;

        return new LayoutSize(consumedWidth, consumedHeight);
    }
}