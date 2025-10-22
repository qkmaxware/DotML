namespace Qkmaxware.Terminal.Layout;

public class Columns: IElement
{
    private List<IElement> elements = new();

    public Columns() { }

    public Columns(params IEnumerable<IElement> elements)
    {
        this.elements.AddRange(elements);
    }

    public void Add(IElement element) => this.elements.Add(element);

    public void AddRange(IEnumerable<IElement> elements) => this.elements.AddRange(elements);

    public LayoutSize Render(Graphics graphics) {
        var subregionWidth = graphics.DrawingRegion.Width / elements.Count;

        var index = 0;
        var consumedWidth = graphics.DrawingRegion.Width;
        var consumedHeight = 0;
        foreach (var element in this.elements)
        {
            var subregion = new LayoutRect(
                graphics.DrawingRegion.X + index * subregionWidth, // Offset by column number
                graphics.DrawingRegion.Y,
                subregionWidth
            );

            var size = element.Render(graphics.WithRegion(subregion));
            consumedHeight = Math.Max(consumedHeight, size.Height);

            index++;
        }

        return new LayoutSize(consumedWidth, consumedHeight);
    }
}