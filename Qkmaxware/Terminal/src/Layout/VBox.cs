namespace Qkmaxware.Terminal.Layout;

public class VBox: IElement
{
    private List<IElement> elements = new();

    public VBox() { }

    public VBox(params IEnumerable<IElement> elements)
    {
        this.elements.AddRange(elements);
    }

    public void Add(IElement element) => this.elements.Add(element);

    public void AddRange(IEnumerable<IElement> elements) => this.elements.AddRange(elements);

    public LayoutSize Render(Graphics graphics) {
        var index = 0;
        var consumedWidth = 0;
        var consumedHeight = 0;
        foreach (var element in this.elements)
        {
            var subregion = new LayoutRect(
                graphics.DrawingRegion.X,
                graphics.DrawingRegion.Y + consumedHeight, // Offset by how much we've vertically eaten up
                graphics.DrawingRegion.Width
            );
            
            var size = element.Render(graphics.WithRegion(subregion));
            consumedWidth = Math.Max(consumedWidth, size.Width);
            consumedHeight += size.Height;

            index++;
        }

        return new LayoutSize(consumedWidth, consumedHeight);
    }
}