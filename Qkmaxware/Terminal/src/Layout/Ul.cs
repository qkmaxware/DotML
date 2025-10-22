namespace Qkmaxware.Terminal.Layout;

public class UnorderedList: IElement
{
    private List<IElement> elements = new();

    public UnorderedList() { }

    public UnorderedList(params IEnumerable<IElement> elements)
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
                graphics.DrawingRegion.X + 2,
                graphics.DrawingRegion.Y + consumedHeight, // Offset by how much we've vertically eaten up
                graphics.DrawingRegion.Width
            );

            var size = element.Render(graphics.WithRegion(subregion));
            if (size.Height > 0)
                graphics.Draw(new System.Drawing.Point(0, consumedHeight), '•');
                
            consumedWidth = Math.Max(consumedWidth, size.Width);
            consumedHeight += size.Height;

            index++;
        }

        return new LayoutSize(consumedWidth, consumedHeight);
    }
}