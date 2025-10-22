using System.Drawing;

namespace Qkmaxware.Terminal.Elements;

public class Label : IElement
{
    public string? Text { get; set; }

    public Func<string>? TextGenerator { get; set; }

    public Label(string? text)
    {
        this.Text = text;
    }

    public Label(Func<string> text)
    {
        this.TextGenerator = text;
    }

    public LayoutSize Render(Graphics graphics)
    {
        return graphics.DrawClipped(new Point(0, 0), Text ?? TextGenerator?.Invoke() ?? string.Empty);
    }
}