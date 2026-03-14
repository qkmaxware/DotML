using System.Drawing;

namespace Qkmaxware.Terminal.Elements;

public class Paragraph : IElement
{
    public string? Text { get; set; }
    public Func<string>? TextGenerator { get; set; }

    public Paragraph(string? text)
    {
        this.Text = text;
    }

    public Paragraph(Func<string> text)
    {
        this.TextGenerator = text;
    }

    public LayoutSize Render(Graphics graphics)
    {
        return graphics.DrawWrapped(new Point(0, 0), Text ?? TextGenerator?.Invoke() ?? string.Empty);
    }
}