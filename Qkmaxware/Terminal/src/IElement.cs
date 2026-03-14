using System.Drawing;
using Qkmaxware.Terminal.Layout;

namespace Qkmaxware.Terminal;

public interface IElement
{
    public LayoutSize Render(Graphics graphics);
}

public static class IElementExtensions
{
    public static IElement WithMargin(this IElement? element, int pad = 0)
        => new Padding(element, top: pad, bottom: pad, left: pad, right: pad);
    public static IElement WithMargin(this IElement? element, int horizontal = 0, int vertical = 0)
        => new Padding(element, top: vertical, bottom: vertical, left: horizontal, right: horizontal);
    public static IElement WithMargin(this IElement? element, int left = 0, int top = 0, int right = 0, int bottom = 0)
        => new Padding(element, top: top, bottom: bottom, left: left, right: right);
}