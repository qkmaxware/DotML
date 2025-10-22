using System.Drawing;

namespace Qkmaxware.Terminal.Elements;

public class RTC : IElement
{
    private int hh;
    private int mm;
    private int ss;
    private string? text;

    public RTC() { }

    public LayoutSize Render(Graphics graphics)
    {
        var now = DateTime.Now;
        if (text is null || (now.Hour != hh || now.Minute != mm || now.Second == ss))
        {
            text = DateTime.Now.ToString("HH:mm:ss");
            hh = now.Hour;
            mm = now.Minute;
            ss = now.Second;
        }
        return graphics.DrawClipped(new Point(0, 0), text);
    }
}