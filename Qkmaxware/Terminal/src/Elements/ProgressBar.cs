using System.Drawing;
using System.Text;

namespace Qkmaxware.Terminal.Elements;

public abstract class ProgressBar : IElement
{

    public char EmptyCharacter = DefaultEmptyCharacter;
    public char FilledCharacter = DefaultFilledCharacter;
    public char StartCharacter = DefaultStartCharacter;
    public char EndCharacter = DefaultEndCharacter;

    public ProgressBar() { }

    const char DefaultEmptyCharacter = '-';
    const char DefaultFilledCharacter = '█';
    const char DefaultStartCharacter = '|';
    const char DefaultEndCharacter = '|';

    public abstract float GetNormalizedPercent();

    public LayoutSize Render(Graphics graphics)
    {
        if (graphics.DrawingRegion.Width < 3)
        {
            return LayoutSize.Empty;
        }

        var percent = Math.Clamp(GetNormalizedPercent(), 0f, 1f);

        graphics.Draw(new Point(0, 0), StartCharacter);
        graphics.Draw(new Point(graphics.DrawingRegion.Width - 1, 0), EndCharacter);

        var innerWidth = graphics.DrawingRegion.Width - 2;
        var filledInt = Math.Round(percent * innerWidth);

        for (var i = 0; i < graphics.DrawingRegion.Width - 2; i++)
        {
            graphics.Draw(new Point(1 + i, 0), i < filledInt ? FilledCharacter : EmptyCharacter);
        }

        return new LayoutSize(graphics.DrawingRegion.Width, 1);
    }

    public static string ToString(float percent, int width)
    {
        if (width < 3)
        {
            return string.Empty;
        }

        percent = Math.Clamp(percent, 0f, 1f);
        var innerWidth = width - 2;
        var filledInt = Math.Round(percent * innerWidth);

        StringBuilder sb = new StringBuilder();
        sb.Append(DefaultStartCharacter);
        for (var i = 0; i < innerWidth; i++)
        {
            sb.Append(i < filledInt ? DefaultFilledCharacter : DefaultEmptyCharacter);
        }
        sb.Append(DefaultEndCharacter);

        return sb.ToString();
    }
}

public class StaticProgressBar : ProgressBar
{
    public float Percent { get; init; }

    public StaticProgressBar(float percent)
    {
        this.Percent = percent;
    }

    public override float GetNormalizedPercent()
    {
        return Percent;
    }
}

public class DynamicProgressBar : ProgressBar
{
    public Func<float> Source { get; init; }

    public DynamicProgressBar(Func<float> source)
    {
        this.Source = source;
    }

    public override float GetNormalizedPercent()
    {
        return Source();
    }
}