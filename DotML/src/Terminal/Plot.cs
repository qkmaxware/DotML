using System.Collections;
using System.Drawing;
using System.Numerics;
using System.Reflection.PortableExecutable;
using System.Text;

namespace DotML.Terminal;

/// <summary>
/// An object that can be rendered to a text writer
/// </summary>
public interface ITextualRenderer
{
    public void Draw(TextWriter terminal);
}

/// <summary>
/// An object that can be rendered to the terminal
/// </summary>
public interface ITerminalRenderer : ITextualRenderer
{
    public void DrawToStdOut() => Draw(System.Console.Out);
}

/// <summary>
/// A series of data in a 2D plot
/// </summary>
public class Series2D : IEnumerable<(float X, float Y)>
{
    public string? Name { get; set; }
    public char Symbol { get; set; } = '*';
    private SortedList<float, (float X, float Y)> points;

    public bool IsEnabled { get; set; } = true;

    public (float From, float To) Range => (points.Values[0].Y, points.Values[points.Count - 1].Y);
    public (float From, float To) Domain => (points.Keys[0], points.Keys[points.Count - 1]);

    public Series2D(string? name)
    {
        this.Name = name;
        this.points = new SortedList<float, (float X, float Y)>();
    }

    public float this[float x]
    {
        get
        {
            if (points.Count == 0)
                return 0.0f;
            if (points.Count == 1)
                return points.Values[0].Y;

            var smallest = points.Values[0];
            var largest = points.Values[points.Count - 1];
            if (x < smallest.X)
                return smallest.Y;
            if (x > largest.X)
                return largest.Y;

            // Find bounding points
            var lower = smallest; var upper = largest;
            foreach (var pt in points)
            {
                if (x >= pt.Value.X)
                {
                    lower = pt.Value;
                }
                else
                {
                    upper = pt.Value;
                    break;
                }
            }

            // Linearly interpolate between them
            var t = (upper.X - x) / (upper.X - lower.X);
            return (1.0f - t) * lower.Y + t * upper.Y;
        }
    }

    public void Clear() => points.Clear();

    public IEnumerable<(float X, float Y)> PointsInRange(float from, float to)
    {
        if (from > to)
        {
            // If from > to swap the 2 so that from is < to
            (from, to) = (to, from);
        }

        return points.Where((pt) => pt.Value.Y >= from && pt.Value.Y < to).Select(pt => pt.Value);
    }

    public IEnumerable<(float X, float Y)> PointsInDomain(float from, float to)
    {
        if (from > to)
        {
            // If from > to swap the 2 so that from is < to
            (from, to) = (to, from);
        }

        return points.Where((pt) => pt.Value.X >= from && pt.Value.X < to).Select(pt => pt.Value);
    }

    public IEnumerable<(float X, float Y)> PointsInRegion(float yfrom, float yto, float xfrom, float xto)
    {
        if (yfrom > yto)
        {
            // If from > to swap the 2 so that from is < to
            (yfrom, yto) = (yto, yfrom);
        }
        if (xfrom > xto)
        {
            // If from > to swap the 2 so that from is < to
            (xfrom, xto) = (xto, xfrom);
        }

        return points.Where((pt) => pt.Value.X >= yfrom && pt.Value.X < yto && pt.Value.X >= xfrom && pt.Value.X < xto).Select(pt => pt.Value);
    }

    public void Add(float x, float y)
    {
        this.points.Add(x, (x, y));
    }

    public IEnumerator<(float X, float Y)> GetEnumerator()
    {
        foreach (var p in this.points)
            yield return p.Value;
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}

/// <summary>
/// A 2D plot of several series of data on the terminal
/// </summary>
public class Plot2D : ITerminalRenderer
{
    public string? Title { get; set; }
    public string? YAxisLabel { get; set; }
    public string? XAxisLabel { get; set; }
    public Size Size { get; set; } = new Size(80, 80);
    public List<Series2D>? Series { get; private set; } = new List<Series2D>();

    public Plot2D(string? title = null)
    {
        this.Title = title;
    }

    public void Draw(TextWriter terminal)
    {
        // Draw header
        /*
                         TITLE
        */
        if (!string.IsNullOrEmpty(this.Title))
            terminal.WriteLine(Center(this.Title));

        if (this.Series is not null && this.Series.Count > 0)
        {
            // Compute overall domain and range
            var range = this.Series[0].Range;
            var domain = this.Series[0].Domain;
            foreach (var series in this.Series.Skip(1))
            {
                var r = series.Range;
                range.From = float.Min(range.From, r.From);
                range.To = float.Max(range.To, r.To);

                var d = series.Domain;
                domain.From = float.Min(domain.From, d.From);
                domain.To = float.Max(domain.To, d.To);
            }
            var rangeLength = range.To - range.From;
            var domainLength = domain.To - domain.From;

            // Draw body
            /*
            ========================================
            12.5
              |
            H |
            E |
            A |
            D |
              |
              +-----------------------------------
            0.0, 0.0          FOOTER            1.2
            =========================================
            */
            terminal.WriteLine(new string('=', Size.Width));
            terminal.WriteLine(range.To);
            var drawingRegion = new Size(Size.Width - 3, Size.Height - 3);
            var bucketHeight = drawingRegion.Height / rangeLength;
            var bucketWidth = drawingRegion.Width / domainLength;
            var yLabelLength = (YAxisLabel?.Length ?? 0);
            var yLabelVPadding = (drawingRegion.Height - yLabelLength) / 2;
            for (var vBucket = 0; vBucket < drawingRegion.Height; vBucket++)
            {
                // Y-Axis
                var yLabelCharacterIndex = -yLabelVPadding + vBucket;
                if (yLabelCharacterIndex >= 0 && yLabelCharacterIndex < yLabelLength && YAxisLabel is not null)
                {
                    terminal.Write(YAxisLabel[yLabelCharacterIndex]);
                }
                else
                {
                    terminal.Write(' ');
                }
                terminal.Write(' ');
                terminal.Write('|');

                // Series data
                var yfrom = range.From + bucketHeight * vBucket;
                var yto = range.From + bucketHeight * (vBucket + 1);
                for (var xBucket = 0; xBucket < drawingRegion.Width; xBucket++)
                {
                    var xfrom = domain.From + bucketWidth * xBucket;
                    var xto = domain.From + bucketHeight * (xBucket + 1);

                    bool wrote = false;
                    // Get all points that fall in this domain and range slice
                    // This is probably not efficient, buuuuuuuut.
                    foreach (var pair in this.Series.Where(s => s.IsEnabled).SelectMany(s => s.PointsInRegion(yfrom, yto, xfrom, xto).Select(pt => (Series: s, Point: pt))))
                    {
                        // Write the first terminal symbol we encounter
                        terminal.Write(pair.Series.Symbol);
                        wrote = true;
                        break;
                    }
                    if (!wrote)
                    {
                        // Nothing written, just put a space
                        terminal.Write(' ');
                    }
                }

                terminal.WriteLine();
            }
            terminal.Write("  +"); terminal.Write(new string('-', drawingRegion.Width));
            terminal.WriteLine(LeftCenterRight($"{range.From},{domain.From}", XAxisLabel, domain.To));
            terminal.WriteLine(new string('=', Size.Width));

            // Draw legend
            /*
            + Sine Series, * Cosine Series
            */
            foreach (var series in this.Series.Where(s => s.IsEnabled))
            {
                terminal.Write(series.Symbol); terminal.Write(' '); terminal.Write(series.Name); terminal.Write(", ");
            }
        }
    }

    private string Center(object? obj)
    {
        var text = obj?.ToString() ?? string.Empty;

        if (text.Length >= Size.Width) return text.Substring(0, Size.Width);
        int pad = (Size.Width - text.Length) / 2;
        return new string(' ', pad) + text;
    }

    private string LeftRight(object? left, object? right)
    {
        var halfWidth = Size.Width / 2;
        StringBuilder sb = new StringBuilder();
        {
            var text = left?.ToString() ?? string.Empty;
            if (text.Length >= Size.Width) return text.Substring(0, halfWidth);
            int pad = (Size.Width - text.Length);
            sb.Append(text); sb.Append(new string(' ', pad));
        }
        {
            var text = right?.ToString() ?? string.Empty;
            if (text.Length >= Size.Width) return text.Substring(0, halfWidth);
            int pad = (Size.Width - text.Length);
            sb.Append(new string(' ', pad)); sb.Append(text);
        }
        return sb.ToString();
    }

    private string LeftCenterRight(object? left, object? center, object? right)
    {
        var halfWidth = Size.Width / 3;
        StringBuilder sb = new StringBuilder();
        {
            var text = left?.ToString() ?? string.Empty;
            if (text.Length >= Size.Width) return text.Substring(0, halfWidth);
            int pad = (Size.Width - text.Length);
            sb.Append(text); sb.Append(new string(' ', pad));
        }
        {
            var text = left?.ToString() ?? string.Empty;
            if (text.Length >= Size.Width) return text.Substring(0, halfWidth);
            int pad = (Size.Width - text.Length) / 2;
            sb.Append(new string(' ', pad)); sb.Append(text); sb.Append(new string(' ', pad));
        }
        {
            var text = right?.ToString() ?? string.Empty;
            if (text.Length >= Size.Width) return text.Substring(0, halfWidth);
            int pad = (Size.Width - text.Length);
            sb.Append(new string(' ', pad)); sb.Append(text);
        }
        return sb.ToString();
    }
}