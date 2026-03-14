using System.Collections;
using System.Drawing;
using System.Text;

namespace Qkmaxware.Terminal.Elements;

/// <summary>
/// A series of data in a 2D plot
/// </summary>
public class Series2D : IEnumerable<(float X, float Y)>
{
    public string? Name { get; set; }
    public char Symbol { get; set; } = '*';
    private SortedList<float, (float X, float Y)> points;

    public bool IsEnabled { get; set; } = true;

    public const char SymbolAsterisk = '*';
    public const char SymbolCircle = 'o';
    public const char SymbolPlus = '+';
    public const char SymbolAt = '@';
    public const char SymbolBullet = '•';
    public const char SymbolMiddleDot = '·';
    public const char SymbolSmallSquare = '▪';
    public const char SymbolTimes = 'x';
    public const char SymbolHash = '#';

    public (float From, float To) Range => (points.Min(pt => pt.Value.Y), points.Max(pt => pt.Value.Y));
    public (float From, float To) Domain => (points.Keys[0], points.Keys[points.Count - 1]);

    public Series2D(string? name, char? symbol = null)
    {
        this.Name = name;
        this.points = new SortedList<float, (float X, float Y)>();

        if (symbol.HasValue)
            this.Symbol = symbol.Value;
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

        return points.Where((pt) => pt.Value.Y >= yfrom && pt.Value.Y < yto && pt.Value.X >= xfrom && pt.Value.X < xto).Select(pt => pt.Value);
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
public class Plot2D : IElement
{
    public string? Title { get; set; }
    public string? YAxisLabel { get; set; }
    public string? XAxisLabel { get; set; }
    public int Height { get; set; } = 12;
    public List<Series2D>? Series { get; private set; } = new List<Series2D>();

    public Plot2D(string? title = null, int height = 12, IEnumerable<Series2D>? data = null, string? xLabel = null, string? yLabel = null)
    {
        this.Title = title;
        this.Height = Math.Max(0, height);
        this.YAxisLabel = yLabel;
        this.XAxisLabel = xLabel;
        if (data is not null)
            Series.AddRange(data);
    }

    public LayoutSize Render(Graphics graphics)
    {
        var Width = graphics.DrawingRegion.Width;
        var Height = this.Height;
        var consumed = 0;

        // Draw header
        /*
                         TITLE
        */
        if (!string.IsNullOrEmpty(this.Title)) {
            graphics.DrawClipped(new Point(0, 0), Center(this.Title, Width));
            consumed += 1;
        }

        // Draw points
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
            graphics.DrawClipped(new Point(0, 1), range.To.ToString("F1")); consumed++;
            var drawingWidth = Math.Max(0, Width - 3);
            var bucketHeight = rangeLength / Height;
            var bucketWidth = domainLength / drawingWidth;
            var yLabelLength = (YAxisLabel?.Length ?? 0);
            var yLabelVPadding = (Height - yLabelLength) / 2;
            var img_start = consumed;
            for (var vBucket = 0; vBucket < Height; vBucket++)
            {
                // Draw y-axis
                var line = img_start + vBucket;
                var yLabelCharacterIndex = -yLabelVPadding + vBucket;
                if (yLabelCharacterIndex >= 0 && yLabelCharacterIndex < yLabelLength && YAxisLabel is not null)
                {
                    // Label
                    graphics.Draw(new Point(0, line), YAxisLabel[yLabelCharacterIndex]);
                }
                graphics.Draw(new Point(2, line), '│');

                // Series data
                var yfrom = range.From + bucketHeight * vBucket;
                var yto = range.From + bucketHeight * (vBucket + 1);
                for (var xBucket = 0; xBucket < drawingWidth; xBucket++)
                {
                    var xfrom = domain.From + bucketWidth * xBucket;
                    var xto = domain.From + bucketWidth * (xBucket + 1);

                    // Get all points that fall in this domain and range slice
                    // This is probably not efficient, buuuuuuuut.
                    foreach (var pair in this.Series.Where(s => s.IsEnabled).SelectMany(s => s.PointsInRegion(yfrom, yto, xfrom, xto).Select(pt => (Series: s, Point: pt))))
                    {
                        // Write the first terminal symbol we encounter
                        graphics.Draw(new Point(3 + xBucket, line), pair.Series.Symbol);
                        break;
                    }
                }
                consumed++;
            }

            // Draw x-axis
            graphics.Draw(new Point(2, consumed), '└');
            graphics.DrawClipped(new Point(3, consumed), new string('─', drawingWidth));
            consumed++;

            // Draw x-axis labels
            graphics.DrawClipped(new Point(0, consumed), LeftCenterRight($"{domain.From:F1},{range.From:F1}", XAxisLabel, domain.To.ToString("F1"), Width));
            consumed++;

            // Draw legend
            /*
            + Sine Series, * Cosine Series
            */
            int seriesIndex = 0;
            consumed++; // Create a gap
            StringBuilder sb = new StringBuilder();
            foreach (var series in this.Series.Where(s => s.IsEnabled))
            {
                if (seriesIndex != 0)
                    sb.Append(", ");
                sb.Append(series.Symbol); sb.Append(' '); sb.Append(series.Name ?? $"Series {seriesIndex}");
                seriesIndex++;
            }
            var drawn = graphics.DrawWrapped(new Point(0, consumed), sb.ToString());
            consumed += drawn.Height;
        }
        
        return new LayoutSize(Width, consumed);
    }

    private string Center(object? obj, int Width)
    {
        var text = obj?.ToString() ?? string.Empty;

        if (text.Length >= Width) return text.Substring(0, Width);
        int pad = (Width - text.Length) / 2;
        return new string(' ', pad) + text;
    }

    private string LeftRight(object? left, object? right, int Width)
    {
        var halfWidth = Width / 2;
        StringBuilder sb = new StringBuilder();
        {
            var text = left?.ToString() ?? string.Empty;
            if (text.Length >= Width) return text.Substring(0, halfWidth);
            int pad = (Width - text.Length);
            sb.Append(text); sb.Append(new string(' ', pad));
        }
        {
            var text = right?.ToString() ?? string.Empty;
            if (text.Length >= Width) return text.Substring(0, halfWidth);
            int pad = (Width - text.Length);
            sb.Append(new string(' ', pad)); sb.Append(text);
        }
        return sb.ToString();
    }

    private string LeftCenterRight(object? left, object? center, object? right, int Width)
    {
        var halfWidth = Width / 3;
        StringBuilder sb = new StringBuilder();
        {
            var text = left?.ToString() ?? string.Empty;
            if (text.Length >= halfWidth) return text.Substring(0, halfWidth);
            int pad = (halfWidth - text.Length);
            sb.Append(text); sb.Append(new string(' ', pad));
        }
        {
            var text = center?.ToString() ?? string.Empty;
            if (text.Length >= halfWidth) return text.Substring(0, halfWidth);
            int pad = (halfWidth - text.Length) / 2;
            sb.Append(new string(' ', pad)); sb.Append(text); sb.Append(new string(' ', pad));
        }
        {
            var text = right?.ToString() ?? string.Empty;
            if (text.Length >= halfWidth) return text.Substring(0, halfWidth);
            int pad = (halfWidth - text.Length);
            sb.Append(new string(' ', pad)); sb.Append(text);
        }
        return sb.ToString();
    }
}