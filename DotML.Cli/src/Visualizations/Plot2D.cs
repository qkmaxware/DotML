using System.Drawing;
using System.Numerics;

namespace DotML.Cli.Visualizations;

public class Plot2D {

    private string? Title;    
    private Range Domain;
    private Range Range;

    public Plot2D(Range domain, Range range, string? title = null) {
        this.Title = title;
        this.Domain = domain;
        this.Range = range;
    }

    public void Draw(IEnumerable<Vector2> points) {
        // TODO make this more "dynamic"
        var width = 320;
        var height = 32;
        var empty_line = new string(' ', width);

        // Framework
        if (!string.IsNullOrEmpty(this.Title)) {
            var padding = (width - this.Title.Length) / 2;
            Console.Write(new string(' ', padding));
            Console.WriteLine(this.Title);
        }
        Console.WriteLine(this.Range.End);
        var chart_start = Console.GetCursorPosition();
        var vline = " |";
        for (var i = 0; i < height; i++) {
            Console.Write(vline);
            Console.WriteLine(empty_line);
        }
        var chart_end = Console.GetCursorPosition();
        var start_str = $"{this.Range.Start},{this.Domain.Start}";
        var end_str = this.Domain.End.ToString();
        Console.Write(' '); Console.WriteLine(new string('-', width));
        Console.Write(start_str);
        Console.Write(new string(' ', Math.Max(0, width - start_str.Length - end_str.Length))); 
        Console.WriteLine(end_str);

        // Points
        var done = Console.GetCursorPosition();
        var colour = Console.ForegroundColor;
        Vector2? last_point = null;
        Graphics graphics = new Graphics(
            top: chart_start.Top, 
            left: vline.Length,
            width: width,
            height: height
        );
        foreach (var point in points.OrderBy(x => x.X)) {
            if (!last_point.HasValue) {
                last_point = point;
                continue;
            }

            var x = (int)(point.X / width);
            var y = height - (int)(point.Y / height);

            var _x = (int)(last_point.Value.X / width);
            var _y = height - (int)(last_point.Value.Y / height);

            graphics.DrawLine((_x, _y), (x, y));
            last_point = point;
        }
        Console.SetCursorPosition(done.Left, done.Top);
        Console.ForegroundColor = colour;
    }

}