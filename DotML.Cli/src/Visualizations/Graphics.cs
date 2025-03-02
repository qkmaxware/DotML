using System.Drawing;
using System.Numerics;

namespace DotML.Cli.Visualizations;

public class Graphics {

    private int top;
    private int bottom;
    private int left; 
    private int right;

    public int Width => right - left;
    public int Height => bottom - top;

    public Graphics(int width, int height) : this(Console.GetCursorPosition().Top, Console.GetCursorPosition().Left, width, height) {}

    public Graphics(int top, int left, int width, int height) {
        this.top = top;
        this.left = left;
        this.bottom = top + height;
        this.right = left + width;
    }

    private static char Filled = '*';
    private static char Empty = ' ';

    public void Clear() {
        for (var r = 0; r < Height; r++) {
            for (var c = 0; c < Width; c++) {
                SetPixel(c, r, false);
            }
        }
    }

    public void ClearRect(int x, int y, int width, int height) {
        for (var r = 0; r < height; r++) {
            for (var c = 0; c < width; c++) {
                SetPixel(c + x, r + y, false);
            }
        }
    }
    public void ClearRect(Rectangle rect) => ClearRect(rect.X, rect.Y, rect.Width, rect.Height);
    
    public void SetPixel(int x, int y, ConsoleColor colour) {
        if (x >= 0 && y >= 0 && x < Width && y < Height) {
            Console.SetCursorPosition(left + x, top + y);
            var prev = Console.ForegroundColor;
            Console.ForegroundColor = colour;
            Console.Write(Filled);
            Console.ForegroundColor = prev;
        }
    }
    public void SetPixel(Point p, ConsoleColor colour) => SetPixel(p.X, p.Y, colour);

    public void SetPixel(int x, int y, bool filled) {
        if (x >= 0 && y >= 0 && x < Width && y < Height) {
            Console.SetCursorPosition(left + x, top + y);
            Console.Write(filled ? Filled : Empty);
        }
    }
    public void SetPixel(Point p, bool filled) => SetPixel(p.X, p.Y, filled);

    // TODO more drawing methods like DrawCircle, DrawOval, DrawPoint, etc etc
    public void DrawLine((int X, int Y) from, (int X, int Y) to) => DrawLine(new Point(from.X, from.Y), new Point(to.X, to.Y));
    public void DrawLine(Point from, Point to) {
        // Bresenham's line algorithm
        var dx = Math.Abs(to.X - from.X);
        var sx = from.X < to.X ? 1 : -1;
        var dy = -Math.Abs(to.Y - from.Y);
        var sy = from.Y < to.Y ? 1 : -1;
        var error = dx + dy;

        var x = from.X;
        var y = from.Y;

        while (true) {
            // Plot
            SetPixel(x, y, true);
            // ----

            var e2 = 2 * error;
            if (e2 >= dy) {
                if (x == to.X)
                    break;
                error = error + dy;
                x = x + sx;
            }
            if (e2 <= dx) {
                if (y == to.Y)
                    break;
                error = error + dx;
                y = y + sy;
            }
        }
    }
}