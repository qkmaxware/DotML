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
    
    //public void ClearRect(int x, int y, int width, int height) { }

    public void SetPixel(int x, int y, ConsoleColor color) {
        if (x >= 0 && y >= 0 && x < Width && y < Height) {
            Console.SetCursorPosition(left + x, top + y);
            var prev = Console.ForegroundColor;
            Console.ForegroundColor = color;
            Console.Write(Filled);
            Console.ForegroundColor = prev;
        }
    }

    public void SetPixel(int x, int y, bool filled) {
        if (x >= 0 && y >= 0 && x < Width && y < Height) {
            Console.SetCursorPosition(left + x, top + y);
            Console.Write(filled ? Filled : Empty);
        }
    }

    // TODO more drawing methods like DrawCircle, DrawOval, DrawPoint, etc etc

    public void DrawLine((int X, int Y) from, (int X, int Y) to) {
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