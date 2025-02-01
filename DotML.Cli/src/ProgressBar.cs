namespace DotML.Cli;

public class ProgressBar {

    private int left;
    private int top;

    private int total_width;
    private int bar_width;

    public ProgressBar(int width) {
        width = Math.Max(4, width);
        total_width = width + 5;
        bar_width = width - 2;
        (left, top) = Console.GetCursorPosition();
        Update(0);
    }

    private char begin = '[';
    private char end = ']';
    private char filled = '-';
    private char empty = ' ';
    private char mark = '/';

    private double mark_location = 5;

    public void Mark(int amount, int total) {
        Mark((double)amount / (double)total);
    }

    public void Mark(double percent) {
        this.mark_location = Math.Max(0, percent);
    }

    public void Update(int amount, int total) {
        Update((double)amount / (double)total);
    }

    public void Update(double percent) {
        percent = Math.Clamp(percent, 0.0, 1.0);
        Console.SetCursorPosition(left, top);
        Console.Write(begin);
        for (var i = 0; i < bar_width; i++) {
            var i_percent = i / (double)(bar_width - 1);
            var next_i_percent = (i + 1) / (double)(bar_width - 1);
            if (mark_location >= i_percent && mark_location < next_i_percent) {
                Console.Write(mark);
            } else {
                if (percent >= i_percent) {
                    Console.Write(filled);
                } else {
                    Console.Write(empty);
                }
            }
        }
        Console.Write(end);
        // These last 5 characters are just a number
        Console.Write(' ');                     // 1 character spacer
        Console.Write((int)(percent * 100));    // 3 characters 0 to 100
        Console.Write('%');                     // 1 character uom
    }
}