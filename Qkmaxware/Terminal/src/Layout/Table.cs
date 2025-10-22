using System.Drawing;
using System.Reflection;
using System.Runtime.Versioning;

namespace Qkmaxware.Terminal.Layout;

// Example table
/*
┌──────┬──────┐
│Name  │ Age  │
├──────┼──────│
│John  | 23   |
|Jane  | 26   |
└──────┴──────┘
*/

public class Table<TRow> : IElement
{
    private List<PropertyInfo> columns;
    private IEnumerable<TRow>? Rows { get; set; }

    public Table()
    {
        this.columns = typeof(TRow).GetProperties(BindingFlags.Instance | BindingFlags.Public).ToList();
    }

    public Table(IEnumerable<TRow> rows): this()
    {
        this.Rows = rows;
    }

    const char TopLeft = '┌';
    const char TopMiddle = '─';
    const char TopT = '┬';
    const char LeftT = '├';
    const char RightT = '┤';
    const char MiddleT = '┼';
    const char BottomT = '┴';
    const char TopRight = '┐';
    const char LeftMiddle = '│';
    const char RightMiddle = '│';
    const char BottomLeft = '└';
    const char BottomMiddle = '─';
    const char BottomRight = '┘';

    public LayoutSize Render(Graphics graphics)
    {
        var columnWidth = (graphics.DrawingRegion.Width / columns.Count) - (columns.Count + 1);
        var realWidth = columnWidth * this.columns.Count + (columns.Count + 1);
        var right = realWidth - 1;

        // Columns
        graphics.Draw(new Point(0, 0), TopLeft);
        graphics.Draw(new Point(right, 0), TopRight);
        graphics.Draw(new Point(0, 1), LeftMiddle);
        graphics.Draw(new Point(right, 1), LeftMiddle);
        graphics.Draw(new Point(0, 2), LeftT);
        graphics.Draw(new Point(right, 2), RightT);
        int columnIndex = 0;
        foreach (var column in this.columns)
        {
            var columnStart = columnIndex * columnWidth + columnIndex + 1;
            if(columnIndex != 0)
            {
                graphics.Draw(new Point(columnStart - 1, 0), TopT);
                graphics.Draw(new Point(columnStart - 1, 1), LeftMiddle);
                graphics.Draw(new Point(columnStart - 1, 2), MiddleT);
            }
            
            for (var i = 0; i < columnWidth; i++)
            {
                graphics.Draw(new Point(columnStart + i, 0), TopMiddle);
                graphics.Draw(new Point(columnStart + i, 2), TopMiddle);
            }
            graphics.DrawClipped(new Point(columnStart, 1), column.Name);
            columnIndex++;
        }

        // Rows
        var rowIndex = 0;
        var consumedWidth = graphics.DrawingRegion.Width;
        var consumedHeight = 0;
        if (Rows is not null) {
            foreach (var row in Rows)
            {
                var rowOffset = 3 + rowIndex;
                graphics.Draw(new Point(0, rowOffset), LeftMiddle);
                graphics.Draw(new Point(right, rowOffset), RightMiddle);
                columnIndex = 0;
                foreach (var column in this.columns)
                {
                    var columnStart = columnIndex * columnWidth + columnIndex + 1;
                    if (columnIndex != 0)
                    {
                        graphics.Draw(new Point(columnStart - 1, rowOffset), LeftMiddle);
                    }
                    
                    var prop = column.GetValue(row)?.ToString() ?? string.Empty;
                    var region = new LayoutRect(
                        graphics.DrawingRegion.X + columnStart,
                        graphics.DrawingRegion.Y + rowOffset,
                        columnWidth - 1
                    );
                    graphics.WithRegion(region).DrawClipped(new Point(0, 0), prop);
                    columnIndex++;
                }
                rowIndex++;
                consumedHeight += 1;
            }
        }

        // Footer
        columnIndex = 0;
        var yFooter = 3 + consumedHeight;
        graphics.Draw(new Point(0, yFooter), BottomLeft);
        graphics.Draw(new Point(right, yFooter), BottomRight);
        foreach (var column in this.columns)
        {
            var columnStart = columnIndex * columnWidth + columnIndex + 1;
            if (columnIndex != 0)
            {
                graphics.Draw(new Point(columnStart - 1, yFooter), BottomT);
            }
            for (var i = 0; i < columnWidth; i++)
            {
                graphics.Draw(new Point(columnStart + i, yFooter), BottomMiddle);
            }
            columnIndex++;
        }

        return new LayoutSize(consumedWidth, consumedHeight + 3 + 1);
    }
}