using System.Text;

namespace Qkmaxware.Terminal;

public struct BufferedCharacter
{
    public char Character;
}

public class CharBuffer
{
    private BufferedCharacter[] buffer;
    public int Height { get; private set; }
    public int Width { get; private set; }

    public int Count => Width * Height;
    public int Capacity => buffer.Length;

    public ref BufferedCharacter this[int x, int y]
    {
        get
        {
            return ref buffer[y * Width + x];
        }
    }

    public CharBuffer(int width)
    {
        this.Height = 1;
        this.Width = Math.Max(0, width);
        this.buffer = new BufferedCharacter[Width];
    }

    public void Clear()
    {
        Array.Fill(this.buffer, new BufferedCharacter());
    }

    public void ClearAndSetWidth(int width)
    {
        width = Math.Max(0, width);
        var desired = this.Height * width;
        if (desired > Capacity)
        {
            this.buffer = new BufferedCharacter[this.Height * width];
        } else
        {
            Clear();
        }
        this.Width = width;
    }

    public void ClearAndSetHeight(int height)
    {
        height = Math.Max(0, height);
        var desired = height * this.Width;
        if (desired > Capacity)
        {
            this.buffer = new BufferedCharacter[height * this.Width];
        } else
        {
            Clear();
        }
        this.Height = height;
    }

    public void ClearAndResize(int width, int height)
    {
        width = Math.Max(0, width);
        height = Math.Max(0, height);
        var desired = height * width;
        if (desired > Capacity)
        {
            this.buffer = new BufferedCharacter[height * width];
        } else
        {
            Clear();
        }
        this.Height = height;
        this.Width = width;
    }

    public void AddRows(int rows = 1)
    {
        rows = Math.Max(0, rows);
        var nextHeight = Height + rows;

        var buffer = new BufferedCharacter[nextHeight * this.Width];
        Array.Copy(this.buffer, buffer, this.buffer.Length);
        this.buffer = buffer;
        this.Height = nextHeight;
    }

    public void EnsureHeight(int y)
    {
        if (y < Height)
            return;

        var delta = (y - Height) + 1;
        AddRows(delta);
    }

    public void Flush(StringBuilder sb)
    {
        sb.Clear();
        for (int row = 0, index = 0; row < this.Height; row++)
        {
            for (var col = 0; col < this.Width; col++, index++)
            {
                var character = this.buffer[index].Character;
                sb.Append(character == default(char) ? ' ' : character);
            }
            sb.AppendLine();
        }
    }
}