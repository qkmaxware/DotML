using System.Text;

namespace Qkmaxware.Terminal;

public abstract class ConsoleApp
{

    protected IElement? Root;

    public ConsoleApp() { }
    
    public ConsoleApp(IElement? root) { this.Root = root;  }

    public bool HasConsole() => Console.LargestWindowWidth != 0;

    public void RenderWhile(Predicate<ConsoleApp> condition)
    {
        var hasConsole = HasConsole();
        var width = hasConsole ? Console.BufferWidth - 1 : 320;
        CharBuffer buffer = new CharBuffer(width);
        Initialize();

        (int Left, int Top) start = hasConsole ? Console.GetCursorPosition() : (0, 0);
        StringBuilder rendered = new StringBuilder(buffer.Width * 10); // Some arbitrary default

        if (hasConsole)
            Console.CursorVisible = false;
        while (condition(this))
        {
            // Clear buffer and resize if the console changed width
            var newWidth = hasConsole ? Console.BufferWidth - 1 : 320;
            buffer.ClearAndSetWidth(newWidth);
            Graphics rootGraphics = new Graphics(buffer, new LayoutRect(0, 0, buffer.Width));

            // Read I/O
            if (hasConsole && Console.KeyAvailable)
            {
                OnKey(Console.ReadKey(intercept: true));
            }

            // Render and flush buffer to the console
            BeforeRender();
            Root?.Render(rootGraphics);

            buffer.Flush(rendered);

            if (hasConsole)
            {
                Console.SetCursorPosition(start.Left, start.Top);
                foreach (var chunk in rendered.GetChunks())
                {
                    Console.Out.Write(chunk.Span); // writes Span<char> directly
                }
            }
            AfterRender();
        }
        if (hasConsole)
            Console.CursorVisible = true;
    }

    public void RenderLoop()
    {
        RenderWhile(static (self) => true);
    }

    public void RenderOnce()
    {
        bool first = true;
        RenderWhile((self) => { if (first) { first = false; return true; } else { return false; } });
    }

    public virtual void Initialize() { }

    public virtual void BeforeRender() { }

    public virtual void OnKey(ConsoleKeyInfo key) { }

    public virtual void AfterRender() { }
}